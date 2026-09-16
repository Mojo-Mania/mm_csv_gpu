"""Tests for the GPU CSV scan.

Everything here is the same question asked in different shapes: does the GPU
index agree, entry for entry, with a scalar walk of the same document? The
scalar walk is fifteen lines and obviously right, which is the point.

The cases that matter are the ones where a chunk cannot answer for itself.
A quoted region spanning chunks, a CRLF split across them, and -- unique to
this implementation -- documents large enough that the quote parity has to
travel through all three levels of the prefix scan: within a block, across
blocks, and across tiles of blocks.
"""

from max.gpu.host import DeviceContext
from mm_csv_gpu import GpuCsvTable
from mm_csv_gpu.kernels import CR, CRLF_BIT, LF, OFFSET_MASK, QUOTE
from std.pathlib import Path
from std.testing import TestSuite, assert_equal, assert_true

comptime SCRATCH = "/tmp/mm_csv_gpu_test.csv"


def _reference(text: String, separator: UInt8) -> List[UInt32]:
    """A scalar walk producing the index the GPU should produce."""
    var entries = List[UInt32]()
    var ptr = text.unsafe_ptr()
    var n = text.byte_length()
    var in_quotes = False
    for i in range(n):
        var byte = ptr[unsafe_offset=i]
        if byte == QUOTE:
            in_quotes = not in_quotes
        elif in_quotes:
            pass
        elif byte == separator:
            entries.append(UInt32(i))
        elif byte == LF:
            var flag = UInt32(0)
            if i > 0 and ptr[unsafe_offset=i - 1] == CR:
                flag = CRLF_BIT
            entries.append(UInt32(i) | flag)
    if n > 0 and ptr[unsafe_offset=n - 1] != LF:
        entries.append(UInt32(n))
    return entries^


def _assert_matches(
    ctx: DeviceContext,
    text: String,
    context: String,
    separator: UInt8 = UInt8(ord(",")),
) raises:
    """Writes `text`, scans it on the GPU, and compares every entry."""
    with open(SCRATCH, "w") as f:
        f.write(text)
    var expected = _reference(text, separator)
    var table = GpuCsvTable(ctx, Path(SCRATCH))
    assert_equal(len(table), len(expected), String(context, ": field count"))
    for i in range(len(expected)):
        assert_equal(
            Int(table._index[i]),
            Int(expected[i]),
            String(context, ": entry ", i),
        )


def test_rfc_shapes() raises:
    var ctx = DeviceContext()
    _assert_matches(ctx, "aaa,bbb,ccc\r\nzzz,yyy,xxx\r\n", "CRLF rows")
    _assert_matches(ctx, "aaa,bbb,ccc\r\nzzz,yyy,xxx", "no trailing break")
    _assert_matches(ctx, "a,b\nc,d\n", "bare LF")
    _assert_matches(ctx, 'a,"b,c",d\r\n', "separator inside quotes")
    _assert_matches(ctx, 'a,"line\r\nbreak",c\r\n', "break inside quotes")
    _assert_matches(ctx, 'a,"he said ""hi""",c\r\n', "doubled quote")
    _assert_matches(ctx, ",,\r\n,,\r\n", "empty fields")
    _assert_matches(ctx, "é,ü\r\nnaïve,日本\r\n", "non-ASCII")


def test_empty_document() raises:
    var ctx = DeviceContext()
    with open(SCRATCH, "w") as f:
        f.write("")
    var table = GpuCsvTable(ctx, Path(SCRATCH))
    assert_equal(len(table), 0, "an empty document has no fields")
    assert_equal(table.row_count(), 0, "and no rows")


def test_every_length() raises:
    """Each tail alignment, which is the lane mask in `chunk_masks`."""
    var ctx = DeviceContext()
    var text = String()
    for i in range(200):
        _assert_matches(
            ctx, String(text, "z"), String("length ", text.byte_length() + 1)
        )
        text += "a,b\r\n" if i % 7 == 6 else "q"


def test_quotes_across_chunks() raises:
    """The carry the whole design exists for: a quoted region spanning
    chunks, walked across every lane of the boundary."""
    var ctx = DeviceContext()
    var trailer = String()
    for _ in range(20):
        trailer += "e,f,g\r\n"
    for pad in range(0, 140):
        var filler = String()
        for _ in range(pad):
            filler += "y"
        _assert_matches(
            ctx,
            String('a,"', filler, ',still one field",c\r\n', trailer),
            String("quote pad ", pad),
        )


def test_crlf_across_chunks() raises:
    """A CR at the end of one chunk and its LF at the start of the next.

    `chunk_masks` reads the byte before the chunk rather than carrying a flag,
    so this is the test that the read is there and correct.
    """
    var ctx = DeviceContext()
    var trailer = String()
    for _ in range(20):
        trailer += "c,d\r\n"
    for pad in range(0, 140):
        var filler = String()
        for _ in range(pad):
            filler += "x"
        _assert_matches(
            ctx,
            String(filler, ",b\r\n", trailer),
            String("CRLF pad ", pad),
        )


def test_across_blocks() raises:
    """More than one block, so the parity crosses the second scan level."""
    var ctx = DeviceContext()
    var text = String()
    for i in range(4000):
        text += "aaa,bbb,ccc\r\n" if i % 3 != 0 else 'aaa,"b,b",ccc\r\n'
    _assert_matches(ctx, text, "4000 rows, many blocks")


def test_across_scan_tiles() raises:
    """More than `THREADS` blocks, so `scan_totals_kernel` has to loop.

    A block covers 16 KiB, so the totals scan only tiles past 4 MiB. Below
    that this whole level of the scan is a single pass and never tested.
    """
    var ctx = DeviceContext()
    var text = String()
    for i in range(400_000):
        text += "aaa,bbb,ccc\r\n" if i % 5 != 0 else 'aaa,"b,b",ccc\r\n'
    assert_true(
        text.byte_length() > 4 * 1024 * 1024,
        "the document has to be large enough to tile the totals scan",
    )
    _assert_matches(ctx, text, "5 MB, many tiles of blocks")


def test_unterminated_quote() raises:
    """A quote that never closes: every delimiter after it is data."""
    var ctx = DeviceContext()
    _assert_matches(ctx, 'a,b\r\n"c,d\r\ne,f\r\n', "unterminated quote")


def test_custom_separator() raises:
    var ctx = DeviceContext()
    var text = String("a\tb\r\nc\td\r\n")
    with open(SCRATCH, "w") as f:
        f.write(text)
    var table = GpuCsvTable[UInt8(ord("\t"))](ctx, Path(SCRATCH))
    assert_equal(table.column_count, 2, "tab separated columns")
    assert_equal(String(table.field(1, 0)), "c", "tab separated value")


def test_fields_read_back() raises:
    var ctx = DeviceContext()
    with open(SCRATCH, "w") as f:
        f.write('name,note\r\nAda,"first, and foremost"\r\n')
    var table = GpuCsvTable(ctx, Path(SCRATCH))
    assert_equal(table.column_count, 2, "columns")
    assert_equal(table.row_count(), 2, "rows")
    assert_equal(String(table.field(0, 0)), "name", "header")
    assert_equal(
        String(table.field(1, 1)),
        '"first, and foremost"',
        "raw quoted field",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
