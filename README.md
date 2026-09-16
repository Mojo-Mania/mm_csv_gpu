# mm_csv_gpu

Finding the fields of a CSV document on the GPU, in [Mojo](https://mojolang.org).

```mojo
from max.gpu.host import DeviceContext
from mm_csv_gpu import GpuCsvTable
from std.pathlib import Path

var ctx = DeviceContext()
var table = GpuCsvTable(ctx, Path("data/large.csv"))
print(table.row_count(), table.column_count)
print(table.field(1, 3))            # borrowed, nothing copied
```

This is the GPU counterpart to [mm_csv](https://github.com/Mojo-Mania/mm_csv),
the way [mm_radix_sort_gpu](https://github.com/Mojo-Mania/mm_radix_sort_gpu) is
to mm_radix_sort. It produces the same index — one `UInt32` per field holding
the offset of the delimiter that closed it, with bit 31 set when that delimiter
was an LF preceded by a CR.

## Read this before reaching for it

**The scan is fast and the surrounding work is not.** On a 253 MB document the
three kernels take 3.5 ms between them, which is 72 GiB/s against the CPU
library's 9.4. Getting to and from them takes another 57 ms, almost all of it
allocating a pinned host buffer the size of the document and faulting its pages
in on first touch. **End to end, one-shot, this currently loses to the CPU.**
The numbers are below and the reason is in
[`docs/improvements.md`](docs/improvements.md), along with the fix, which is to
stop allocating per parse.

**Unified memory is doing a lot of work here.** These numbers are Apple silicon,
where the upload is a 185 GiB/s copy between two regions of the same RAM. On a
discrete card the document crosses PCIe and the arithmetic is different.

**A kernel cannot read a pinned host buffer.** It reads zeros, silently, and a
timing taken that way looks wonderful. The pinned buffer here is a *source* for
the upload, which is what makes the upload 185 GiB/s instead of 6.

## How it works

A byte is a delimiter only if it is outside a quoted region, and whether it is
inside one depends on every quote before it in the whole document. The CPU
carries that state forward a chunk at a time. Thousands of threads cannot, so
the state is computed instead of carried:

1. **analyse** — one thread per 64 bytes. Each reports the parity of its own
   quote count, and its delimiter count **both ways**: the count if the chunk
   begins outside a quoted region, and the count if it begins inside. Computing
   both is what lets this run before the answer is known, and it costs one
   extra `popcount`.
2. **scan** — an exclusive prefix sum over the parities. Its low bit is each
   chunk's incoming quote state, because an exclusive prefix *XOR* over one-bit
   values is the low bit of an exclusive prefix *sum* over them.
3. **select** — each chunk picks the count that its now-known carry makes true.
4. **scan** — again, over those counts, giving each chunk where to write.
5. **emit** — recompute the masks, apply the carry, write one entry per field.

Two passes over the document, not three: the masks are recomputed in the emit
rather than stored, because storing them costs 24 bytes for every 64 of
document and recomputing costs a second read, which is cheaper here.

The CR of a CRLF is read straight from the document at `base - 1` rather than
carried between chunks, so a row ending across a chunk boundary needs no state.

## Install

```toml
[dependencies]
mm_csv_gpu = { git = "https://github.com/Mojo-Mania/mm_csv_gpu.git" }
```

## API

| | |
| --- | --- |
| `GpuCsvTable[separator](ctx, path)` | Reads and scans `path`. |
| `.column_count` | Fields in the first row. |
| `.row_count()` | Rows. |
| `len(table)` | Fields, in total. |
| `.field(row, column)` | The raw field, borrowed. Raises if out of range. |
| `.is_ragged()` | Whether some row has a different field count. |

`separator` is a compile-time `UInt8` and defaults to `,`. Documents must be
under 2 GiB: field positions are indexed with 31 bits and a flag.

Fields come back raw — a quoted value keeps its quotes and its doubled quotes.
Undoing RFC 4180 escaping is `mm_csv`'s `get`, and is not here.

## Performance

Apple M4 Max, `-D ASSERT=none`, best of five. Reproduce with
`bash data/setup.sh && pixi run bench`. `small.csv` is byte for byte mm_csv's
`no_escaping.csv` — same generator, same seed, same columns — so the CPU
comparison is that library's measured number on the same bytes rather than a
reimplementation here.

**`large.csv`** — 253 MB, 2 800 001 rows, 8 columns, 22 400 008 fields:

| | ms | GiB/s | ns/field |
| --- | ---: | ---: | ---: |
| **total, file to index** | **60.6** | 3.89 | 2.70 |
| read into pinned memory | 13.4 | 17.6 | 0.60 |
| upload | 1.3 | **184.9** | 0.06 |
| analyse | 1.2 | **192.5** | 0.05 |
| one prefix scan (of two) | 0.6 | **376.4** | 0.03 |
| emit | 1.1 | **222.7** | 0.05 |
| index back | 0.5 | **522.4** | 0.02 |

**`small.csv`** — 23 MB, 255 361 rows, 8 columns, 2 042 888 fields:

| | ms | GiB/s |
| --- | ---: | ---: |
| total, file to index | 6.3 | 3.42 |
| the five kernels together | 0.7 | ~30 |
| — mm_csv, CPU, parse only | **2.2** | **9.42** |

Two things to read off these.

**The kernels are not the problem.** All five together are 3.5 ms on the 253 MB
document — analyse and emit each read the whole document at about 200 GiB/s,
and the two scans are almost free. That is roughly seven times the CPU
library's parse rate on the same shape of document.

**The allocation is.** Of the 60.6 ms total, about 42 is neither file I/O nor
compute: a fresh 253 MB pinned host buffer costs 4.5 ms to allocate and 22.5 ms
to fault in on first touch, and the index buffers cost their own. A caller that
scans one document and exits pays all of it. A caller that scans many should
not, and the type as written gives it no way to avoid it — see
[`docs/improvements.md`](docs/improvements.md).

## Development

```bash
pixi run test      # 10 tests, against a scalar reference
pixi run bench     # needs `bash data/setup.sh` first
pixi run main      # the example
pixi run format
pixi run docs
```

The tests compare every index entry against a fifteen-line scalar walk, over
documents built to break the parts a chunk cannot answer for itself: a quoted
region spanning chunks, a CRLF split across them, every tail alignment, and
documents large enough to push the quote parity through all three levels of the
prefix scan — within a block, across blocks, and across tiles of blocks.

## License

MIT. See [LICENSE](LICENSE).
