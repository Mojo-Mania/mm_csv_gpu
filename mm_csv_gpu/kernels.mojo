"""The kernels one CSV scan is made of.

Finding the fields in a CSV document is a scan in the parallel-prefix sense,
not just in the "walk the bytes" sense, and that is the whole difficulty. A
byte is a delimiter only if it is outside a quoted region, and whether it is
inside one depends on how many quotes came before it -- in the whole document,
not in this thread's sixty-four bytes. The CPU carries that state forward a
chunk at a time. Thousands of threads cannot.

So the state is computed rather than carried. Each chunk reports the parity of
its own quote count; an exclusive prefix sum over those parities says, for
every chunk, whether it begins inside a quoted region. That is two passes over
the document with a scan in between:

1. `analyse_kernel` -- per chunk: the quote parity, and the delimiter count
   **both ways**, for a chunk that starts outside a quoted region and for one
   that starts inside. Computing both is what lets this happen before the scan
   rather than after, and it costs one extra `popcount`.
2. the scan -- `scan_block_kernel`, `scan_totals_kernel`, `scan_apply_kernel`
   give an exclusive prefix sum over any length. Run over the parities, its
   low bit is each chunk's incoming quote state. Run again over the counts
   that state selects, it gives each chunk where to write.
3. `emit_kernel` -- recompute the masks, apply the now-known carry, and write
   one `UInt32` per field.

The masks are recomputed in the second pass rather than stored. Storing them
would cost 24 bytes for every 64 of document; recomputing costs a second read
of the document, which on this hardware is the cheaper of the two.

The CR of a CRLF is read straight from the document at `base - 1` rather than
carried between chunks, so a row ending across a chunk boundary needs no state
at all. The CPU package carries a flag for this and it is the piece of state
whose tests took longest to get right.
"""

from max.gpu import barrier, block_dim, block_idx, thread_idx
from max.gpu.memory import AddressSpace
from std.bit import count_trailing_zeros, pop_count
from std.memory import stack_allocation

comptime QUOTE = UInt8(ord('"'))
comptime LF = UInt8(ord("\n"))
comptime CR = UInt8(ord("\r"))
comptime COMMA = UInt8(ord(","))

comptime CHUNK = 64
"""Bytes one thread looks at: the width of the integer that carries one flag
per byte, which is what lets the quote analysis be integer arithmetic."""

comptime THREADS = 256
"""Threads per block, for every kernel here."""

comptime CRLF_BIT = UInt32(1) << 31
"""Set on an index entry whose delimiter was an LF with a CR in front."""

comptime OFFSET_MASK = (UInt32(1) << 31) - 1
"""Written out rather than as `UInt32.MAX >> 1`; see the CPU package for what
`Int(UInt32.MAX)` does."""

comptime _WEIGHTS = SIMD[DType.uint16, 16](
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
)


@always_inline
def _movemask(
    m0: SIMD[DType.bool, 16],
    m1: SIMD[DType.bool, 16],
    m2: SIMD[DType.bool, 16],
    m3: SIMD[DType.bool, 16],
) -> UInt64:
    """Packs sixty-four lane flags into a `UInt64`, one bit each.

    `std.memory.pack_bits` is what the CPU package uses and it cannot be used
    here: Metal's shader compiler does not survive it -- the pipeline build
    fails with `XPC_ERROR_CONNECTION_INTERRUPTED` rather than a diagnostic.
    Multiplying the comparison result by the bit weights and reducing gives
    the same integer in four operations instead of sixteen shifts.
    """
    return (
        UInt64((m0.cast[DType.uint16]() * _WEIGHTS).reduce_add())
        | (UInt64((m1.cast[DType.uint16]() * _WEIGHTS).reduce_add()) << 16)
        | (UInt64((m2.cast[DType.uint16]() * _WEIGHTS).reduce_add()) << 32)
        | (UInt64((m3.cast[DType.uint16]() * _WEIGHTS).reduce_add()) << 48)
    )


@always_inline
def _prefix_xor(var bits: UInt64) -> UInt64:
    """Returns, for each bit, the XOR of every bit at or below it.

    Applied to the quote positions this answers "is this byte inside a quoted
    region?" for all sixty-four bytes at once, assuming the chunk begins
    outside one. The carry-less multiply the CPU package uses is an ARM
    instruction; six shifts is what runs everywhere, this device included.
    """
    bits ^= bits << 1
    bits ^= bits << 2
    bits ^= bits << 4
    bits ^= bits << 8
    bits ^= bits << 16
    bits ^= bits << 32
    return bits


@fieldwise_init
struct Masks(TrivialRegisterPassable):
    """What one chunk of sixty-four bytes says, before the carry is known."""

    var quotes: UInt64
    """Quote characters."""

    var delimiters: UInt64
    """Separators and line feeds, before quoted ones are removed."""

    var crlf: UInt64
    """Line feeds with a carriage return in front, before the same."""


@always_inline
def chunk_masks[
    separator: UInt8
](data: Pointer[UInt8, MutUntrackedOrigin], base: Int, valid: Int) -> Masks:
    """Reads sixty-four bytes at `base` and reduces them to three masks.

    `valid` is how many of those bytes are inside the document; the rest are
    padding and are masked off. The buffer must hold `CHUNK` bytes past the
    document so the loads themselves are always in bounds.

    Parameters:
        separator: The byte between fields.

    Args:
        data: The document.
        base: Where this chunk starts.
        valid: Document bytes remaining from `base`.

    Returns:
        The chunk's quote, delimiter and CRLF masks.
    """
    var quote_v = SIMD[DType.uint8, 16](QUOTE)
    var sep_v = SIMD[DType.uint8, 16](separator)
    var lf_v = SIMD[DType.uint8, 16](LF)
    var cr_v = SIMD[DType.uint8, 16](CR)

    var b0 = data.unsafe_offset(base).unsafe_load[width=16]()
    var b1 = data.unsafe_offset(base + 16).unsafe_load[width=16]()
    var b2 = data.unsafe_offset(base + 32).unsafe_load[width=16]()
    var b3 = data.unsafe_offset(base + 48).unsafe_load[width=16]()

    var quotes = _movemask(
        b0.eq(quote_v), b1.eq(quote_v), b2.eq(quote_v), b3.eq(quote_v)
    )
    var seps = _movemask(b0.eq(sep_v), b1.eq(sep_v), b2.eq(sep_v), b3.eq(sep_v))
    var lfs = _movemask(b0.eq(lf_v), b1.eq(lf_v), b2.eq(lf_v), b3.eq(lf_v))
    var crs = _movemask(b0.eq(cr_v), b1.eq(cr_v), b2.eq(cr_v), b3.eq(cr_v))

    # The CR that pairs with a line feed in lane zero is the last byte of the
    # previous chunk, and reading it from the document is why no carry is
    # needed for CRLF.
    var carried = UInt64(0)
    if base > 0 and data[unsafe_offset=base - 1] == CR:
        carried = UInt64(1)
    var cr_shifted = (crs << 1) | carried

    var keep = UInt64.MAX
    if valid < CHUNK:
        keep = (UInt64(1) << UInt64(valid)) - 1 if valid > 0 else UInt64(0)

    return Masks(quotes & keep, (seps | lfs) & keep, (lfs & cr_shifted) & keep)


def analyse_kernel[
    separator: UInt8
](
    data: Pointer[UInt8, MutUntrackedOrigin],
    parity: Pointer[UInt32, MutUntrackedOrigin],
    outside: Pointer[UInt32, MutUntrackedOrigin],
    inside: Pointer[UInt32, MutUntrackedOrigin],
    length: Int32,
    chunks: Int32,
):
    """Reports each chunk's quote parity and both of its delimiter counts.

    The counts are given for both possible incoming quote states because the
    scan that decides which one applies has not run yet. A chunk that starts
    outside a quoted region keeps the delimiters where `prefix_xor` is zero; a
    chunk that starts inside keeps exactly the others, because entering inside
    inverts the mask.

    Parameters:
        separator: The byte between fields.

    Args:
        data: The document, padded by `CHUNK` bytes.
        parity: Output, 1 if the chunk holds an odd number of quotes.
        outside: Output, delimiters if the chunk starts outside a quote.
        inside: Output, delimiters if it starts inside one.
        length: Document bytes.
        chunks: How many chunks that is.
    """
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if tid >= Int(chunks):
        return
    var base = tid * CHUNK
    var m = chunk_masks[separator](data, base, Int(length) - base)
    var pxor = _prefix_xor(m.quotes)
    parity[unsafe_offset=tid] = UInt32(pop_count(m.quotes) & 1)
    outside[unsafe_offset=tid] = UInt32(pop_count(m.delimiters & ~pxor))
    inside[unsafe_offset=tid] = UInt32(pop_count(m.delimiters & pxor))


def select_counts_kernel(
    carry: Pointer[UInt32, MutUntrackedOrigin],
    outside: Pointer[UInt32, MutUntrackedOrigin],
    inside: Pointer[UInt32, MutUntrackedOrigin],
    counts: Pointer[UInt32, MutUntrackedOrigin],
    chunks: Int32,
):
    """Picks each chunk's delimiter count now that its carry is known.

    Args:
        carry: Exclusive prefix sum of the parities; the low bit is the state.
        outside: Counts for a chunk starting outside a quoted region.
        inside: Counts for one starting inside.
        counts: Output.
        chunks: How many chunks.
    """
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if tid >= Int(chunks):
        return
    var which = carry[unsafe_offset=tid] & 1
    counts[unsafe_offset=tid] = (
        inside[unsafe_offset=tid] if which != 0 else outside[unsafe_offset=tid]
    )


def emit_kernel[
    separator: UInt8
](
    data: Pointer[UInt8, MutUntrackedOrigin],
    carry: Pointer[UInt32, MutUntrackedOrigin],
    offsets: Pointer[UInt32, MutUntrackedOrigin],
    index: Pointer[UInt32, MutUntrackedOrigin],
    length: Int32,
    chunks: Int32,
):
    """Writes one entry per field: the delimiter offset, CRLF flag in bit 31.

    Parameters:
        separator: The byte between fields.

    Args:
        data: The document, padded by `CHUNK` bytes.
        carry: Each chunk's incoming quote state in its low bit.
        offsets: Where each chunk's entries begin.
        index: Output.
        length: Document bytes.
        chunks: How many chunks.
    """
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if tid >= Int(chunks):
        return
    var base = tid * CHUNK
    var m = chunk_masks[separator](data, base, Int(length) - base)
    var pxor = _prefix_xor(m.quotes)
    var quoted = pxor if (carry[unsafe_offset=tid] & 1) == 0 else ~pxor
    var delimiters = m.delimiters & ~quoted
    var crlf = m.crlf & ~quoted

    var cursor = Int(offsets[unsafe_offset=tid])
    while delimiters != 0:
        var lane = Int(count_trailing_zeros(delimiters))
        delimiters &= delimiters - 1
        var flag = ((crlf >> UInt64(lane)) & 1).cast[DType.uint32]() << 31
        index[unsafe_offset=cursor] = UInt32(base + lane) | flag
        cursor += 1
