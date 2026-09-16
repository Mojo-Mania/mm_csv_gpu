"""`GpuCsvTable` -- the host side: buffers, launches, and reading the result.

The work is five launches and one synchronisation:

    analyse  ->  scan (parities)  ->  select  ->  scan (counts)  ->  emit

with a single readback in the middle, because the index cannot be allocated
until the second scan has said how many fields there are. Everything else is
enqueued and never waited on.

The document is read from the file straight into a **pinned** host buffer and
uploaded from there. That matters more than it looks: the same upload from
ordinary allocated memory runs at about 6 GiB/s, and from pinned memory at
190. The pinned buffer is a fast source for the copy, not memory the device
can read -- a kernel handed a host buffer pointer reads zeros, silently.
"""

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.pathlib import Path

from .kernels import (
    CHUNK,
    COMMA,
    CRLF_BIT,
    LF,
    OFFSET_MASK,
    THREADS,
    analyse_kernel,
    emit_kernel,
    select_counts_kernel,
)
from .scan import scan_apply_kernel, scan_block_kernel, scan_totals_kernel


struct GpuCsvTable[separator: UInt8 = COMMA](Movable, Sized):
    """A CSV document whose fields were found on the GPU.

    Parameters:
        separator: The byte between fields. `,` by default; `\\t` for TSV.
    """

    var _text: HostBuffer[DType.uint8]
    """The document, pinned. Fields borrow from it."""

    var _index: HostBuffer[DType.uint32]
    """One entry per field: the offset of the delimiter that closed it, with
    bit 31 set when that delimiter was an LF preceded by a CR."""

    var _length: Int
    var _count: Int

    var column_count: Int
    """Fields in the first row, which RFC 4180 says every row must match."""

    def __init__(out self, ctx: DeviceContext, path: Path) raises:
        """Reads `path` and finds every field in it.

        Args:
            ctx: The device to run on.
            path: The document to read.

        Raises:
            Error: If the file cannot be read, is 2 GiB or larger, or a device
                allocation or launch fails.
        """
        var length = Int(path.stat().st_size)
        # Written out rather than `Int(UInt32.MAX >> 1)`, which is a trap in
        # the CPU package's history.
        comptime MAX_LENGTH = (1 << 31) - 1
        if length > MAX_LENGTH:
            raise Error(
                "GpuCsvTable indexes fields with 31 bits and cannot take a"
                " document of 2 GiB or more"
            )

        self._length = length
        self._count = 0
        self.column_count = 0

        # Padded by a chunk so the four loads in `chunk_masks` are always in
        # bounds; the padding is zeroed and masked off by `valid`.
        self._text = ctx.enqueue_create_host_buffer[DType.uint8](length + CHUNK)
        self._index = ctx.enqueue_create_host_buffer[DType.uint32](1)
        ctx.synchronize()

        if length == 0:
            return

        with open(path, "r") as f:
            var got = f.read(
                Span[UInt8, MutUntrackedOrigin](
                    unsafe_ptr=self._text.unsafe_ptr(), length=length
                )
            )
            if got != length:
                raise Error("short read: got ", got, " of ", length, " bytes")
        for i in range(length, length + CHUNK):
            self._text[i] = UInt8(0)

        var chunks = (length + CHUNK - 1) // CHUNK
        var blocks = (chunks + THREADS - 1) // THREADS

        var document = ctx.enqueue_create_buffer[DType.uint8](length + CHUNK)
        document.enqueue_copy_from(
            Span[UInt8, MutUntrackedOrigin](
                unsafe_ptr=self._text.unsafe_ptr(), length=length + CHUNK
            )
        )

        # Both scans run in place: `scan_block_kernel` reads its value into a
        # register before writing, so input and output may be the same array.
        # `select_counts_kernel` cannot do the same -- passing one buffer as
        # two of its arguments is rejected as aliasing -- so `counts` is its
        # own array.
        var carry = ctx.enqueue_create_buffer[DType.uint32](chunks)
        var outside = ctx.enqueue_create_buffer[DType.uint32](chunks)
        var inside = ctx.enqueue_create_buffer[DType.uint32](chunks)
        var counts = ctx.enqueue_create_buffer[DType.uint32](chunks)
        var block_totals = ctx.enqueue_create_buffer[DType.uint32](blocks)
        var block_offsets = ctx.enqueue_create_buffer[DType.uint32](blocks)
        var grand = ctx.enqueue_create_buffer[DType.uint32](1)
        ctx.synchronize()

        ctx.enqueue_function[analyse_kernel[Self.separator]](
            document.unsafe_ptr(),
            carry.unsafe_ptr(),
            outside.unsafe_ptr(),
            inside.unsafe_ptr(),
            Int32(length),
            Int32(chunks),
            grid_dim=blocks,
            block_dim=THREADS,
        )

        Self._scan(
            ctx, carry, block_totals, block_offsets, grand, chunks, blocks
        )

        ctx.enqueue_function[select_counts_kernel](
            carry.unsafe_ptr(),
            outside.unsafe_ptr(),
            inside.unsafe_ptr(),
            counts.unsafe_ptr(),
            Int32(chunks),
            grid_dim=blocks,
            block_dim=THREADS,
        )

        Self._scan(
            ctx, counts, block_totals, block_offsets, grand, chunks, blocks
        )

        # The one place the host has to wait: the index cannot be sized until
        # the scan has totalled the counts. `map_to_host` does not wait for
        # enqueued work by itself -- without this the total read back is
        # whatever the allocation happened to hold, which was zero.
        ctx.synchronize()
        var fields: Int
        with grand.map_to_host() as host:
            fields = Int(host[0])

        var trailing = 1 if self._text[length - 1] != LF else 0
        self._count = fields + trailing

        var index = ctx.enqueue_create_buffer[DType.uint32](self._count + 1)
        ctx.enqueue_function[emit_kernel[Self.separator]](
            document.unsafe_ptr(),
            carry.unsafe_ptr(),
            counts.unsafe_ptr(),
            index.unsafe_ptr(),
            Int32(length),
            Int32(chunks),
            grid_dim=blocks,
            block_dim=THREADS,
        )

        self._index = ctx.enqueue_create_host_buffer[DType.uint32](
            self._count + 1
        )
        ctx.synchronize()
        index.enqueue_copy_to(
            Span[UInt32, MutUntrackedOrigin](
                unsafe_ptr=self._index.unsafe_ptr(), length=self._count + 1
            )
        )
        ctx.synchronize()

        # A document not ending in a line break leaves its last field open, and
        # the end of the text closes it.
        if trailing == 1:
            self._index[fields] = UInt32(length)

        # The first unquoted line feed closes the first row. Nothing in the
        # index says which delimiters were line feeds, so this asks the
        # document -- once, not per field.
        for i in range(self._count):
            var raw = self._index[i]
            if self._text[Int(raw & OFFSET_MASK)] == LF:
                self.column_count = i + 1
                break
        if self.column_count == 0:
            self.column_count = self._count

    @staticmethod
    def _scan(
        ctx: DeviceContext,
        values: DeviceBuffer[DType.uint32],
        block_totals: DeviceBuffer[DType.uint32],
        block_offsets: DeviceBuffer[DType.uint32],
        grand: DeviceBuffer[DType.uint32],
        n: Int,
        blocks: Int,
    ) raises:
        """Turns `values` into its own exclusive prefix sum, in place."""
        ctx.enqueue_function[scan_block_kernel](
            values.unsafe_ptr(),
            values.unsafe_ptr(),
            block_totals.unsafe_ptr(),
            Int32(n),
            grid_dim=blocks,
            block_dim=THREADS,
        )
        ctx.enqueue_function[scan_totals_kernel](
            block_totals.unsafe_ptr(),
            block_offsets.unsafe_ptr(),
            grand.unsafe_ptr(),
            Int32(blocks),
            grid_dim=1,
            block_dim=THREADS,
        )
        ctx.enqueue_function[scan_apply_kernel](
            values.unsafe_ptr(),
            block_offsets.unsafe_ptr(),
            Int32(n),
            grid_dim=blocks,
            block_dim=THREADS,
        )

    def __len__(self) -> Int:
        """Returns the number of fields the document holds.

        Returns:
            Every field in every row, including empty ones.
        """
        return self._count

    def row_count(self) -> Int:
        """Returns the number of rows.

        Returns:
            Fields divided by columns, or 0 for an empty document.
        """
        if self.column_count == 0:
            return 0
        return self._count // self.column_count

    def is_ragged(self) -> Bool:
        """Returns whether some row has a different field count from the first.

        Returns:
            True if the field count is not a whole number of rows.
        """
        if self.column_count == 0:
            return self._count != 0
        return self._count % self.column_count != 0

    @always_inline
    def _bounds(self, index: Int) -> Tuple[Int, Int]:
        """Returns the half-open byte range of the field at `index`."""
        var start = 0
        if index != 0:
            start = Int(self._index[index - 1] & OFFSET_MASK) + 1
        var raw = self._index[index]
        var end = Int(raw & OFFSET_MASK) - Int(raw >> 31)
        return (start, end)

    def field[
        origin: ImmOrigin, //
    ](ref[origin] self, row: Int, column: Int) raises -> StringSlice[origin]:
        """Returns a field exactly as it appears in the document.

        Parameters:
            origin: The origin of the borrow.

        Args:
            row: Zero-based row index.
            column: Zero-based column index.

        Raises:
            Error: If `row` or `column` is out of range.

        Returns:
            The bytes between the delimiters, borrowed.
        """
        if column < 0 or column >= self.column_count:
            raise Error(
                "column ", column, " is outside 0..<", self.column_count
            )
        var index = row * self.column_count + column
        if row < 0 or index >= self._count:
            raise Error("row ", row, " is outside 0..<", self.row_count())
        var start: Int
        var end: Int
        start, end = self._bounds(index)
        return StringSlice(
            unsafe_from_utf8=Span[UInt8, origin](
                unsafe_ptr=self._text.unsafe_ptr()
                .unsafe_offset(start)
                .as_imm()
                .unsafe_origin_cast[origin](),
                length=end - start,
            )
        )
