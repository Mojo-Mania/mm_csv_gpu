"""`GpuCsvScanner` -- the buffers, kept, so a second document is cheap.

Scanning a document needs a pinned host buffer its size, a device buffer its
size, four arrays of one `UInt32` per sixty-four bytes, and an index at both
ends. Allocating that per document is what made the first version of this
library slower than the CPU: a fresh 253 MB pinned buffer costs 4.5 ms to
allocate and **22.5 ms to fault in on first touch**, against 3.5 ms for all
five kernels put together.

So the buffers live here and are reused. They grow when a document does not
fit and never shrink, which means a scanner settles at the size of the largest
document it has seen. Fields borrow from the scanner's own memory, so they are
valid until the next `scan` -- that is the bargain this type makes, and
`GpuCsvTable` is the one-document wrapper for callers who would rather not
think about it.

The pipeline is five launches and one synchronisation:

    analyse -> scan (parities) -> select -> scan (counts) -> emit

with the readback in the middle only because the index cannot be sized until
the second scan has said how many fields there are.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.pathlib import Path

from .kernels import (
    CHUNK,
    COMMA,
    LF,
    OFFSET_MASK,
    THREADS,
    analyse_kernel,
    emit_kernel,
    select_counts_kernel,
)
from .scan import scan_apply_kernel, scan_block_kernel, scan_totals_kernel

comptime UPLOAD_SLICE = 16 * 1024 * 1024
"""Bytes read before the upload of what was read is enqueued.

Reading the file runs at about 17.7 GiB/s and the upload at 185, so the
upload is the smaller of the two and can hide behind the next read. Enqueued
work does not block, so slicing the loop is the whole trick. Measured on a
253 MB document: 15.8 ms in one slice, 14.9 at sixteen or more, and flat
after that. A document under this size is one slice and unchanged."""

comptime MAX_LENGTH = (1 << 31) - 1
"""Field positions are 31 bits and a flag. Written out rather than as
`Int(UInt32.MAX >> 1)`; see mm_csv for what `Int(UInt32.MAX)` does."""


struct GpuCsvScanner[separator: UInt8 = COMMA](Movable, Sized):
    """A reusable CSV scanner. Holds its scratch; scan as many documents as
    you like through one of these.

    Parameters:
        separator: The byte between fields. `,` by default; `\\t` for TSV.
    """

    var _text: HostBuffer[DType.uint8]
    var _document: DeviceBuffer[DType.uint8]
    var _carry: DeviceBuffer[DType.uint32]
    var _outside: DeviceBuffer[DType.uint32]
    var _inside: DeviceBuffer[DType.uint32]
    var _counts: DeviceBuffer[DType.uint32]
    var _block_totals: DeviceBuffer[DType.uint32]
    var _block_offsets: DeviceBuffer[DType.uint32]
    var _grand: DeviceBuffer[DType.uint32]
    var _index_device: DeviceBuffer[DType.uint32]
    var _index: HostBuffer[DType.uint32]

    var _bytes_capacity: Int
    var _chunks_capacity: Int
    var _blocks_capacity: Int
    var _index_capacity: Int

    var _length: Int
    var _count: Int

    var column_count: Int
    """Fields in the first row of the document last scanned."""

    def __init__(out self, ctx: DeviceContext, capacity: Int = 0) raises:
        """Builds a scanner, optionally sized for documents up to `capacity`.

        Sizing it up front is the point of the type. A scanner built with
        `capacity=0` allocates on the first `scan` and pays there instead.

        Args:
            ctx: The device to run on.
            capacity: Document bytes to make room for now.

        Raises:
            Error: If `capacity` is 2 GiB or more, or an allocation fails.
        """
        if capacity > MAX_LENGTH:
            raise Error(
                "GpuCsvScanner indexes fields with 31 bits and cannot take a"
                " document of 2 GiB or more"
            )
        self._length = 0
        self._count = 0
        self.column_count = 0
        self._bytes_capacity = 0
        self._chunks_capacity = 0
        self._blocks_capacity = 0
        self._index_capacity = 0

        # Everything starts empty and `_reserve` does the real work, so there
        # is one growth path rather than two.
        self._text = ctx.enqueue_create_host_buffer[DType.uint8](1)
        self._document = ctx.enqueue_create_buffer[DType.uint8](1)
        self._carry = ctx.enqueue_create_buffer[DType.uint32](1)
        self._outside = ctx.enqueue_create_buffer[DType.uint32](1)
        self._inside = ctx.enqueue_create_buffer[DType.uint32](1)
        self._counts = ctx.enqueue_create_buffer[DType.uint32](1)
        self._block_totals = ctx.enqueue_create_buffer[DType.uint32](1)
        self._block_offsets = ctx.enqueue_create_buffer[DType.uint32](1)
        self._grand = ctx.enqueue_create_buffer[DType.uint32](1)
        self._index_device = ctx.enqueue_create_buffer[DType.uint32](1)
        self._index = ctx.enqueue_create_host_buffer[DType.uint32](1)
        ctx.synchronize()

        if capacity > 0:
            self._reserve(ctx, capacity)
            # Touch the pinned pages now rather than inside the first scan,
            # so `capacity` really does buy what it promises.
            for i in range(0, self._bytes_capacity, 4096):
                self._text[i] = UInt8(0)

    def _reserve(mut self, ctx: DeviceContext, length: Int) raises:
        """Makes room for a document of `length` bytes."""
        var needed = length + CHUNK
        if needed <= self._bytes_capacity:
            return
        var chunks = (length + CHUNK - 1) // CHUNK
        var blocks = (chunks + THREADS - 1) // THREADS

        self._text = ctx.enqueue_create_host_buffer[DType.uint8](needed)
        self._document = ctx.enqueue_create_buffer[DType.uint8](needed)
        self._carry = ctx.enqueue_create_buffer[DType.uint32](chunks)
        self._outside = ctx.enqueue_create_buffer[DType.uint32](chunks)
        self._inside = ctx.enqueue_create_buffer[DType.uint32](chunks)
        self._counts = ctx.enqueue_create_buffer[DType.uint32](chunks)
        self._block_totals = ctx.enqueue_create_buffer[DType.uint32](blocks)
        self._block_offsets = ctx.enqueue_create_buffer[DType.uint32](blocks)
        ctx.synchronize()

        self._bytes_capacity = needed
        self._chunks_capacity = chunks
        self._blocks_capacity = blocks

    def _reserve_index(mut self, ctx: DeviceContext, entries: Int) raises:
        """Makes room for `entries` index entries at both ends."""
        if entries <= self._index_capacity:
            return
        self._index_device = ctx.enqueue_create_buffer[DType.uint32](entries)
        self._index = ctx.enqueue_create_host_buffer[DType.uint32](entries)
        ctx.synchronize()
        self._index_capacity = entries

    def scan(mut self, ctx: DeviceContext, path: Path) raises:
        """Finds every field in `path`, replacing whatever was scanned before.

        Any `StringSlice` handed out by `field` for an earlier document points
        into this scanner's buffers and is invalid after this returns.

        Args:
            ctx: The device to run on.
            path: The document to read.

        Raises:
            Error: If the file cannot be read, is 2 GiB or larger, or a device
                allocation or launch fails.
        """
        var length = Int(path.stat().st_size)
        if length > MAX_LENGTH:
            raise Error(
                "GpuCsvScanner indexes fields with 31 bits and cannot take a"
                " document of 2 GiB or more"
            )
        self._length = length
        self._count = 0
        self.column_count = 0
        if length == 0:
            return

        self._reserve(ctx, length)

        # The four loads in `chunk_masks` always read a whole chunk, so the
        # bytes past the document have to be there and have to be harmless.
        # Zeroed first, because the last slice carries them up with it.
        for i in range(length, length + CHUNK):
            self._text[i] = UInt8(0)

        # Read a slice, enqueue its upload, read the next. The upload does not
        # block, so it flies while the next slice is being read.
        with open(path, "r") as f:
            var offset = 0
            while offset < length:
                var take = UPLOAD_SLICE
                if offset + take > length:
                    take = length - offset
                var got = f.read(
                    Span[UInt8, MutUntrackedOrigin](
                        unsafe_ptr=self._text.unsafe_ptr().unsafe_offset(
                            offset
                        ),
                        length=take,
                    )
                )
                if got != take:
                    raise Error("short read: got ", got, " of ", take, " bytes")
                # The last slice carries the padding, which is already zeroed.
                var span = take + CHUNK if offset + take == length else take
                var view = self._document.create_sub_buffer[DType.uint8](
                    offset, span
                )
                view.enqueue_copy_from(
                    Span[UInt8, MutUntrackedOrigin](
                        unsafe_ptr=self._text.unsafe_ptr().unsafe_offset(
                            offset
                        ),
                        length=span,
                    )
                )
                offset += take

        var chunks = (length + CHUNK - 1) // CHUNK
        var blocks = (chunks + THREADS - 1) // THREADS

        ctx.enqueue_function[analyse_kernel[Self.separator]](
            self._document.unsafe_ptr(),
            self._carry.unsafe_ptr(),
            self._outside.unsafe_ptr(),
            self._inside.unsafe_ptr(),
            Int32(length),
            Int32(chunks),
            grid_dim=blocks,
            block_dim=THREADS,
        )

        Self._scan(
            ctx,
            self._carry,
            self._block_totals,
            self._block_offsets,
            self._grand,
            chunks,
            blocks,
        )

        ctx.enqueue_function[select_counts_kernel](
            self._carry.unsafe_ptr(),
            self._outside.unsafe_ptr(),
            self._inside.unsafe_ptr(),
            self._counts.unsafe_ptr(),
            Int32(chunks),
            grid_dim=blocks,
            block_dim=THREADS,
        )

        Self._scan(
            ctx,
            self._counts,
            self._block_totals,
            self._block_offsets,
            self._grand,
            chunks,
            blocks,
        )

        # `map_to_host` does not wait for enqueued work by itself, and without
        # this the total read back is whatever the allocation held.
        ctx.synchronize()
        var fields: Int
        with self._grand.map_to_host() as host:
            fields = Int(host[0])

        var trailing = 1 if self._text[length - 1] != LF else 0
        self._count = fields + trailing
        self._reserve_index(ctx, self._count + 1)

        ctx.enqueue_function[emit_kernel[Self.separator]](
            self._document.unsafe_ptr(),
            self._carry.unsafe_ptr(),
            self._counts.unsafe_ptr(),
            self._index_device.unsafe_ptr(),
            Int32(length),
            Int32(chunks),
            grid_dim=blocks,
            block_dim=THREADS,
        )
        self._index_device.enqueue_copy_to(
            Span[UInt32, MutUntrackedOrigin](
                unsafe_ptr=self._index.unsafe_ptr(), length=self._count + 1
            )
        )
        ctx.synchronize()

        # A document not ending in a line break leaves its last field open,
        # and the end of the text closes it.
        if trailing == 1:
            self._index[fields] = UInt32(length)

        # The first unquoted line feed closes the first row. Nothing in the
        # index says which delimiters were line feeds, so this asks the
        # document -- once, not per field.
        for i in range(self._count):
            if self._text[Int(self._index[i] & OFFSET_MASK)] == LF:
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
        """Turns `values` into its own exclusive prefix sum, in place.

        In place is safe because `scan_block_kernel` reads its element into a
        register before writing one.
        """
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
        """Returns the number of fields the last document held.

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
        return (start, Int(raw & OFFSET_MASK) - Int(raw >> 31))

    def field[
        origin: ImmOrigin, //
    ](ref[origin] self, row: Int, column: Int) raises -> StringSlice[origin]:
        """Returns a field exactly as it appears in the document.

        The slice borrows the scanner's own buffer and is invalid after the
        next `scan`.

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
