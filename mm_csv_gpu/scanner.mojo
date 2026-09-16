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

The pipeline is five launches:

    analyse -> scan (parities) -> select -> scan (counts) -> emit

with a readback before the emit, because the index cannot be placed until the
second scan has said how many fields there are.

It runs one of two ways. With unified memory the whole document goes up, is
scanned, and the index comes back once -- there is no bus to share. On a
discrete device the upload and the download are most of the cost, and PCIe
carries both directions at once, so the document is scanned `STREAM_SLICE` at
a time: each slice goes up and is counted on the first stream, and a slice
later is emitted and comes back on a second stream while the next slices are
still going up. On a laptop RTX 4050 that took a 253 MB document from 33.3 ms
to 27.3, and a 23 MB one from 4.1 to 2.9.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, DeviceStream, HostBuffer
from std.memory import unsafe_memcpy
from std.pathlib import Path

from .kernels import (
    CHUNK,
    COMMA,
    LF,
    OFFSET_MASK,
    THREADS,
    analyse_kernel,
    emit_kernel,
    fold_parity_kernel,
    select_counts_kernel,
)
from .scan import scan_apply_kernel, scan_block_kernel, scan_totals_kernel

comptime UPLOAD_SLICE = 16 * 1024 * 1024
"""Bytes read before the upload of what was read is enqueued.

Reading the file runs at about 17.7 GiB/s and the upload at 185, so the
upload is the smaller of the two and can hide behind the next read. Enqueued
work does not block, so slicing the loop is the whole trick. Measured on a
253 MB document: 15.8 ms in one slice, 14.9 at sixteen or more, and flat
after that. A document under this size is one slice and unchanged.

This is the slice for unified memory, where the whole document goes up before
anything is scanned. `STREAM_SLICE` is the one for a discrete device."""

comptime STREAM_SLICE = 4 * 1024 * 1024
"""Bytes per slice when each slice is scanned and downloaded on its own.

Smaller slices start downloading sooner and leave less of the last one without
an upload to overlap, but each costs a host synchronisation. Measured on an
RTX 4050, reused scanner, `small.csv` / `large.csv`: 1 MiB 2.9 / 30.0 ms,
2 MiB 2.8 / 27.8, 4 MiB 2.8 / 27.2, 8 MiB 3.3 / 27.4, 16 MiB 4.2 / 28.0,
32 MiB 4.6 / 29.4, 64 MiB 4.6 / 32.3. Must be a multiple of `CHUNK`."""

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
    var _totals: DeviceBuffer[DType.uint32]
    var _totals2: DeviceBuffer[DType.uint32]
    var _grand: DeviceBuffer[DType.uint32]
    var _index_device: DeviceBuffer[DType.uint32]
    var _index: HostBuffer[DType.uint32]
    var _incoming: DeviceBuffer[DType.uint32]
    var _slice_counts: DeviceBuffer[DType.uint32]
    var _slice_counts_host: HostBuffer[DType.uint32]

    var _down: DeviceContext
    """Where the index and the per-slice field counts come back.

    A second stream on a discrete device, so the download can run while the
    next slice is still going up; the device context itself when the host
    memory is unified and there is no bus to share."""
    var _stream: Optional[DeviceStream]
    """Kept alive for `_down`."""
    var _streamed: Bool
    """Whether `scan` works a slice at a time. Private, and settable, so the
    tests can drive both paths on one machine."""

    var _bytes_capacity: Int
    var _chunks_capacity: Int
    var _blocks_capacity: Int
    var _index_capacity: Int
    var _slices_capacity: Int

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
        self._slices_capacity = 0

        self._streamed = not ctx.is_host_unified()
        if self._streamed:
            self._stream = ctx.create_stream()
            self._down = ctx.select_stream(ctx.num_streams() - 1)
        else:
            self._stream = None
            self._down = ctx.copy()

        # Everything starts empty and `_reserve` does the real work, so there
        # is one growth path rather than two.
        self._text = ctx.enqueue_create_host_buffer[DType.uint8](1)
        self._document = ctx.enqueue_create_buffer[DType.uint8](1)
        self._carry = ctx.enqueue_create_buffer[DType.uint32](1)
        self._outside = ctx.enqueue_create_buffer[DType.uint32](1)
        self._inside = ctx.enqueue_create_buffer[DType.uint32](1)
        self._counts = ctx.enqueue_create_buffer[DType.uint32](1)
        self._totals = ctx.enqueue_create_buffer[DType.uint32](1)
        self._totals2 = ctx.enqueue_create_buffer[DType.uint32](1)
        self._grand = ctx.enqueue_create_buffer[DType.uint32](1)
        self._incoming = ctx.enqueue_create_buffer[DType.uint32](1)
        # Everything the second stream copies from or into is created on it.
        # A copy through a buffer created on the first stream waits for that
        # stream, and a download waiting for the upload is the whole problem.
        self._index_device = self._down.enqueue_create_buffer[DType.uint32](1)
        self._index = self._down.enqueue_create_host_buffer[DType.uint32](1)
        self._slice_counts = self._down.enqueue_create_buffer[DType.uint32](1)
        self._slice_counts_host = self._down.enqueue_create_host_buffer[
            DType.uint32
        ](1)
        ctx.synchronize()
        self._down.synchronize()

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
        var blocks2 = (blocks + THREADS - 1) // THREADS

        self._text = ctx.enqueue_create_host_buffer[DType.uint8](needed)
        self._document = ctx.enqueue_create_buffer[DType.uint8](needed)
        self._carry = ctx.enqueue_create_buffer[DType.uint32](chunks)
        self._outside = ctx.enqueue_create_buffer[DType.uint32](chunks)
        self._inside = ctx.enqueue_create_buffer[DType.uint32](chunks)
        self._counts = ctx.enqueue_create_buffer[DType.uint32](chunks)
        self._totals = ctx.enqueue_create_buffer[DType.uint32](blocks)
        self._totals2 = ctx.enqueue_create_buffer[DType.uint32](blocks2)
        ctx.synchronize()

        self._bytes_capacity = needed
        self._chunks_capacity = chunks
        self._blocks_capacity = blocks

    def _reserve_index(mut self, entries: Int, keep: Int) raises:
        """Makes room for `entries` index entries at both ends.

        The first `keep` entries of the host index survive; nothing on the
        device does, so every download must have finished before this is
        called with a `keep` that matters.
        """
        if entries <= self._index_capacity:
            return
        var host = self._down.enqueue_create_host_buffer[DType.uint32](entries)
        self._index_device = self._down.enqueue_create_buffer[DType.uint32](
            entries
        )
        self._down.synchronize()
        if keep > 0:
            unsafe_memcpy(
                dest=host.unsafe_ptr(), src=self._index.unsafe_ptr(), count=keep
            )
        self._index = host^
        self._index_capacity = entries

    def _reserve_slices(mut self, slices: Int) raises:
        """Makes room for one field count per slice, at both ends."""
        if slices <= self._slices_capacity:
            return
        self._slice_counts = self._down.enqueue_create_buffer[DType.uint32](
            slices
        )
        self._slice_counts_host = self._down.enqueue_create_host_buffer[
            DType.uint32
        ](slices)
        self._down.synchronize()
        self._slices_capacity = slices

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

        # The loads in `chunk_masks` always read a whole chunk, so the bytes
        # past the document have to be there and have to be harmless. Zeroed
        # first, because the last slice carries them up with it.
        for i in range(length, length + CHUNK):
            self._text[i] = UInt8(0)

        var fields: Int
        if self._streamed:
            fields = self._scan_streamed(ctx, path, length)
        else:
            fields = self._scan_whole(ctx, path, length)

        # A document not ending in a line break leaves its last field open,
        # and the end of the text closes it.
        var trailing = 1 if self._text[length - 1] != LF else 0
        self._count = fields + trailing
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

    def _read_slice(
        mut self, mut f: FileHandle, length: Int, offset: Int, size: Int
    ) raises -> Int:
        """Reads the slice at `offset` and enqueues its upload.

        The upload does not block, so it flies while the next slice is read.

        Returns:
            The document bytes in the slice.
        """
        var take = size
        if offset + take > length:
            take = length - offset
        var got = f.read(
            Span[UInt8, MutUntrackedOrigin](
                unsafe_ptr=self._text.unsafe_ptr().unsafe_offset(offset),
                length=take,
            )
        )
        if got != take:
            raise Error("short read: got ", got, " of ", take, " bytes")
        # The last slice carries the padding, which is already zeroed.
        var span = take + CHUNK if offset + take == length else take
        var view = self._document.create_sub_buffer[DType.uint8](offset, span)
        view.enqueue_copy_from(
            Span[UInt8, MutUntrackedOrigin](
                unsafe_ptr=self._text.unsafe_ptr().unsafe_offset(offset),
                length=span,
            )
        )
        return take

    def _count_fields(
        mut self,
        ctx: DeviceContext,
        length: Int,
        first: Int,
        chunks: Int,
        total: DeviceBuffer[DType.uint32],
    ) raises:
        """Enqueues everything up to the field count for chunks `first..<chunks`.

        analyse, the parity scan, select, and the count scan, leaving the
        count in `total[0]` and each chunk's write offset -- relative to the
        first chunk -- in `_counts`. The quote state at `first` comes from
        `_incoming`, which this moves past the last chunk.
        """
        var n = chunks - first
        var blocks = (n + THREADS - 1) // THREADS
        ctx.enqueue_function[analyse_kernel[Self.separator]](
            self._document.unsafe_ptr(),
            self._carry.unsafe_ptr(),
            self._outside.unsafe_ptr(),
            self._inside.unsafe_ptr(),
            Int32(length),
            Int32(first),
            Int32(chunks),
            grid_dim=blocks,
            block_dim=THREADS,
        )
        Self._scan(
            ctx,
            self._carry.create_sub_buffer[DType.uint32](first, n),
            self._totals,
            self._totals2,
            self._grand,
            n,
        )
        ctx.enqueue_function[select_counts_kernel](
            self._carry.unsafe_ptr(),
            self._outside.unsafe_ptr(),
            self._inside.unsafe_ptr(),
            self._counts.unsafe_ptr(),
            self._incoming.unsafe_ptr(),
            Int32(first),
            Int32(chunks),
            grid_dim=blocks,
            block_dim=THREADS,
        )
        ctx.enqueue_function[fold_parity_kernel](
            self._incoming.unsafe_ptr(),
            self._grand.unsafe_ptr(),
            grid_dim=1,
            block_dim=1,
        )
        Self._scan(
            ctx,
            self._counts.create_sub_buffer[DType.uint32](first, n),
            self._totals,
            self._totals2,
            total,
            n,
        )

    def _emit(
        mut self,
        ctx: DeviceContext,
        length: Int,
        first: Int,
        chunks: Int,
        start: Int,
    ) raises:
        """Enqueues the emit for chunks `first..<chunks`, from entry `start`."""
        var blocks = (chunks - first + THREADS - 1) // THREADS
        ctx.enqueue_function[emit_kernel[Self.separator]](
            self._document.unsafe_ptr(),
            self._carry.unsafe_ptr(),
            self._counts.unsafe_ptr(),
            self._index_device.unsafe_ptr(),
            Int32(length),
            Int32(first),
            Int32(chunks),
            Int32(start),
            grid_dim=blocks,
            block_dim=THREADS,
        )

    def _scan_whole(
        mut self, ctx: DeviceContext, path: Path, length: Int
    ) raises -> Int:
        """Reads the whole document up, then scans it in one pass.

        For unified memory, where the index comes back at 500 GiB/s and there
        is nothing for a second stream to overlap.

        Returns:
            Fields closed by a delimiter.
        """
        with open(path, "r") as f:
            var offset = 0
            while offset < length:
                offset += self._read_slice(f, length, offset, UPLOAD_SLICE)

        var chunks = (length + CHUNK - 1) // CHUNK
        self._incoming.enqueue_fill(UInt32(0))
        self._count_fields(
            ctx,
            length,
            0,
            chunks,
            self._grand.create_sub_buffer[DType.uint32](0, 1),
        )

        # `map_to_host` does not wait for enqueued work by itself, and without
        # this the total read back is whatever the allocation held.
        ctx.synchronize()
        var fields: Int
        with self._grand.map_to_host() as host:
            fields = Int(host[0])

        self._reserve_index(fields + 1, 0)
        self._emit(ctx, length, 0, chunks, 0)
        ctx.synchronize()
        self._index_device.enqueue_copy_to(
            Span[UInt32, MutUntrackedOrigin](
                unsafe_ptr=self._index.unsafe_ptr(), length=fields + 1
            )
        )
        self._down.synchronize()
        return fields

    def _scan_streamed(
        mut self, ctx: DeviceContext, path: Path, length: Int
    ) raises -> Int:
        """Reads, scans and downloads a slice at a time, on two streams.

        On a discrete device the upload and the index download are most of the
        cost, and PCIe carries both directions at once. So each slice goes up,
        is counted, and -- one slice later, once its field count is on the
        host and says where its entries go -- is emitted and comes back on the
        second stream while the next slices are still going up.

        Nothing else in a slice needs the host. The quote state crosses slice
        boundaries on the device, in `_incoming`; a chunk's CR test reads the
        previous slice's last byte, which is already up.

        Returns:
            Fields closed by a delimiter.
        """
        var slices = (length + STREAM_SLICE - 1) // STREAM_SLICE
        self._reserve_slices(slices)
        self._incoming.enqueue_fill(UInt32(0))

        var start = 0
        with open(path, "r") as f:
            for k in range(slices):
                var offset = k * STREAM_SLICE
                var take = self._read_slice(f, length, offset, STREAM_SLICE)
                self._count_fields(
                    ctx,
                    length,
                    offset // CHUNK,
                    (offset + take + CHUNK - 1) // CHUNK,
                    self._slice_counts.create_sub_buffer[DType.uint32](k, 1),
                )
                if k > 0:
                    # Waits for the previous slice's count, not for this
                    # slice's upload, which the second stream is not behind.
                    self._down.synchronize()
                    start = self._emit_slice(ctx, length, slices, k - 1, start)
                self._down.enqueue_wait_for(ctx)
                self._slice_counts.create_sub_buffer[DType.uint32](
                    k, 1
                ).enqueue_copy_to(
                    Span[UInt32, MutUntrackedOrigin](
                        unsafe_ptr=self._slice_counts_host.unsafe_ptr().unsafe_offset(
                            k
                        ),
                        length=1,
                    )
                )
        self._down.synchronize()
        start = self._emit_slice(ctx, length, slices, slices - 1, start)
        ctx.synchronize()
        self._down.synchronize()
        return start

    def _emit_slice(
        mut self,
        ctx: DeviceContext,
        length: Int,
        slices: Int,
        k: Int,
        start: Int,
    ) raises -> Int:
        """Emits slice `k` from entry `start` and enqueues its download.

        Its field count must already be on the host.

        Returns:
            The entry after the slice's last.
        """
        var count = Int(self._slice_counts_host[k])
        var end = start + count
        if end + 1 > self._index_capacity:
            # Everything pending has to land before the index moves. A scanner
            # that has seen a document this size before never gets here.
            ctx.synchronize()
            self._down.synchronize()
            var scanned = (k + 1) * STREAM_SLICE
            if scanned > length:
                scanned = length
            # Extrapolate from what has been counted, with an eighth to
            # spare, so a first scan grows once or twice rather than per slice.
            var estimate = end * length // scanned * 9 // 8 + slices
            self._reserve_index(max(end + 1, estimate), start)
        var offset = k * STREAM_SLICE
        var take = STREAM_SLICE if offset + STREAM_SLICE <= length else (
            length - offset
        )
        self._emit(
            ctx,
            length,
            offset // CHUNK,
            (offset + take + CHUNK - 1) // CHUNK,
            start,
        )
        if count > 0:
            self._down.enqueue_wait_for(ctx)
            self._index_device.create_sub_buffer[DType.uint32](
                start, count
            ).enqueue_copy_to(
                Span[UInt32, MutUntrackedOrigin](
                    unsafe_ptr=self._index.unsafe_ptr().unsafe_offset(start),
                    length=count,
                )
            )
        return end

    @staticmethod
    def _scan(
        ctx: DeviceContext,
        values: DeviceBuffer[DType.uint32],
        totals: DeviceBuffer[DType.uint32],
        totals2: DeviceBuffer[DType.uint32],
        grand: DeviceBuffer[DType.uint32],
        n: Int,
    ) raises:
        """Turns `values` into its own exclusive prefix sum, in place.

        In place is safe because `scan_block_kernel` and `scan_totals_kernel`
        both read an element into a register before writing one.

        The recursion is written out rather than looped, because it is never
        more than three levels deep. A chunk is 64 bytes and a block scans 256
        of them, so a 2 GiB document -- the largest this library takes -- has
        33.5 M chunks, 131 072 block totals, and 512 totals of those. The
        innermost level walks its input in tiles of `THREADS` with a running
        carry, so at 2 GiB it walks two tiles, and at 253 MB it walks one.

        That is the whole point of the second level. Without it the innermost
        kernel is the only one there is, and it walks **every** block total:
        61 serial tiles on a 253 MB document and 512 on a 2 GiB one, in a
        single block, while the rest of the device idles.
        """
        if n <= THREADS:
            ctx.enqueue_function[scan_totals_kernel](
                values.unsafe_ptr(),
                values.unsafe_ptr(),
                grand.unsafe_ptr(),
                Int32(n),
                grid_dim=1,
                block_dim=THREADS,
            )
            return

        var blocks = (n + THREADS - 1) // THREADS
        ctx.enqueue_function[scan_block_kernel](
            values.unsafe_ptr(),
            values.unsafe_ptr(),
            totals.unsafe_ptr(),
            Int32(n),
            grid_dim=blocks,
            block_dim=THREADS,
        )

        if blocks <= THREADS:
            ctx.enqueue_function[scan_totals_kernel](
                totals.unsafe_ptr(),
                totals.unsafe_ptr(),
                grand.unsafe_ptr(),
                Int32(blocks),
                grid_dim=1,
                block_dim=THREADS,
            )
        else:
            var blocks2 = (blocks + THREADS - 1) // THREADS
            ctx.enqueue_function[scan_block_kernel](
                totals.unsafe_ptr(),
                totals.unsafe_ptr(),
                totals2.unsafe_ptr(),
                Int32(blocks),
                grid_dim=blocks2,
                block_dim=THREADS,
            )
            ctx.enqueue_function[scan_totals_kernel](
                totals2.unsafe_ptr(),
                totals2.unsafe_ptr(),
                grand.unsafe_ptr(),
                Int32(blocks2),
                grid_dim=1,
                block_dim=THREADS,
            )
            ctx.enqueue_function[scan_apply_kernel](
                totals.unsafe_ptr(),
                totals2.unsafe_ptr(),
                Int32(blocks),
                grid_dim=blocks2,
                block_dim=THREADS,
            )

        ctx.enqueue_function[scan_apply_kernel](
            values.unsafe_ptr(),
            totals.unsafe_ptr(),
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
