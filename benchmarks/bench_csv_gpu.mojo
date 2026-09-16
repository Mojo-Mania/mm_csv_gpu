"""What finding the fields of a CSV document on the GPU costs.

Run `bash data/setup.sh` first; it builds three files and prints their shape.
None of them is committed.

Two things are measured. **One-shot** is what `GpuCsvTable` costs: allocate
everything, read the file, upload, five kernels, copy the index back.
**Reused** is the same work through a `GpuCsvScanner` that already owns its
buffers, which is the number that matters for anything scanning more than one
document. **Phases** is the same work with a synchronisation between each
step, launched by hand from the library's own kernels, so the parts add up to
something close to the total but not exactly -- the synchronisations are not
free and the real path does not have them.

`small.csv` is byte for byte mm_csv's `no_escaping.csv`, so the CPU column is
that library's measured number on the same machine and the same bytes rather
than a reimplementation here.
"""

from max.gpu.host import DeviceContext
from mm_csv_gpu import GpuCsvScanner, GpuCsvTable
from mm_csv_gpu.kernels import (
    CHUNK,
    COMMA,
    THREADS,
    analyse_kernel,
    emit_kernel,
    select_counts_kernel,
)
from mm_csv_gpu.scan import (
    scan_apply_kernel,
    scan_block_kernel,
    scan_totals_kernel,
)
from std.pathlib import Path, cwd
from std.time import perf_counter_ns

comptime REPEATS = 5


def fixed(value: Float64, decimals: Int = 1) -> String:
    var scale = 1
    for _ in range(decimals):
        scale *= 10
    var scaled = Int(value * Float64(scale) + 0.5)
    var digits = String(scaled % scale)
    while digits.byte_length() < decimals:
        digits = String("0", digits)
    return String(scaled // scale, ".", digits)


def rjust(text: String, width: Int) -> String:
    var out = text.copy()
    while out.byte_length() < width:
        out = String(" ", out)
    return out


def ljust(text: String, width: Int) -> String:
    var out = text.copy()
    while out.byte_length() < width:
        out += " "
    return out


def report(label: String, nanos: Float64, bytes: Int, fields: Int):
    var gib = Float64(bytes) / (1024.0 * 1024.0 * 1024.0)
    var seconds = nanos / 1e9
    print(
        ljust(label, 24),
        rjust(fixed(nanos / 1e6), 9),
        rjust(fixed(gib / seconds, 2), 10),
        rjust(fixed(nanos / Float64(fields), 2), 10),
    )


def bench_file(ctx: DeviceContext, name: String) raises:
    var path = cwd() / "data" / name
    if not path.exists():
        raise Error(
            "data/", name, " is missing. Run `bash data/setup.sh` first."
        )
    var bytes = Int(path.stat().st_size)

    var probe = GpuCsvTable(ctx, path)
    var fields = len(probe)
    print()
    print(
        name,
        ": ",
        bytes,
        " bytes, ",
        probe.row_count(),
        " rows, ",
        probe.column_count,
        " columns, ",
        fields,
        " fields",
        sep="",
    )
    print(
        ljust("", 24), rjust("ms", 9), rjust("GiB/s", 10), rjust("ns/field", 10)
    )

    var best = Float64(1e30)
    for _ in range(REPEATS):
        var start = perf_counter_ns()
        var table = GpuCsvTable(ctx, path)
        var elapsed = Float64(perf_counter_ns() - start)
        if len(table) != fields:
            raise Error("field count moved between runs")
        if elapsed < best:
            best = elapsed
    report("one-shot GpuCsvTable", best, bytes, fields)

    # The same work through a scanner sized up front, which is what the type
    # is for: the allocation and the first-touch faults happen once, here,
    # outside the measurement.
    var scanner = GpuCsvScanner(ctx, capacity=bytes)
    scanner.scan(ctx, path)
    var best_warm = Float64(1e30)
    for _ in range(REPEATS):
        var start = perf_counter_ns()
        scanner.scan(ctx, path)
        var elapsed = Float64(perf_counter_ns() - start)
        if len(scanner) != fields:
            raise Error("field count moved between runs")
        if elapsed < best_warm:
            best_warm = elapsed
    report("reused GpuCsvScanner", best_warm, bytes, fields)

    # The phases, with a synchronisation between each.
    var length = bytes
    var chunks = (length + CHUNK - 1) // CHUNK
    var blocks = (chunks + THREADS - 1) // THREADS

    var staging = ctx.enqueue_create_host_buffer[DType.uint8](length + CHUNK)
    var document = ctx.enqueue_create_buffer[DType.uint8](length + CHUNK)
    var carry = ctx.enqueue_create_buffer[DType.uint32](chunks)
    var outside = ctx.enqueue_create_buffer[DType.uint32](chunks)
    var inside = ctx.enqueue_create_buffer[DType.uint32](chunks)
    var counts = ctx.enqueue_create_buffer[DType.uint32](chunks)
    # Distinct from `carry` only because passing one buffer as two arguments
    # of a launch is rejected as aliasing; the library scans in place.
    var scanned = ctx.enqueue_create_buffer[DType.uint32](chunks)
    var totals = ctx.enqueue_create_buffer[DType.uint32](blocks)
    var offs = ctx.enqueue_create_buffer[DType.uint32](blocks)
    var grand = ctx.enqueue_create_buffer[DType.uint32](1)
    var index = ctx.enqueue_create_buffer[DType.uint32](fields + 1)
    var index_host = ctx.enqueue_create_host_buffer[DType.uint32](fields + 1)
    ctx.synchronize()

    var best_read = Float64(1e30)
    for _ in range(REPEATS):
        var start = perf_counter_ns()
        with open(path, "r") as f:
            _ = f.read(
                Span[UInt8, MutUntrackedOrigin](
                    unsafe_ptr=staging.unsafe_ptr(), length=length
                )
            )
        var elapsed = Float64(perf_counter_ns() - start)
        if elapsed < best_read:
            best_read = elapsed
    report("  read into pinned", best_read, bytes, fields)

    var best_up = Float64(1e30)
    for _ in range(REPEATS):
        var start = perf_counter_ns()
        document.enqueue_copy_from(
            Span[UInt8, MutUntrackedOrigin](
                unsafe_ptr=staging.unsafe_ptr(), length=length + CHUNK
            )
        )
        ctx.synchronize()
        var elapsed = Float64(perf_counter_ns() - start)
        if elapsed < best_up:
            best_up = elapsed
    report("  upload", best_up, bytes, fields)

    var best_analyse = Float64(1e30)
    for _ in range(REPEATS):
        var start = perf_counter_ns()
        ctx.enqueue_function[analyse_kernel[COMMA]](
            document.unsafe_ptr(),
            carry.unsafe_ptr(),
            outside.unsafe_ptr(),
            inside.unsafe_ptr(),
            Int32(length),
            Int32(chunks),
            grid_dim=blocks,
            block_dim=THREADS,
        )
        ctx.synchronize()
        var elapsed = Float64(perf_counter_ns() - start)
        if elapsed < best_analyse:
            best_analyse = elapsed
    report("  analyse", best_analyse, bytes, fields)

    var best_scan = Float64(1e30)
    for _ in range(REPEATS):
        var start = perf_counter_ns()
        ctx.enqueue_function[scan_block_kernel](
            carry.unsafe_ptr(),
            scanned.unsafe_ptr(),
            totals.unsafe_ptr(),
            Int32(chunks),
            grid_dim=blocks,
            block_dim=THREADS,
        )
        ctx.enqueue_function[scan_totals_kernel](
            totals.unsafe_ptr(),
            offs.unsafe_ptr(),
            grand.unsafe_ptr(),
            Int32(blocks),
            grid_dim=1,
            block_dim=THREADS,
        )
        ctx.enqueue_function[scan_apply_kernel](
            scanned.unsafe_ptr(),
            offs.unsafe_ptr(),
            Int32(chunks),
            grid_dim=blocks,
            block_dim=THREADS,
        )
        ctx.synchronize()
        var elapsed = Float64(perf_counter_ns() - start)
        if elapsed < best_scan:
            best_scan = elapsed
    report("  one prefix scan", best_scan, bytes, fields)

    var best_emit = Float64(1e30)
    for _ in range(REPEATS):
        var start = perf_counter_ns()
        ctx.enqueue_function[emit_kernel[COMMA]](
            document.unsafe_ptr(),
            carry.unsafe_ptr(),
            counts.unsafe_ptr(),
            index.unsafe_ptr(),
            Int32(length),
            Int32(chunks),
            grid_dim=blocks,
            block_dim=THREADS,
        )
        ctx.synchronize()
        var elapsed = Float64(perf_counter_ns() - start)
        if elapsed < best_emit:
            best_emit = elapsed
    report("  emit", best_emit, bytes, fields)

    var best_down = Float64(1e30)
    for _ in range(REPEATS):
        var start = perf_counter_ns()
        index.enqueue_copy_to(
            Span[UInt32, MutUntrackedOrigin](
                unsafe_ptr=index_host.unsafe_ptr(), length=fields + 1
            )
        )
        ctx.synchronize()
        var elapsed = Float64(perf_counter_ns() - start)
        if elapsed < best_down:
            best_down = elapsed
    report("  index back", best_down, bytes, fields)


def main() raises:
    print("Finding CSV fields on the GPU. Best of", REPEATS, "runs.")
    var ctx = DeviceContext()
    bench_file(ctx, "small.csv")
    bench_file(ctx, "large.csv")
    bench_file(ctx, "quoted.csv")
