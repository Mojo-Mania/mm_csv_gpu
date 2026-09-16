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

Scanning more than one document? Keep the buffers:

```mojo
var scanner = GpuCsvScanner(ctx, capacity=512 << 20)
for path in paths:
    scanner.scan(ctx, path)
    print(len(scanner), "fields in", path)
```

This is the GPU counterpart to [mm_csv](https://github.com/Mojo-Mania/mm_csv),
the way [mm_radix_sort_gpu](https://github.com/Mojo-Mania/mm_radix_sort_gpu) is
to mm_radix_sort. It produces the same index — one `UInt32` per field holding
the offset of the delimiter that closed it, with bit 31 set when that delimiter
was an LF preceded by a CR.

## Read this before reaching for it

**Reuse the scanner, or most of the win goes to the allocator.** A fresh
pinned host buffer the size of the document costs 4.5 ms to allocate and
22.5 ms to fault in on first touch -- against 2.6 ms for all five kernels. So
`GpuCsvTable`, which allocates per document, runs a 253 MB file in 40 ms, and
a `GpuCsvScanner` that already owns its buffers runs the same file in
**18.5 ms**. Use the scanner for anything but a one-off.

**Size matters more than usual.** At 23 MB a warm scan is 2.3 ms against the
CPU library's 2.2 -- a wash. At 253 MB it is 18.6 ms against about 25 plus the
CPU's own file read. The fixed costs are large and the marginal cost is tiny,
so the crossover is somewhere in the tens of megabytes.

**Unified memory is doing a lot of work here.** These numbers are Apple silicon,
where the upload is a 185 GiB/s copy between two regions of the same RAM. On a
discrete card the document crosses PCIe and the arithmetic is different.

**A kernel cannot read a pinned host buffer.** It reads zeros, silently, and a
timing taken that way looks wonderful. The pinned buffer here is a *source* for
the upload, which is what makes the upload 185 GiB/s instead of 6.

**Reading the file is now the bottleneck.** 13.3 ms of the 18.6 is
`FileHandle.read`, at 17.7 GiB/s. Everything on the device put together is
4.4. Making this faster is a file-I/O problem, not a GPU one.

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
   values is the low bit of an exclusive prefix *sum* over them. It is two
   levels deep: the block totals are scanned the same way the values are, and
   only the totals of *those* go to a single block.
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
| `GpuCsvScanner[separator](ctx, capacity=0)` | A reusable scanner. Size it now or pay on first scan. |
| `.scan(ctx, path)` | Scans `path`, replacing what was there. |
| `GpuCsvTable[separator](ctx, path)` | One document, own buffers. Convenience over the above. |
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
| one-shot `GpuCsvTable` | 40.2 | 5.86 | 1.80 |
| **reused `GpuCsvScanner`** | **18.5** | **12.76** | **0.82** |
| read into pinned memory | 13.4 | 17.6 | 0.60 |
| upload | 1.3 | **187.7** | 0.06 |
| analyse | 1.2 | **202.2** | 0.05 |
| one prefix scan (of two) | 0.6 | **375.2** | 0.03 |
| emit | 1.1 | **219.6** | 0.05 |
| index back | 0.5 | **512.2** | 0.02 |

**`small.csv`** — 23 MB, 255 361 rows, 8 columns, 2 042 888 fields:

| | ms | GiB/s |
| --- | ---: | ---: |
| one-shot `GpuCsvTable` | 5.8 | 3.73 |
| reused `GpuCsvScanner` | 2.3 | 9.49 |
| — mm_csv, CPU, parse only | **2.2** | **9.42** |

Three things to read off these.

**The device work is small and scales well.** On the 253 MB document analyse
and emit each read the whole thing at about 285 GiB/s and the two scans cost
1 ms between them: 2.6 ms of kernels, against 25 ms for the CPU library to
parse the same shape. Add the upload and the index download and the device
side is 4.4 ms.

**Allocation was three quarters of the one-shot cost**, and reusing a scanner
removes it: 58.8 ms to 18.5. That is the single largest thing in this
repository's history and it is not an optimisation of the algorithm at all.
Overlapping the read with the upload took the one-shot path from 58.8 to 40 by
hiding the same page faults a different way — see
[`docs/improvements.md`](docs/improvements.md), where it is also the one
measurement that helped a path it was not aimed at.

**What is left is the file read.** 13.3 ms of the 18.6. Two ways of making it
faster have been measured and neither works — see
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

**They need a real GPU**, so they run here before a push rather than in CI.
What CI does instead is compile the kernels for four architectures —
`metal:4`, `metal:1`, `sm_80`, `gfx942` — through
`mojo build --target-accelerator`, which names the target rather than asking
the machine for one. Two of those nobody here owns. That is the check that
matters: the one bug this library has had that no amount of host-side testing
would find was `pack_bits` taking Metal's shader compiler down, and device
codegen is the only thing that sees it.

## License

MIT. See [LICENSE](LICENSE).
