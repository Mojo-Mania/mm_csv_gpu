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

**Which GPU matters more than anything else here.** On an Apple M4 Max a
reused scanner finds the fields of a 253 MB document in 18.5 ms, against
about 25 for the CPU library to parse the same shape. On a laptop RTX 4050
the same scan takes 27.4 ms, and the CPU next to it parses the same bytes in
14.1. Measure on the machine you will ship to.

**On Apple silicon, reuse the scanner, or most of the win goes to the
allocator.** A fresh pinned host buffer the size of the document costs 4.5 ms
to allocate and 22.5 ms to fault in on first touch, against 3.5 ms for all
five kernels. So `GpuCsvTable`, which allocates per document, runs the 253 MB
file in 40 ms, and a `GpuCsvScanner` that already owns its buffers runs it in
**18.5 ms**. On the RTX 4050 under Linux the two are within 2% of each other,
so reusing buffers is not what decides the speed there.

**Size matters more than usual.** At 23 MB a warm scan on the M4 Max is
2.3 ms against the CPU library's 2.2 -- a wash. The fixed costs are large and
the marginal cost is tiny, so on that machine the crossover is somewhere in
the tens of megabytes. On the RTX 4050 there is no crossover in the sizes
measured: the GPU is a little over twice as slow at 23 MB and about twice as
slow at 253 MB.

**Unified memory is doing a lot of work on Apple.** There the upload is a
188 GiB/s copy between two regions of the same RAM. On a discrete card the
document crosses PCIe: 12.5 GiB/s on the RTX 4050's PCIe 4.0 x8 link, so
18.8 ms to upload 253 MB and 6.8 ms to download the index. The download now
runs on a second stream while the upload is still going, but the upload alone
is more than the CPU needs to parse the whole document.

**On Metal a kernel cannot read a pinned host buffer.** It reads zeros,
silently, and a timing taken that way looks wonderful. The pinned buffer here
is a *source* for the upload, which is what made the upload 188 GiB/s on Apple
instead of 6. On CUDA a kernel *can* read one, and gets the right values -- it
is just slower than uploading: analyse reading the pinned document directly
took 30.4 ms against 20.4 for upload plus analyse, and emit writing straight
into a pinned index took a full second.

**What limits the speed depends on the machine.** On the M4 Max it is
reading the file: 13.4 ms of the 18.5 is `FileHandle.read`, and everything on
the device together is 5.3. On the RTX 4050 it is the PCIe link: the upload
alone is 18.8 ms of the 27.4, and all five kernels together are 4.6.

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

On a discrete GPU the same five steps run a 4 MiB slice at a time, so that the
index can come back while the rest of the document is still going up. The
quote state crosses from one slice to the next on the device, in a single
integer that `select` folds in. Each slice is emitted one slice late, once its
field count is on the host and says where its entries go, and it downloads on
a second stream. With unified memory there is nothing for a second stream to
overlap, and the whole document is scanned at once.

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

`-D ASSERT=none`, best of five. Reproduce with
`bash data/setup.sh && pixi run bench`. `small.csv` is byte for byte mm_csv's
`no_escaping.csv` — same generator, same seed, same columns — so the CPU
comparison is that library's measured number on the same bytes rather than a
reimplementation here.

The phase rows are each timed with a synchronisation around them, so they do
not add up to the total. The real path has no synchronisations in between, and
it overlaps the read with the upload in 16 MiB slices.

### Apple M4 Max

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

**The device work is small and scales well.** On the 253 MB document analyse
and emit each read the whole thing at over 200 GiB/s and the two scans cost
1.2 ms between them: 3.5 ms of kernels, against about 25 ms for the CPU
library to parse the same shape. Add the upload and the index download and
the device side is 5.3 ms.

**Allocation was three quarters of the one-shot cost**, and reusing a scanner
removes it: 58.8 ms to 18.5. That is the single largest thing in this
repository's history and it is not an optimisation of the algorithm at all.
Overlapping the read with the upload took the one-shot path from 58.8 to 40 by
hiding the same page faults a different way — see
[`docs/improvements.md`](docs/improvements.md), where it is also the one
measurement that helped a path it was not aimed at.

**What is left is the file read.** 13.4 ms of the 18.5. Two ways of making it
faster have been measured and neither works — see
[`docs/improvements.md`](docs/improvements.md).

### NVIDIA GeForce RTX 4050 Laptop GPU, AMD Ryzen AI 9 HX 370

A discrete card with 6 GB of memory, PCIe 4.0 x8, driver 610.57.04, CUDA 13.3,
Linux, Mojo 1.2.0.dev2026091505. All 12 tests pass on it. Three runs of the
benchmark agree to within 2%. The CPU row is mm_csv's `CsvTable` (AVX-512
VBMI2) on the same laptop, same files, best of five, three runs.

**`large.csv`** — 253 MB, 2 800 001 rows, 8 columns, 22 400 008 fields:

| | ms | GiB/s | ns/field |
| --- | ---: | ---: | ---: |
| one-shot `GpuCsvTable` | 27.9 | 8.45 | 1.25 |
| reused `GpuCsvScanner` | 27.4 | 8.60 | 1.22 |
| read into pinned memory | 19.6 | 12.0 | 0.87 |
| upload | 18.9 | 12.5 | 0.84 |
| analyse | 1.6 | 143.9 | 0.07 |
| one prefix scan (of two) | 0.2 | **1 180** | 0.01 |
| emit | 2.4 | 97.0 | 0.11 |
| index back | 6.8 | 34.7 | 0.30 |
| — mm_csv, CPU, parse only | **14.1** | **16.7** | **0.63** |

**`quoted.csv`** — 272 MB, 2 200 001 rows, 10 columns, 22 000 010 fields,
one quoted field containing a comma in every row:

| | ms | GiB/s | ns/field |
| --- | ---: | ---: | ---: |
| one-shot `GpuCsvTable` | 27.9 | 9.10 | 1.27 |
| reused `GpuCsvScanner` | 27.6 | 9.19 | 1.25 |
| read into pinned memory | 18.6 | 13.6 | 0.85 |
| upload | 20.3 | 12.5 | 0.92 |
| analyse | 1.8 | 143.9 | 0.08 |
| one prefix scan (of two) | 0.2 | **1 168** | 0.01 |
| emit | 2.6 | 98.2 | 0.12 |
| index back | 6.7 | 38.0 | 0.30 |
| — mm_csv, CPU, parse only | **15.1** | **16.8** | **0.69** |

**`small.csv`** — 23 MB:

| | ms | GiB/s |
| --- | ---: | ---: |
| one-shot `GpuCsvTable` | 3.3 | 6.55 |
| reused `GpuCsvScanner` | 2.8 | 7.58 |
| — mm_csv, CPU, parse only | **1.2** | **18.2** |

**The CPU still wins, at every size.** The CPU library parses the 253 MB
document in 14.1 ms, less than the upload alone. Leave out the file read on
both sides and the GPU still spends 18.8 ms just getting the document there.

The phase rows are each timed alone, one after another. The real path overlaps
the read with the upload and the download with both, which is why the total is
well under their sum.

**The kernels were five times slower than they needed to be.** The first
measurement on this card had analyse at 7.8 ms and emit at 9.0, 25-30 GiB/s
against 200-220 on the M4 Max. NVIDIA's backend does not keep `SIMD` vectors
as vectors: a 16-byte load becomes a loop inserting one byte at a time, and
every compare is split into scalars. On non-Apple devices the chunk is now
read as eight 64-bit words with SWAR compares, which took analyse to 1.6 ms
and emit to 2.4, the whole scan from 46.4 ms to 33.2, and changed no index
entry. Apple keeps the `SIMD` version, which is what it was measured
with.

**The download now overlaps the upload.** PCIe carries both directions at
once, but only between buffers created on different streams: MAX makes a
copy through a buffer wait for the stream that created it, and with one
stream the 6.8 ms download simply queued behind the upload. Scanning in 4 MiB
slices, with each slice's index downloading on a second stream while later
slices go up, took the scan from 33.2 ms to **27.4**, and `small.csv` from 4.1
to **2.8**. The slice size is from a sweep: 1 MiB paid too many host
synchronisations, and 16 MiB or more left too much of the download at the end
with nothing to overlap.

**The upload is what is left.** 12.5 GiB/s is about what PCIe 4.0 x8
carries, and 18.8 ms of it is not going anywhere on this card. A card on x16
should roughly halve it; that has not been measured.

**Reusing the scanner makes no difference here.** One-shot and reused agree to
within 2% on all three documents, where on the M4 Max they are 40 ms against
18.5. Why allocation costs so little under CUDA on Linux has not been
measured.

**Quoting costs nothing extra.** `quoted.csv` scans at about the same rate as
`large.csv` per field, on both the GPU and the CPU. The quote state is
computed, not carried, so a quoted field is no more work than any other byte.

## Development

```bash
pixi run test      # 12 tests, against a scalar reference
pixi run bench     # needs `bash data/setup.sh` first
pixi run main      # the example
pixi run format
pixi run docs
```

The tests compare every index entry against a fifteen-line scalar walk, over
documents built to break the parts a chunk cannot answer for itself: a quoted
region spanning chunks, a CRLF split across them, every tail alignment, and
documents large enough to push the quote parity through every part of the
prefix scan — within a block, across blocks, and through the second level
that scans the block totals. One more puts quoted regions and CRLFs across
the 4 MiB slice boundaries and runs the same document down both paths, sliced
and whole.

**They need a real GPU**, so they run on a machine that has one before a push,
not in CI. They have passed on an M4 Max (Metal) and an RTX 4050 (CUDA).
What CI does instead is compile the kernels for four architectures —
`metal:4`, `metal:1`, `sm_80`, `gfx942` — through
`mojo build --target-accelerator`, which names the target rather than asking
the machine for one. Two of those nobody here owns. That is the check that
matters: the one bug this library has had that no amount of host-side testing
would find was `pack_bits` taking Metal's shader compiler down, and device
codegen is the only thing that sees it.

## License

MIT. See [LICENSE](LICENSE).
