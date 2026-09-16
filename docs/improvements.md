# Identified improvements

## Done: the allocation was the whole cost

`GpuCsvScanner` keeps its buffers and rescans through them. On the 253 MB
document that is **58.8 ms to 18.6**, and on 23 MB 6.1 to 2.3. The estimate
below said "near 19" before the type existed, which is the one time in this
repository's history a prediction has come out right.

The buffers grow when a document does not fit and never shrink, so a scanner
settles at the size of the largest document it has seen. `capacity` in the
constructor allocates and touches the pinned pages up front, so the first
`scan` is as cheap as the second. Fields borrow the scanner's own memory and
are invalid after the next `scan` -- that is the bargain, and `GpuCsvTable`
still exists for callers who would rather not take it.

What is left of the 18.6 ms is 13.3 of file read and 4.4 of device work. The
next section was the reasoning; the one after it is where the time went next.

## The allocation was the whole cost, and this is why

On `large.csv` — 253 MB — the total is 60.6 ms and the parts that do work add
up to about 18.7:

| | ms |
| --- | ---: |
| total, file to index | 60.6 |
| read into pinned memory | 13.4 |
| upload | 1.3 |
| analyse | 1.2 |
| two prefix scans | 1.2 |
| emit | 1.1 |
| index back | 0.5 |
| **unaccounted** | **~42** |

The unaccounted part is allocation, measured on its own:

| | µs |
| --- | ---: |
| allocate a 253 MB pinned host buffer | 4 501 |
| **first touch of its pages** | **22 462** |
| allocate a 253 MB device buffer | 168 |
| allocate the 90 MB index, device | 367 |
| allocate the 90 MB index, pinned host | 357 |

Page-locked memory is expensive to get and expensive to touch the first time,
and `GpuCsvTable` takes a fresh one of document size on every construction,
plus another of index size. A caller scanning one document and exiting has to
pay that. A caller scanning a directory of documents should pay it once, and
this type gives it no way to.

**The fix is a scanner that owns its scratch.** Something like

```mojo
var scanner = GpuCsvScanner(ctx, capacity=512 << 20)
for path in paths:
    var table = scanner.scan(path)
```

where the buffers are allocated once at the largest size wanted and reused,
growing only when a document does not fit. Everything else stays as it is. On
these numbers that would take the 253 MB document from 60.6 ms to something
near 19, which is where the measured work actually is, and would put it ahead
of the CPU library rather than behind it.

Until that exists the honest summary is the one in the README: the kernels are
about seven times the CPU's parse rate and the type around them gives that
back.

## Reading the file is the next largest piece

13.4 ms of the 18.7 is `FileHandle.read` into the pinned buffer — 17.6 GiB/s,
against the 185 GiB/s the upload manages. Both of these would be worth trying:

- **Read and upload in overlapping slices.** Read a slice, start its upload,
  read the next while that one flies. The scan cannot start until the whole
  document is up, but the read and the upload can overlap almost completely,
  which would hide the smaller of the two.
- **`mmap` the file** and upload from the mapping instead of reading into a
  buffer. Whether a mapping is a fast DMA source or a slow one is not known;
  ordinary pageable memory manages 6 GiB/s against pinned memory's 185, so the
  question is which of those a mapping behaves like.
  [mm_mmap](https://github.com/Mojo-Mania/mm_mmap) is the binding.

## Things that were tried and are not worth trying again

**A kernel reading pinned host memory directly.** It compiles, it runs, it is
fast, and it reads **zeros**. There is no error and no warning. This was
measured at 97 GiB/s and believed for several hours before a separate probe --
writing to a host buffer from a kernel, which also silently does nothing --
prompted checking the values rather than the timings. The lesson is the usual
one: a benchmark that does not verify its answer is measuring something else.

**`std.memory.pack_bits` in device code.** It is what the CPU library uses for
the movemask. Metal's shader compiler does not survive it: the pipeline build
fails with `XPC_ERROR_CONNECTION_INTERRUPTED` and no diagnostic. Multiplying
the comparison result by bit weights and reducing gives the same integer in
four operations and compiles fine.

**Storing the chunk masks instead of recomputing them.** Three `UInt64` per 64
bytes is 0.375x the document in extra memory to save one re-read at 200 GiB/s.
On a device where allocation is already the bottleneck, spending memory to save
bandwidth is the wrong direction.

## Smaller things

- **`scan_totals_kernel` is one block.** It walks the block totals in tiles of
  256 with a running carry, so a 512 MB document puts 32 768 totals through one
  block. It measures at 0.6 ms for both scans together, so this has not been
  worth fixing, but it is the one part of the pipeline that does not scale.
- **The index could be narrower.** The CPU library measured 3-byte slots as
  *slower* because the store count is what costs, not the bytes. On a GPU the
  download is 90 MB and bandwidth-bound, so the trade may go the other way.
  Unmeasured.
- **`is_quoted` and `get` are not here.** Undoing RFC 4180 escaping is
  per-field work on the host; mm_csv does it. A GPU version would have to
  decide where the unescaped bytes live.
