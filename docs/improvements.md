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

## Narrowing the index: tried, and it loses here too

The CPU library found 3-byte slots slower because the store *count* is what
costs there, not the bytes. On a GPU the index is 85 MiB and has to be
downloaded, so the trade looked like it might go the other way. It does not.

Same walk, same number of stores, same alignment, half the bytes:

| | µs |
| --- | ---: |
| emit, 4-byte entries | 1158 |
| emit, **2-byte** entries | **1479** |
| download 4-byte, 85 MiB | 489 |
| download 2-byte, 43 MiB | 269 |

Halving the entry makes the emit **28% slower** and the download 220 µs
faster: **net 101 µs worse**. A sixteen-bit store per lane does not serve the
memory system as well as a thirty-two-bit one, and that costs more than the
bytes saved are worth.

That is before the ceiling, which settles it anyway. The offset needs the
whole entry bar the CRLF flag, so two bytes caps a document at 32 KiB and
three bytes at 8 MiB. This library exists for documents in the hundreds of
megabytes. Narrowing trades away three orders of magnitude of document size
for a change that is negative.

**The first attempt at measuring this was invalid**, in exactly the way this
repository's sibling already had written down. The plan was to time the emit
with the store removed, to price the writes; the variant accumulated into a
register instead, which creates a serial dependency the real loop does not
have, and it measured *slower* than storing -- 1212 µs against 1159. mm_csv's
re-profile records the same trap in the same words. Knowing about a trap and
walking into it are different skills.

## Done: the prefix scan goes two levels deep

`_scan` used to be one level: scan each block's slice, then hand *every* block
total to a single block that walked them in tiles with a running carry. A
chunk is 64 bytes and a block scans 256 of them, so that inner walk was 61
serial tiles on a 253 MB document and 512 on a 2 GiB one, on one block, while
the rest of the device idled.

It now scans the totals the same way it scans the values, and only the totals
*of those* go to the single-block kernel. Two levels covers everything under
the 2 GiB ceiling: 33.5 M chunks, 131 072 totals, 512 totals of those, and the
innermost kernel walks two tiles.

Measured on the scan alone, same data, same process:

| document | block totals | one level | two levels |
| --- | ---: | ---: | ---: |
| 64 MiB | 4 096 | 519 µs | **220 µs** |
| 253 MiB | 16 192 | 644 µs | **519 µs** |
| 1 GiB | 65 536 | 2 103 µs | **1 670 µs** |

**End to end this is worth about 1.4%** -- two scans at 0.52 ms against two at
0.64 -- which is under the noise floor of the benchmark, and the warm total
did not visibly move. It is a scaling fix rather than a speed fix, and the
honest reason to have it is the 2 GiB column rather than the 253 MB one.

## Smaller things

- **`scan_totals_kernel` being one block: fixed.** See below.
- **Narrowing the index: measured, and it loses.** See below.
- **`is_quoted` and `get` are not here.** Undoing RFC 4180 escaping is
  per-field work on the host; mm_csv does it. A GPU version would have to
  decide where the unescaped bytes live.
