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
next section was the reasoning, written before the scanner existed; the one
after it is where the time went next.

All of this is Apple M4 Max. On a laptop RTX 4050 one-shot and reused are
within 2% of each other, so this fix is worth nothing there -- see the last
section.

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
against the 185 GiB/s the upload manages. Both of these have since been
tried. Overlapping the read with the upload shipped, and took the one-shot
path from 58.8 ms to 40 and the warm one by about 2%. `mmap` does not help:
`enqueue_copy_from` over a mapping crashes the runtime, and filling the pinned
buffer by memcpy from one is 38% slower than reading into it. The ideas as
first written:

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
four operations and compiles fine. (That movemask is gone too now: the chunk
is read as SWAR words -- see "On a discrete card" below.)

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

- **`scan_totals_kernel` being one block: fixed.** See above.
- **Narrowing the index: measured, and it loses.** See above.
- **`is_quoted` and `get` are not here.** Undoing RFC 4180 escaping is
  per-field work on the host; mm_csv does it. A GPU version would have to
  decide where the unescaped bytes live.

## On a discrete card: SWAR for the kernels, and then the link

First measured on a laptop RTX 4050 (PCIe 4.0 x8, CUDA 13.3, Linux) next to an
AMD Ryzen AI 9 HX 370. All tests pass there.

### Done: read the chunk as words, not vectors

The first run had analyse at 7.8 ms and emit at 9.0 on the 253 MB document --
25-30 GiB/s, against 200-220 on the M4 Max -- while the prefix scan on the
same card ran at over 1 000 GiB/s. So launching kernels was fine and the two
passes that read the document were not.

`dump_llvm` on the launch says why. NVIDIA's backend does not keep `SIMD`
vectors as vectors. `unsafe_load[width=16]` becomes a sixteen-iteration loop
of `insertelement`, four of them per chunk, and the compares and weighted
reduces come out as scalars: about 950 lines of NVPTX assembly per chunk.

`chunk_masks` now loads eight `UInt64` words and finds each byte class with
the exact SWAR equality -- XOR, add `0x7F` to the low seven bits, keep the high
bits that stayed clear -- and a multiply that gathers the eight flags into one
byte. Measured in isolation, same buffers, same process, outputs compared entry
for entry:

| `large.csv` | `SIMD` | SWAR |
| --- | ---: | ---: |
| analyse | 9 780 µs | **1 878 µs** |
| emit | 8 959 µs | **2 038 µs** |

End to end, 46.4 ms to **33.2** on `large.csv` and 5.4 to 4.1 on `small.csv`.
Breaking the SWAR equality fails ten of the eleven tests, so they do run it.

**Faster on Metal too, so it is the only version.** This first went in behind
`comptime if is_apple_gpu()`, with Apple keeping `SIMD` because that is what
its 200 GiB/s had been measured with. On the M4 Max, the two builds differing
only in that branch, alternated, three runs each at best of 100 instead of the
benchmark's five -- at five the kernel rows jump between two levels:

| GiB/s | `SIMD` | SWAR |
| --- | ---: | ---: |
| `large.csv` analyse | 277-291 | **314-340** |
| `large.csv` emit | 275-288 | **336-364** |
| `quoted.csv` analyse | 281-283 | **322-345** |
| `quoted.csv` emit | 279-281 | **362-385** |
| `large.csv` one prefix scan | 531-576 | 526-583 |

The prefix scan does not read the document and is the control. All 12 tests
pass with the words on Metal, and `metal:4`, `metal:1`, `sm_80` and `gfx942`
still compile. The reused scan does not move -- 17.0-17.2 ms against
17.0-17.3 on `large.csv` -- because about 0.3 ms of kernels is lost in a
13 ms file read. It is kept anyway: one path instead of two, and no device
where it is slower. The `SIMD` version and the weighted-reduce movemask it
needed are gone.

### Tried: letting kernels touch pinned host memory

On Metal a kernel reads zeros from a pinned host buffer. On CUDA it reads the
right values, so zero-copy was worth pricing on `large.csv`:

| | ms |
| --- | ---: |
| analyse reading the pinned host document | 30.4 |
| upload, then analyse on the device | 20.4 |
| emit writing into a pinned host index | **1 016** |
| emit on the device, then download | 8.9 |

A kernel reading host memory pays the bus on every access instead of once in
bulk, and writing the index that way is a hundred times worse. Not worth
trying again. `is_host_unified()` is false on this card, as expected.

### Done: download the index while the document is still going up

Timed at the scanner's synchronisation points, the 33 ms whole-document scan
was 21.9 of read and upload overlapping, 2.3 of analyse and both scans, and
8.9 of emit and download. The download sat idle until the whole document was
up, and PCIe is full duplex.

**First, whether the card can do it.** Raw CUDA, two streams, pinned memory,
a 253 MB upload and an 85 MB download: 26.5 ms one after the other, 19.9 at
once. So the hardware has a copy engine each way.

**Then, why MAX could not.** The same two copies issued through MAX on two
streams took 26.5 ms either way -- until the download's buffers were created
*on the second stream*. Then 19.9, the same as raw CUDA, whichever copy call
was used. A copy through a buffer waits for the stream that created it.
Kernels take raw pointers and do not, so the emit on the first stream writes
into the second stream's index freely.

**The scan, sliced.** Everything a chunk needs from the chunks before it is
the quote state and its first index entry. So on a discrete device the
document goes a slice at a time: read, upload, analyse, both scans and select
on the first stream, with the quote state carried to the next slice in a
one-element device buffer that `select` folds in and `fold_parity_kernel`
advances -- no host involvement. The slice's field count is copied back on
the second stream. One slice later the host has that count, which says where
the slice's entries go, and enqueues its emit on the first stream and its
download on the second, behind an `enqueue_wait_for`. The index grows between
slices when it has to, after both streams drain; a scanner that has seen a
document that size never does.

With unified memory none of this helps -- the index comes back at 500 GiB/s --
so that path is unchanged and still scans the document whole.

Slice size, reused scanner, best of five:

| slice | `small.csv` | `large.csv` |
| --- | ---: | ---: |
| 1 MiB | 2.9 ms | 30.0 ms |
| 2 MiB | 2.8 | 27.8 |
| **4 MiB** | **2.8** | **27.2** |
| 8 MiB | 3.3 | 27.4 |
| 16 MiB | 4.2 | 28.0 |
| 32 MiB | 4.6 | 29.4 |
| 64 MiB | 4.6 | 32.3 |

Small slices pay a host synchronisation each; large ones leave the last
slice's download with no upload to hide behind, which is why `small.csv`, two
slices at 16 MiB, gained nothing until they shrank. The same scanner forced
down the whole path on the same machine measures 33.3 ms and 4.1, so the
gain is 18% and 30%.

**Tests.** `test_quotes_across_stream_slices` puts quoted regions and CRLFs
across 4 MiB boundaries and checks one document streamed, whole, and streamed
again against the scalar reference. Not folding the parity fails it and only
it. Ignoring `start` in the emit fails it and the two other multi-slice
tests. Dropping the download's wait for the emit crashes the suite: the
column count reads the document at whatever garbage offsets came back.

### What is left is the upload

The upload runs at 12.5 GiB/s, which is PCIe 4.0 x8 full, so more streams or a
faster read cannot help it; 18.8 ms of the 27 is that. What could still move:

- **Download less.** Keeping the index on the device for further GPU work, or
  returning only what a caller asks for, removes the rest of the download and
  changes the API.
- **The Radeon 890M on the same laptop shares RAM with the CPU**, like Apple
  silicon. MAX does not see it here -- `gpu-query` lists only the RTX 4050 and
  ROCm is not installed -- so whether it would behave like the M4 Max is
  untested.

Even at zero cost for every kernel, the upload alone here is 18.8 ms against
the CPU's 14.1 ms parse. On this card the GPU is not a win for this problem
unless the index stays on the device for further GPU work.
