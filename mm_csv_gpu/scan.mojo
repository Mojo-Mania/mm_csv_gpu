"""An exclusive prefix sum over an array of any length.

Three kernels. Each block scans its own `THREADS` elements and reports its
total; one block scans those totals; every block then adds its own base to
what it wrote. That is the standard shape, and it is here rather than in
`kernels.mojo` because the CSV scan needs it twice for two different things --
once over quote parities, where the low bit of the result is what matters, and
once over delimiter counts, where the whole number is.

Both uses are sums. An exclusive prefix *XOR* over one-bit values is the low
bit of an exclusive prefix sum over the same values, so the parity pass needs
no separate implementation.
"""

from max.gpu import barrier, block_dim, block_idx, thread_idx
from max.gpu.memory import AddressSpace
from std.memory import stack_allocation

from .kernels import THREADS


@always_inline
def _scan_tile(
    shared: Pointer[
        UInt32, MutUntrackedOrigin, address_space=AddressSpace.SHARED
    ],
    t: Int,
):
    """Turns `shared` into its inclusive prefix sum, in place.

    Hillis-Steele: every thread adds the value `step` places to its left, and
    `step` doubles. `THREADS` must be a power of two, which it is.
    """
    var step = 1
    while step < THREADS:
        var addend = UInt32(0)
        if t >= step:
            addend = shared[unsafe_offset=t - step]
        barrier()
        if t >= step:
            shared[unsafe_offset=t] += addend
        barrier()
        step *= 2


def scan_block_kernel(
    values: Pointer[UInt32, MutUntrackedOrigin],
    scanned: Pointer[UInt32, MutUntrackedOrigin],
    block_totals: Pointer[UInt32, MutUntrackedOrigin],
    n: Int32,
):
    """Scans each block's slice and reports the block's total.

    Args:
        values: Input.
        scanned: Output, each block's exclusive scan of its own slice.
        block_totals: Output, one total per block.
        n: How many values there are.
    """
    var shared = stack_allocation[
        THREADS, UInt32, address_space=AddressSpace.SHARED
    ]()
    var t = Int(thread_idx.x)
    var gid = Int(block_idx.x) * THREADS + t
    var value = values[unsafe_offset=gid] if gid < Int(n) else UInt32(0)
    shared[unsafe_offset=t] = value
    barrier()

    _scan_tile(shared, t)

    var inclusive = shared[unsafe_offset=t]
    if gid < Int(n):
        scanned[unsafe_offset=gid] = inclusive - value
    if t == THREADS - 1:
        block_totals[unsafe_offset=Int(block_idx.x)] = inclusive


def scan_totals_kernel(
    totals: Pointer[UInt32, MutUntrackedOrigin],
    scanned: Pointer[UInt32, MutUntrackedOrigin],
    grand: Pointer[UInt32, MutUntrackedOrigin],
    n: Int32,
):
    """Scans the block totals, in one block, however many there are.

    A single thread walking them would serialise a quarter of a million
    additions on a large document, so this walks them in tiles of `THREADS`
    with a running carry every thread agrees on.

    Args:
        totals: Input, one per block of the previous kernel.
        scanned: Output, their exclusive prefix sum.
        grand: Output, element 0 is the sum of everything.
        n: How many totals there are.
    """
    var shared = stack_allocation[
        THREADS, UInt32, address_space=AddressSpace.SHARED
    ]()
    var t = Int(thread_idx.x)
    var count = Int(n)
    var running = UInt32(0)
    var start = 0

    while start < count:
        var gid = start + t
        var value = totals[unsafe_offset=gid] if gid < count else UInt32(0)
        shared[unsafe_offset=t] = value
        barrier()

        _scan_tile(shared, t)

        var inclusive = shared[unsafe_offset=t]
        if gid < count:
            scanned[unsafe_offset=gid] = running + inclusive - value
        # Every thread reads the same tile total, so `running` stays in step
        # across the block without anything being broadcast.
        var tile_total = shared[unsafe_offset=THREADS - 1]
        barrier()
        running += tile_total
        start += THREADS

    if t == 0:
        grand[unsafe_offset=0] = running


def scan_apply_kernel(
    scanned: Pointer[UInt32, MutUntrackedOrigin],
    block_offsets: Pointer[UInt32, MutUntrackedOrigin],
    n: Int32,
):
    """Adds each block's base to the scan it wrote.

    Args:
        scanned: Updated in place.
        block_offsets: Where each block's slice starts.
        n: How many values there are.
    """
    var gid = Int(block_idx.x) * THREADS + Int(thread_idx.x)
    if gid < Int(n):
        scanned[unsafe_offset=gid] += block_offsets[
            unsafe_offset=Int(block_idx.x)
        ]
