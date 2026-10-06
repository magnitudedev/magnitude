# Lane order

How resident weight codes are ordered in device memory, and why. This is guidance for kernel
authors on every backend. Measurements behind it are in `specs/26-10-04/status.md`.

## Terms

- **Lane**: one thread of the device's cooperative multiply (a Metal simdgroup thread, a CUDA warp
  thread).
- **Patch**: the set of matrix elements one lane contributes to one multiply operand. The device's
  multiply primitive fixes it, up to the freedoms below.
- **Row order**: codes stored one whole weight row after another, as the model file has them.
- **Lane order**: codes stored so that the bytes a lane needs next are one aligned load from one
  contiguous stream. The layout is derived from the patch.

## The rule

Resident weight codes are in lane order for the device's multiply primitive. The layout is a
function of the primitive, not of the chip or the kernel: a backend states its patch, and the
layout follows.

Lane order changes only where codes sit. It never changes the codes, the scales, the weight
memory, or the model file. It is applied once, when weights become resident.

## Why lane over row order

A cooperative multiply wants each lane to hold a small two-dimensional patch of the operand. In row
order that patch is scattered across rows that are kilobytes apart, so prefill has two options, and
both cost:

- **Gather**: each lane issues several small loads per step, from several streams.
- **Stage**: extra threads copy codes into threadgroup memory in patch order, and the multiply
  reads them back. This spends threads, threadgroup memory, and barriers.

In lane order the multiply reads its operand directly. The gain grows as the arithmetic gets
cheaper (integer and packed forms), because the load path is then a larger share of the kernel.

## Deriving the layout for a backend

The primitive fixes the patch only up to three freedoms. Use them in this order:

1. **Operand side.** The weights may be the left or the right operand; the product is transposed
   either way, at no cost in the multiply. Pick the side whose patch runs along a weight row.
2. **Column association.** Which inner-dimension index a fragment column stands for is free, as
   long as the activations are prepared in the same association. Activations are produced every
   step, so their order costs nothing. Pick the association that matches the stored nibble
   arrangement, so codes are extracted with one shift and one mask.
3. **Slot order.** Lanes address their loads by index, so any fixed permutation of lane to storage
   slot is free. Pick the one that makes a simdgroup's loads contiguous.

The target is that **every load is one weight row's bytes, and the rows a simdgroup multiplies
together sit next to each other**. A layout that puts several rows in one load is a last resort:
it serves prefill and taxes decode.

Applying the three steps to both Metal primitives gives the same layout, **tile-row order**:

- weight rows are grouped in tiles of 32;
- inside a tile's code plane, column units of 32 columns follow in order;
- inside a unit, the tile's rows follow one after another, each in its stored arrangement.

Only the code planes are in tile-row order. They interleave because the multiply reads them a
tile at a time. Coefficient planes (scales, minimums, super factors) do not: every consumer reads
them a row at a time (decode per packet or per run of packets; the int8 form once per row and
256-column block in a pre-pass that writes its own table), so inside the tile each row's
coefficients are contiguous, the rows one after another, as in row order.

| Backend primitive | Weights side | A lane's load |
| --- | --- | --- |
| Metal tensor operations (M6), 16 × 32 fragments | left | 8 bytes of one row, two rows per fragment |
| Metal simdgroup matrix (M4 Pro, M1), 8 × 8 fragments | left | 8 bytes of one row, one row per fragment |
| CUDA | derive by the same steps | — |

The table is the derivation, not the engine's prefill kernels. Metal weights are resident in
tile-row order (`rows32`), and every kernel reads them through the packet library, which addresses
a row by its tile and its index in the tile. The exact prefill forms still stage decoded weights,
and the int8 form on tensor operations still has the weights as the right operand and gathers each
row's codes with small loads, now from the tile. Prefill forms that take the weights as the left
operand and read a tile's codes in place are not in the engine yet.

## Decode and prefill

Decode has no cooperative multiply: each lane streams codes, applies a row's scales, and sums. It
is bound by memory traffic, and lane order does not change the bytes read. So decode on lane order
costs the same as on row order, provided the layout keeps one row per load and the kernel is
written for it.

In the standalone harness the fastest mapping for a tile was one weight row per lane with the
simdgroups splitting the columns. In the engine's decode kernels (`dense_output`, `dense_expand`)
on the M4 Pro and M1 it is not (that mapping was never measured in the engine on the M6):

- **Addressing.** A row of a tile is addressed by its tile's base and its index in the tile. That
  arithmetic is 64-bit, and a decode loop that repeats it at every packet runs 20–80 more
  instructions a packet than on row order (30 of 458 in the GEMV at 4 activation rows, 78 of 235
  in the batched GEMV over q6k, which also spilled registers). A lane locates its rows once, and
  the offset inside a tile's plane is 32-bit. With that the batched GEMV is faster on tile-row
  order than on row order for every format.

- **GEMV at one and two activation rows.** The engine keeps the row-order mapping (a lane group
  owns a weight row, its lanes every LANES-th packet of the row), each lane group locating its
  rows once. A form of its own for the tile (one weight row per thread, a simdgroup's lanes
  consecutive rows of a tile walking one contiguous range of packets in step, so their code
  loads are adjacent and each reads its coefficients once per run) was measured and is not in
  the engine yet: the figures below that name it are of a prototype. On the M4 Pro the row
  mapping was the faster of the two; on the M6 the row mapping measured 4–8% slower on tile-row
  order than on row order (a lane group's 16 code loads are 512 bytes apart in a tile) and the
  tile form, tuned, within 1% of tuned row order. From three rows up the tile form lost to the
  row mapping on q4k and q8.
- **Small-row GEMV (3 to 8 activation rows).** The row-order mapping. One weight row per lane with
  the row's packets interleaved among its threads was up to twice as slow from 4 activation rows
  up, where all 32 lanes of a simdgroup read the same staged activations at once; staggering the
  lane groups did not recover it, and the engine does not keep that mapping. Giving each lane a
  contiguous range of the row's packets, so that it reads coefficients once per run, is not
  faster either: equal on tile-row order where every lane's range starts on a 256-column
  boundary, slower elsewhere (the lanes of a simdgroup reload at different steps, so every step
  pays the reload), and 12–18% slower on row order.
- **Batched GEMV (8-row blocks on the matrix units).** A block's lanes should not all load from
  the same cache lines. A lane walks its row in packet order and reads the coefficient planes
  once per run of packets, not once per packet; and a block takes every fourth row of its tile,
  so its rows' packets have a line each.
- **Coefficient planes.** Interleaved across the tile they cost decode, and all of the cost is
  theirs: with the coefficient loads removed, the GEMV at 1 to 8 activation rows is as fast on
  tile-row order as on row order; with the code loads removed instead it is 2–11% slower. A
  plane with one scale a packet (q8: two bytes a row) interleaved per packet put one packet's
  scales of all 32 rows in one cache line, and the q8 batched GEMV was 23% slower than on row
  order.

The one cost lane order can add to decode is rows per load: each extra row a load holds is another
set of scales to decode and another accumulator. Tile-row order adds none.

Prefill is the consumer that gains. Decode must not lose. A layout is accepted for a backend only
when both hold on every device of that backend.

## What is established

| Claim | Evidence |
| --- | --- |
| Operand side is free | M6 800 against 803 µs; M4 Pro 1859 against 1864 µs; M1 equal |
| Prefill on tile-row order matches a multi-row lane order on the M6 | 812 against 803 µs; gathered from row order 926 µs |
| Packing on tile-row order against the eight-row patch (harness probe; the engine has no packing form) | M1 7542 against 7564 µs; M4 Pro 1894 against 1858 µs (1.9% short, open) |
| Decode on tile-row order (harness, one row per lane) | M6 +1.4%, M4 Pro −5%, M1 +1.2% against row order |
| Engine single-row GEMV on tile-row order | M4 Pro q4k 56.1 against 56.6 µs, q6k 81.0 against 84.2, q8 59.5 against 59.1; M6 q4k 107.7 against 101.1, q6k 156.2 against 150.0, q8 104.6 against 98.7; M1 (before the addressing change) q4k 241.2 against 237.0, q6k 339.7 against 346.0, q8 222.8 against 222.7 |
| Engine GEMV at 4 activation rows | M4 Pro q4k 98.6 against 96.3 µs, q6k 113.3 against 108.0, q8 77.4 against 75.5; M6 q4k 154.0 against 156.0, q6k 184.6 against 180.4, q8 118.2 against 115.5 |
| Prototype single-row form (one row per thread, contiguous ranges; not in the engine) on tile-row order against the row mapping on row order | M6 q4k 99.8 against 99.7 µs, q6k 146.0 against 149.4, q8 99.6 against 98.4; M4 Pro q4k 57.9 against 56.4, q6k 82.9 against 83.0, q8 60.3 against 59.0 |
| Engine single-row GEMV, one row per lane with interleaved packets (removed) | M4 Pro q4k 62.4 against 58.6 µs for the row-order mapping; M1 262.5 against 239.6 |
| Engine batched GEMV on tile-row order (4 rows) | M4 Pro q4k 153.4 against 155.0 µs, q6k 163.0 against 170.4, q8 100.6 against 105.7; M6 q4k 157.6 against 164.9, q6k 178.0 against 207.0, q8 108.0 against 139.2 |
| Multi-row patches cost decode | eight rows per load: M4 Pro +22%, M1 +12%, M6 +6% |

## What is open

- The last 1–1.5% of decode on the M6 and M1, and 1.9% of packing on the M4 Pro.
- The prefill figures are timing-only probes; the weights-left kernels still need a correctness
  check against the exact form.
- The GEMV at one activation row on the M6: the engine's row mapping is 4–8% slower on tile-row
  order than on row order. The prototype tile form, tuned against tuned, is about 1% behind row
  order for `dense_output` (q4k 98.2 against 97.3 µs, q6k 144.9 against 143.2, q8 96.7 against
  95.8); it is not in the engine yet.
- The GEMV at 2 to 8 activation rows is 1–5% slower on tile-row order on the M4 Pro and up to 3%
  on the M6. It is not instruction count (454 against 449 a packet at 4 rows), not the staged
  chunk size, and absent with either the code loads or the coefficient loads removed; reading
  the coefficients from next to the codes does not remove it. The cause is not established.
- The M1 after the addressing change. Its figures vary by several percent between runs and do
  not resolve differences of this size.

## Measuring layouts

Fill every buffer under comparison with one sequential copy. On the M6 a buffer first touched in
an interleaved order reads 15% slower under any kernel, which twice produced a false layout
penalty in this study.
