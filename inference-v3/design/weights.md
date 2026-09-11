# Weights

**A container is forgotten at residency. Every execution kernel reads an
engine-defined layout whose parameters express numerical meaning, not a source
format or packing name.**

## From container to kernel

```text
container ──► stored weight ──► residency ──► resident weight ──► command
 parser          source spans      bounded       representation      retains the
 and codec       and codec         import        layout + rows       allocation
```

| Stage | Owns | Never knows |
|---|---|---|
| Format | Parsing, validation, artifact identity, wire geometry, source codec | Execution schedules |
| Residency | Representation choice, bounded relayout or conversion, one allocation | Model semantics |
| Resident weight | Numerical representation, canonical layout, backing extent, logical rows | Source codec or container metadata |
| Kernel | Canonical addresses and numerical interpretation | Where bytes came from |

## Numerical representations

A representation is a parameter value. Containers with equal parameters share
the same readers and schedules.

| Meaning | Parameters | Examples |
|---|---|---|
| Dense | dtype | resident FP32 parameters |
| Affine | code planes and interpretation, group, coefficient scheme | MLX Q4, Q4_K, Q5_K, Q6_K, Q8_0 |
| Codebook | code width and table, group, coefficient scheme | IQ4_XS |

Codes distinguish unsigned, offset-binary, and two's-complement
interpretations. Coefficients are either direct floating scale/bias values or a
compact local/supergroup hierarchy. Bias presence and sign belong to the
coefficient scheme; they are not independent packing flags.

Codebook is a separate representation because table lookup changes numerical
meaning. Its immutable table is folded into a specialization and consumes no
resident bytes.

## Canonical layout

`canonical_layout` is the only authority for field offsets and allocation
size. Hierarchical tiles contain, in order:

```text
low codes · high codes · local scales · local biases · super scale · super bias
```

Absent fields consume no space. Codes and local coefficients are bit-contiguous
and little-endian within their fields. A direct affine matrix uses the same
order as matrix-wide planes, which lets MLX codes, scales, and biases copy
directly into their final ranges.

Canonical relayout is a permutation, not dequantization. It cannot widen codes
or coefficients, add per-tile padding, precompute products, or create a second
resident copy. Current compact sizes therefore remain 36 bytes per 64 MLX Q4
values and 144/176/210/34/136 bytes for Q4_K/Q5_K/Q6_K/Q8_0/IQ4_XS blocks.

## Import

A quantized format supplies one stable source codec per wire encoding. The
codec declares source tile geometry and reads logical codes and coefficients.
Residency specializes one generic relayout writer with that codec, processes
complete tiles through bounded two-slot staging, and writes directly into the
final canonical allocation. Reading and uploading the next slot overlaps the
preceding relayout submission. Codec and wire-enum knowledge never enter
inference. The codec boundary is typed as trace buffers and scalar expressions;
it does not expose schedules or an untyped kernel API.

Dense conversion is separate. Stored F16 and BF16 values widen to resident
FP32, with any declared transform applied once. MLX affine planes already have
canonical logical order and are copied without a relayout kernel.

A failed import closes the unpublished target and all staging. A successful
import publishes only the final allocation.

## Ownership and grouping

| Rule | Reason |
|---|---|
| One allocation per weight or compatible row-concatenated group | Every consumer sees the same bytes |
| A group is declared before any member becomes resident | Grouping later would require a copy |
| A group member carries a logical row range over the complete layout | Canonical fields need not form one byte-contiguous member slice |
| Every kernel binds the complete allocation once | No format-specific multi-plane binding contract leaks upward |
| Commands retain allocation views | Sources and residency owners may retire after submission |

Grouping preserves bytes exactly. Quantized members must have equal numerical
representations and complete source tiles per row; declared transforms prevent
grouping.

## Extension

A new container implements parsing and source codecs. If its values match an
existing representation, it needs no inference change. A representation is new
only when its numerical meaning cannot be expressed by `Dense`, `Affine`, or
`Codebook`; a new source packing alone never creates a representation or fast
path.
