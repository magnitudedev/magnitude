---
applies_to:
  - inference/engine/executor/src/progressive.rs
  - inference/engine/executor/src/import_transforms.rs
  - inference/engine/executor/src/residency.rs
  - inference/engine/executor/src/resident_weights.rs
  - inference/engine/executor/src/planning/weights.rs
  - inference/engine/kernels/kernels/*/repack_weight.*
  - inference/seismic/native-cpu/src/repack.rs
---

# Device residency

The execution plan identifies every source tensor, artifact component, resident representation,
and byte charge before a device is opened. ResidencyStore is the sole importer and cache type for
device weights. Its key contains artifact identity, tensor name, and resident representation with
its layout, so tied weights share physical storage while distinct components, representations, and
layouts remain separate. The layout is the one fastest for the opened backend's kernels and may
differ by backend: one map picks the representation from the source format and the layout from
the execution path and backend (native Metal and Vulkan `rows16`, native CUDA `mma16`, native CPU and planned
`packet`). Import converts through the weight's `[B, N, K]` view, which keeps every layout's row
geometry; it never flattens a weight.
Every import is exact: each resident value equals the source format's reference dequantization
bit for bit. A source format without a representation of its own imports, through a registered
Seismic conversion, into an existing representation that holds every value it encodes: Q3_K and
IQ3_S into `q6k` (f16 super-scale × int8 scale per sixteen values × code in [-32, 31]), IQ4_NL
into `iq4g32` (the same table). Such a format adds no execution class, and the plan and the
assessment's decode demand charge the wider representation's resident bytes. A format no representation holds exactly is not
imported; its model is `Unsupported`.
Q4_0, Q5_0, Q5_1, MXFP4 and NVFP4 import into representations of their own, moving codes and
scale fields bit for bit: the 4-bit coded family (a codebook and one scale per 32 or 16 values:
`q4g32s` offset codes with f16 scales, `mxfp4g32` E2M1 with an E8M0 exponent per 32, `nvfp4g16`
E2M1 with a UE4M3 scale per 16, beside `iq4g32`) and the 5-bit `q5g32s` (f16 scale) and `q5g32`
(f16 scale and minimum). E2M1 −0 imports as +0, as the reference dequantization decodes it. A
scale field that is NaN in its format (E8M0 0xff, UE4M3 0x7f) decodes as NaN. A second-level
scale (NVFP4's per-tensor or per-expert F32 `.scale`) is a separate stored tensor, not part of the
representation: the weight's bytes import as stored, and the scale becomes resident beside it as
one F32 value per matrix (a stored single value repeats per matrix of a stack), keyed and charged
with the weight identically by header assessment and load. Entries apply it to their F32
accumulator through an accumulator-scale port; no import or family folds it into another weight.
A missing or misshapen scale, a scaled weight dequantized for dense-only kernels, and a scaled
weight bound by an entry without a scale port are refused at plan time, so the model is
`Unsupported` rather than failing on a device.
A progressive head replaces the stored Q8_0 head by its five planes (`ImportTransform::Progressive`,
each its own `WeightKind::OutputPlane` weight): the offset-binary codes' bits 7..4, bit 3 and bits
2..0, the group scales, and per row the radii of its 4- and 5-bit views (`[V, 2]`, so every
plane's leading rows are a draft head's draft-vocabulary rows; in F32 rounded up, the
Euclidean distance between the row and its view plus `4 · K · 2⁻²⁴ · |w|₂`, which covers both
projections' F32 rounding). The planes hold the head's values bit for bit, so the head's resident
bytes are unchanged and the radii add `8 · V` bytes. The host places each plane from the stored
rows and the device holds the placed bytes as they are: a host-placed weight takes no import entry
and joins no mapped import window. A draft head binds the target's resident planes, imported once.

Each import takes an immutable artifact source and validates its exact WeightPlan. On Metal,
component weights are visited in source-file order. Consecutive whole tensors whose combined
range fits the largest source tensor share one page-rounded read-only mapped window and one
ordered native submission. Their resident destinations are allocated without a host zero-fill;
the attested import entries write every physical byte, including representation padding.
The mapped window stays owned through completion. Resident weights on macOS are wired like every
engine holding ([engine memory](memory.md)); the mapped window is file-backed and is not. Other backends use a one-shot staged source
upload, also without a prefill. The source file streams into that upload in bounded chunks, without
a second whole-tensor host copy. Residency publishes each weight
only after its submission completes; failed imports leave no cache entry.

The target component is imported before engine readiness. Enabled optional head and vision
components are held by typed one-shot ComponentLoaders. Each loader owns its import store and
caches either its assembled component or its typed failure. The head loader inherits the target
store so tied embedding and output weights retain their exact resident tensors; the vision loader
owns an isolated projector store. A separate draft (DFlash, DSpark, DFlash2) is the head lane's drafter: its
fusion weights are target weights (every target step fuses the draft's taps), imported with the
target from the draft component, and its loader imports the rest of the draft component through
the inherited target store. Every preload reads each weight from the component that stores it.
Numerical stages receive loaders rather than shared mutable
cache access. No warm token path prepares or searches for an import kernel.
