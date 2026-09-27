---
applies_to:
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
into `iq4g32` (the same table). Such a format adds no execution class, so the assessment
measurement basis, keyed by resident representation, covers it unchanged, and the plan charges
the wider representation's resident bytes. A format no representation holds exactly is not
imported; its model is `Incompatible`.
Q4_0, Q5_0, Q5_1, MXFP4 and NVFP4 import into representations of their own, moving codes and
scale fields bit for bit: the 4-bit coded family (a codebook and one scale per 32 or 16 values:
`q4g32s` offset codes with f16 scales, `mxfp4g32` E2M1 with an E8M0 exponent per 32, `nvfp4g16`
E2M1 with a UE4M3 scale per 16, beside `iq4g32`) and the 5-bit `q5g32s` (f16 scale) and `q5g32`
(f16 scale and minimum). E2M1 −0 imports as +0, as the reference dequantization decodes it. A
scale field that is NaN in its format (E8M0 0xff, UE4M3 0x7f) decodes as NaN. NVFP4's per-tensor
or per-expert F32 scale is a separate tensor, not part of the representation; the family applies
it.

Each import takes an immutable artifact source and validates its exact WeightPlan. On Metal,
component weights are visited in source-file order. Consecutive whole tensors whose combined
range fits the largest source tensor share one page-rounded read-only mapped window and one
ordered native submission. Their resident destinations are allocated without a host zero-fill;
the attested import entries write every physical byte, including representation padding.
The mapped window stays owned through completion. Other backends use a one-shot staged source
upload, also without a prefill. The source file streams into that upload in bounded chunks, without
a second whole-tensor host copy. Residency publishes each weight
only after its submission completes; failed imports leave no cache entry.

The target component is imported before engine readiness. Enabled optional head and vision
components are held by typed one-shot ComponentLoaders. Each loader owns its import store and
caches either its assembled component or its typed failure. The head loader inherits the target
store so tied embedding and output weights retain their exact resident tensors; the vision loader
owns an isolated projector store. A separate draft (DFlash, DSpark) is the head lane's drafter: its
fusion weights are target weights (every target step fuses the draft's taps), imported with the
target from the draft component, and its loader imports the rest of the draft component through
the inherited target store. Numerical stages receive loaders rather than shared mutable
cache access. No warm token path prepares or searches for an import kernel.
