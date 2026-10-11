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
the execution path and backend (native Metal `rows32`, native Vulkan `rows16`, native CUDA `mma16`,
native CPU `rows8`, planned `packet`). Import converts through the weight's `[B, N, K]` view, which
keeps every layout's row geometry; it never flattens a weight.
A backend's layout is derived from its cooperative multiply (`inference/docs/kernels/lane-order.md`).
Metal's `rows32` groups a matrix's rows in tiles of 32 and pads the row axis to a whole tile. Inside
a tile each code plane holds, per 32 columns, the tile's rows' codes one row after another, so one
aligned load is one row's codes of one packet and a tile's loads are contiguous; coefficient planes
hold the tile's rows one after another, each row's bytes contiguous. A layout changes only where
codes and coefficients sit: values, the model file and the resident bytes of whole tiles are those
of `rows16`. The plan charges the padded tiles, and import writes them, as it does every layout's
padding.
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
The canonical output projection retains its packed representation. Eligible Q4_0 and Q8_0
sources can also provide lossless bounded-selection views, as five
`ImportTransform::Progressive` planes: the coarse half of the offset codes, their next bit,
remaining bits, group scales, and outward-rounded view radii. The code depth remains explicit;
the radii cover the coefficient-view distance and both projections' F32 rounding. Each view is a
separate physical storage identity, while semantic users of the same view share its allocation.
Candidate views are transient preparation inputs, not baseline resident requirements. A completed
first-load composed qualification selects the bounded recipe only for measured selected-row
classes of the exact projection/norm/prepared-feature recipe. It shares the existing one-minute
allocation with entry tuning; an unfinished or unmeasured choice remains packed permanently for
that cache identity. Later loads reuse the completed choice and do not continue searching.
Before admission the load plan retains planes only for selected recipes and drops unused
candidates. Packed and retained planes are charged together through ordinary resource planning;
optional views cannot make an otherwise admitted packed model fail to load. The host places
planes directly from the stored rows. Host placement takes no import entry or mapped import
window. All-logits and indexed projections use canonical packed storage.

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
the inherited target store. Vocabulary projection ownership and model-defined proposal candidate policy are independent.
A trained cluster policy binds its centroid projection, canonical token ordering and selected
cluster count; execution preserves that policy without approximating target readout. Readout
representation eligibility follows required operations rather than artifact ownership.
An owned draft output projection is planned and charged
to the draft component and imported from that artifact; a borrowed projection reuses
the target allocation. Both bind the same readout entry and resident output field. Every preload reads each weight from the component that stores it. A draft head stored in the draft component (`HeadComponent::Draft`) is planned, charged and
imported from that component like a separate draft's own weights, while its token embedding is the
target's resident table.
Numerical stages receive loaders rather than shared mutable
cache access. No warm token path prepares or searches for an import kernel.
