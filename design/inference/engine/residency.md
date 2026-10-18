---
applies_to:
  - inference/engine/executor/src/progressive.rs
  - inference/engine/executor/src/import_transforms.rs
  - inference/engine/executor/src/residency.rs
  - inference/engine/executor/src/resident_weights.rs
  - inference/engine/executor/src/streamed_table.rs
  - inference/engine/executor/src/device_table.rs
  - inference/engine/executor/src/slot_directory.rs
  - inference/engine/executor/src/stored_table.rs
  - inference/engine/artifacts/src/uncached.rs
  - inference/engine/kernels/kernels/*/stored_rows.*
  - inference/engine/kernels/kernels/*/selection_notice.*
  - inference/engine/executor/src/host_tables.rs
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
scale field that is NaN in its format (E8M0 0xff, UE4M3 0x7f) decodes as NaN. IQ2_XXS, IQ2_XS
and IQ3_XXS import into vector-coded representations of their own (`iq2xxs`, `iq2xs`,
`iq3xxs`): each eight-value entry (its grid indices and seven sign bits, the eighth of even
parity) and each 4-bit sub-block scale code move bit for bit, and the f16 block scale becomes an
F32 factor scaled by a power of two, exactly. A second-level
scale (NVFP4's per-tensor or per-expert F32 `.scale`) is a separate stored tensor, not part of the
representation: the weight's bytes import as stored, and the scale becomes resident beside it as
one F32 value per matrix (a stored single value repeats per matrix of a stack), keyed and charged
with the weight identically by header assessment and load. Entries apply it to their F32
accumulator through an accumulator-scale port; no import or family folds it into another weight.
A scaled weight's stored `.input_scale` (the activation scale of kernels that quantize their
inputs to FP4) is checked to the scale's shape and acknowledged, never imported: activations stay
in the activation dtype, so the product needs only the weight's scale.
A missing or misshapen scale, a scaled weight dequantized for dense-only kernels, and a scaled
weight bound without its scale (by an entry without a scale port, or by an operator that binds
the port absent) are refused at plan time, so the model is
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

A gathered table (a weight a step reads by row index: the token embedding, a per-layer input table,
an n-gram input table) is streamed instead of held whole wherever it can be. There is no setting: the weight plan decides,
from the artifact alone, and admission charges what the plan says. A streamed table's results equal
the whole table's bit for bit.
- **A table the host gathers** claims a row cache of at most 16,384 rows in system RAM instead of
  its byte range. Its steps run the graphs and the reads of the whole table's steps; a held row is
  not read again. The host names its rows from host tokens (a per-layer table's row is its
  token's), so its steps run on host tokens: a model with one has no lookahead. An n-gram table is
  a table the device gathers: the device names its rows from the step's token rows
  ([model input](model-input.md)); the host fills its slots (below).
- **A table the device gathers** is streamed unless:
  - another role of its component reads the same stored tensor (a tied output projection reads
    every row);
  - it has no more than 16,384 rows;
  - its rows are not whole tiles of the backend's row-gathering layout that lays out the fewest
    rows together;
  - its rows do not reach the device as stored (the host prepares them, or a second-level scale
    accompanies them).

  Otherwise it is held whole, as planned for any weight.
- **A streamed device table** is one resident weight: its slots, a directory and staging for the
  rows one step brings, all charged to the weight. Its plan's shape is its slots'. Every program
  that reads the table (the target, an executed drafter, a head) shares it through the residency
  cache. An imported table has slots for 16,384 rows; a row reaches a slot the first time a step
  names it, read from the artifact and converted by the weight's own import entry, queued ahead of
  that step.
- **The directory's form** is the plan's, from the table's rows and slots alone: dense (`[rows, 1]`,
  every row's slot or −1) when the rows are at most 16 times the slots, else hashed (`[buckets, 2]`
  of (row, slot), buckets the least power of two at least twice the slots, murmur3 `fmix32` of the
  row masked to the buckets, linear probing). In a hashed directory −1 is an empty bucket, which
  ends a probe, and −2 a removed row's tombstone, which a probe passes, so a reader never misses a
  row a step names while another row leaves. The host rebuilds a hashed directory only when no
  queued step reads it, once live rows and tombstones exceed three quarters of its buckets.
- **A stored table** (an n-gram table, `TableFill::Stored`) is streamed whatever its rows: its
  slots hold rows as stored (IQ4_NL at 18 bytes or Q8_0 at 34 bytes per 32 values), never
  converted, and its reader (`stored_rows`) decodes a slot's blocks as their registered conversion
  (`iq4g32s` or `q8g32s`) does. It has slots for 180 × 16,384 rows (all of them, of a table of no more), no
  staging, and the load uploads nothing. Its slots and directory are host-shared storage that the
  host writes without device work: one filler thread per table takes steps in queue order, reads
  the rows a step names that no slot holds from the artifact (sorted, coalesced and bypassing the
  page cache: `O_DIRECT`, `F_NOCACHE` or `FILE_FLAG_NO_BUFFERING`), writes their slots and
  directory entries, then opens the step's gate (a [host gate](execution-plan.md) ahead of the
  reader). After the reader, the step signals that it has read. A step's rows are the n-gram rows
  of its slots' tokens and the tokens each sequence holds before them (`NGramHash::rows`, the rule
  the device's `ngram_rows` evaluates): of host tokens the host knows them when it queues the step;
  of selected tokens the filler waits for the step's entry graph to copy the selection into
  host-shared storage and signal (the selection notice). A selected row is never a miss.
- **A stored table's slot is reused** only once every step that named its row has signaled its
  read, never for a row the step being filled names. A filler waits for an earlier step's read
  before taking a slot it named. A row that cannot be read is not published and fails the batch
  slots (sequences) that named it; a failure that is no row's (the selection never reached the
  host, no slot came free) fails every slot. The step still opens its gate (a step that never
  opens deadlocks the queue), and its other sequences read and continue as usual. A failed
  sequence is poisoned: its outcome of that step and of every later one is a typed failure
  (`Outcome::TableReadFailed`, `TableReadError`), whose outputs are discarded and whose
  accepted state returns as an orphan's, and the request fails (`WeightReadFailed`) until it
  closes. A lookahead step is claimed as usual; its failed slots fail when it finishes.
- **Reads that bypass the page cache** fall back to reads through the cache, dropping the pages
  each read filled on Linux, for a file whose filesystem refuses direct I/O when it is opened or
  at its first direct read; the fallback is kept for the file.
- **Readers are the same for both.** Every reader of a gathered table addresses its rows through
  a directory (`gathered_row`): an empty one, under which it reads each row it is given, or the
  table's directory of every row's slot. A step whose rows the host knows is given their slots,
  as a whole table's step is given the rows, under the empty directory. A step whose rows the
  device selected reads a streamed table through its directory, in the reader's own address
  calculation: no entry runs before the reader and no reader knows a table is streamed.
- **A device-selected row an imported table does not hold** is a miss: the target's lookahead is orphaned
  and resubmitted with the row brought ([state transactions](state-transactions.md)); a
  head's chained proposal embeds another row, and target verification rejects what follows.
- **Replacement** takes the slot of a row no step named since the replacement last passed it,
  never one the current step names. A directory change that follows a queued read of selected rows
  waits for the fill it belongs to.
- **Rows at load.** Before readiness the slots are filled, most valuable first: the rows steps
  named when the table was last served, then the table's first rows. Load waits for their
  conversion. The rows steps name are kept in a file beside the kernel cache after the table's
  first 256 steps and when the engine releases it. A stored table is seeded with the kept rows
  only: n-gram rows have no order of value without them.
- **Mapped windows** are bounded by the component's largest stored tensor whether or not that
  tensor is imported, so a streamed table does not change the windows of the weights around it.

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
