# Kernel shape portability audit

The dense Qwen 3.5 failure was a schedule-domain error: persistent decode assumed eight query heads per KV group, while the 4B model has four. The main contraction's physical matrix rows had become a restriction on logical heads. More tiling options alone would not repair the missing state, reduction and publication coverage found elsewhere.

## Scope and repairs

Reviewed all 27 authored kernel modules and the related dispatch and independent references. The initial probe had 55 planning cases (29 accepted, 26 rejected) and 15 native cases (10 correct, four numerical mismatches, one construction failure). The resulting repairs are driven by shared operation geometry, without model-name branches or compiler changes.

| Area | Broken assumption | Repair |
|---|---|---|
| Attention | Logical head cohorts equal physical matrix rows; one channel per merge thread | Cohorts remain inside a KV group, physical rows carry masks through corrections and publication, merges cover all channel strips |
| Recurrent preparation/output | Width fits one thread strip; every sequence has a token owner | Complete channel traversal and reductions; explicit empty-sequence state ownership |
| Softmax | Each partition has a finite maximum | Empty local partitions contribute zero mass to the merge |
| Projections/embedding/RMS | Public rank equals the emitter's 2D view; encoded alignment equals the preferred vector tile | Preserve leading ranks; qualify matrix and vector strategies independently |
| MoE | Sparse prefill is invalid; at most 256 experts; all banks are packed | Select a viable strategy by occupancy and representation; cover dense banks and complete route initialization |
| Routing | One thread must own each expert | Tiled selection with explicit reduction-result exchange |
| Packed KV | Lane coordinates partition complete words; vector reads cover a whole tile | Separate coordinate and word ownership; ceil coverage with valid read/write masks |
| Schedule selection | Preferred geometry must fit before alternatives are considered | Prune the candidate family against resources and geometry-dependent workspace before selection |
| Compact attention | A quotient produces a legal matrix tile; computed shared-gather indices survive lowering | Choose bounded power-of-two tiles; materialize owner-local codebook indices before shared gathers |

The rotated-attention undefined `channel` compilation error also reproduces with the original sources. Inspection showed a loop coordinate escaping into a host launch argument during lowering of a computed shared lookup. A thread-local index expresses the gather through existing public IR and passes both history and empty-history cases. No TileLang or TVM edits were needed.

Ten old attention schedule assertions omitted the existing 16-byte interval metadata. Their expected totals were corrected. A four-head decode assertion now expects the newly applicable matrix realization; its numerical obligation is unchanged.

## Qualification

Boundary tests check complete outputs and state against independent references, including physical widths 16/96/192/256/512, head groups 1/2/3/4/8/12/16, more channels or experts than threads, ranks zero through three where permitted, sparse routes, empty sequences and empty local softmax partitions. Fused attention and recurrent output are exercised as composed operations. Genuine mathematical/format constraints remain explicit.

Production Qwen 3.5 4B Q4_K_M executes prefill and two decode advances with finite logits. The previously working 35B A3B Q4_K_M supplies the performance baseline. `performance-protocol.md` defines comparable inputs, warmups, boundaries and the noise rule; JSON files retain raw samples. Complete saved logits are compared, not only selected tokens. Standalone submit/wait timings include host overhead and do not establish kernel speedups.

The consolidated gate passed **189 tests with 21 skipped** in 225.58 seconds. The final resource-selection suite passed **7 tests** (six overlap the consolidated gate). All **55 planning probes** are accepted. See `validation.json` for the command.

| Production stage | Original median ms | Final median ms | Complete logits |
|---|---:|---:|---|
| Prefill, 32 tokens | 57.175 | 55.208 | Identical |
| Decode 1 | 10.393 | 9.750 | Identical |
| Decode 2 | 10.250 | 9.655 | Identical |

There is no measured slowdown under the declared gate. The earlier candidate comparison was within 1% of the baseline; the final sample set is lower. These runs establish no regression in the measured workload, not a repeatable speedup claim. The final six native-library hashes match the baseline. `final-provenance.json` records the final source and saved-logit hashes; the unchanged compiler/submodule identity is explicit.

## Limits

Native and production qualification was on Apple M4 Max / Metal. CUDA cases are present where supported by the suites but require a CUDA host. The audit improves coverage of the shared supported contracts; it does not prove every positive dimension, every representation combination, every model family or every device. Planning acceptance is distinct from native correctness. Previously rejected or incomplete execution has no valid performance baseline.

The durable authoring and review rules are in `inference-v3/design/development/ir-portability.md`; this report holds the concrete findings and run evidence.
