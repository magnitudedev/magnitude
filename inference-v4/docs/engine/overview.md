# Inference engine

**The engine owns inference meaning and policy over Seismic numerical execution.**
The same model boundary supports embedded execution and scheduled generation.

## Owners

| Owner | Owns | Delegates |
| --- | --- | --- |
| Model loading | Artifacts, metadata, architecture descriptions, weight roles | Representation operations and physical imports to Seismic |
| Model executor | Model programs, packed inputs, sequence advances, requested readout | Numerical work and native submissions to Seismic |
| Logical state | Positions, visibility, sharing, tentative advances, accepted history | Physical storage and completion lifetime to Seismic |
| Generation | Accepted tokens, grammar progress, output queues, recovery | Packed computation to the model executor |
| Service | Admission, phase choice, capacity negotiation, fairness | Legal work proposals and resource prices to generation/model owners |
| Execution owner | Serialized access to live model/device state and completion reconciliation | Immutable control requests and results to callers |
| Chat / transport | Preparation, parsing, framing, and client lifetime | Token-level progress to the engine |

## Request lifecycle

```text
artifact + model library → loaded executor
input preparation → model input → request admission
    → legal work proposal
    → capacity preparation + packed submission
    → physical completion
    → per-request validation and state acceptance
    → queued output → publication
```

| Boundary | Invariant |
| --- | --- |
| Loading | Does not implicitly download, serve, or create service workers |
| Preparation | Proposals are pure; capacity and tentative state are acquired during batch preparation |
| Submission | One packed execution may serve multiple independent requests |
| Acceptance | A completed row may be rejected without discarding accepted peer progress |
| Publication | Accepted output is retained until consumed or explicitly discarded by its owner |
| Cancellation | Stops future work; submitted resource lifetime still follows completion |
| Recovery | Reconstructs numerical state from retained logical history and input semantics |

## Public composition

- Library callers can request state-only advancement, raw logits, selected vocabulary
  readout, or generation without HTTP.
- Serving and library loading share artifact interpretation and model execution.
- The service observes generic work, completion, and capacity prices; it does not
  inspect tensor layouts, backend objects, or modality-specific computation.
- Live execution objects stay with their owner. Host boundaries carry immutable
  inputs, options, constraint plans, and bounded results.
- Fatal execution-owner failure affects all dependent requests; row-local failure
  remains local when the execution contract permits it.

## Component contracts

| Document | Scope |
| --- | --- |
| [Models](models.md) | Artifacts, weights, numerical composition, packed execution |
| [Inputs](inputs.md) | Text, conditioned spans, media preparation, feature lifetime |
| [State](state.md) | History, checkpoints, forks, commitment, reclamation |
| [Scheduling](scheduling.md) | Service policy, capacity, execution ownership |
| [Generation](generation.md) | Proposals, acceptance, constraints, output, replay |
| [Chat](chat.md) | Templates, tokenizer, tools/thinking, parsing, serving |
| [Seismic runtime](../seismic/runtime.md) | Physical allocation, binding, submission, completion |
