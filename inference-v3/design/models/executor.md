# Model executor

**A model executor owns architecture composition and logical sequence advances;
all numerical work is a Ops function, and all physical execution is a
Ops compiled callable.**

## Boundary

The executor knows model geometry, weight roles, input conditioning, logical
positions and state transactions. It does not know tensor graph nodes, lowering
candidates, kernels, temporary layouts, TileLang or backend capabilities.

```text
description + bound weights + model tensor function
                         │ compile for workload geometry
                         ▼
                 Ops callable

requests + tentative state views
                         │ packed dynamic inputs
                         ▼
                 outputs + completion
                         │
             accept and commit per request
```

Model construction composes ops formulas as source-level architecture equations.
Ops traces that function for each required specialization and owns physical
implementation beneath it. Typed model/formula navigation is derived from those
actual occurrences, not a second benchmark-only architecture tree.

## Lifecycle

```text
input ──open──► sequence @ position 0
                   │
prepare(requests) ─└──► packed tensors + tentative resource views
                                   │ Ops submit
                                   ▼
                         outputs + completion
                                   │
       advance.commit() ◄── completed and accepted
       advance.abort()  ◄── failed, cancelled or rejected
```

| Rule | Reason |
|---|---|
| Physical execution is packed; acceptance remains per request | One tensor invocation serves peers without merging their logical histories |
| A tentative state view exists before submission | Ops sees explicit resources without learning commit policy |
| Position advances only after completion and acceptance | Logical history never promises state that is unavailable or rejected |
| Requested outputs are explicit | State-only prefill does not compute or materialize unused readout |
| Checkpoints contain reconciled logical state | A checkpoint never captures an unresolved resource version |
| Model equations compose formulas, not physical implementations | Kernel and fusion changes preserve the model's numerical definition |

## Inputs and multimodality

An input layout is a token count plus ordered conditioned spans. Each span states
its identity, boundary behavior, logical history contribution and prepared
feature tensor. Chunking observes those boundaries without learning why a span
exists.

```text
host preparation: media ──► features + aligned spans
model function:    features + tokens + coordinates ──► ordinary ops formulas
```

The executor never interprets raw media inside a language-model tensor function.
A modality adds preprocessing, a Ops encoder/projector function and
conditioned spans. It does not add service branches, a second execution owner or
modality behavior to the tensor compiler. The complete contract is defined in
[inputs.md](inputs.md).

## Reclamation

The executor prices what closing reconciled sequences would release: logical
state claims, idle model resources and compiled-callable caches exclusively owned
by that set. Ops reports physical resource charges and reclaims only
after completion. The service chooses victims; neither the executor nor
Ops performs eviction policy independently.
