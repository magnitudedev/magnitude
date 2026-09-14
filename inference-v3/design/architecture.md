# Architecture

The normative formula, operation, execution and measurement boundaries are in
[Formula-defined execution and measurement](../../design/inference/formula-execution.md).

**The engine owns inference meaning; Ops owns all tensor computation and
physical execution; TileLang owns portable kernel compilation and target
realization. No concern crosses those boundaries disguised as configuration.**

## Components

| Component | Contains | Exposes | Must never expose |
|---|---|---|---|
| Serving | Transport, rendering, media preparation, parsing and framing | Tokens, prepared media tensors and request options | Live parser, tokenizer, processor or device objects across the worker |
| Service | Admission, phase choice, capacity negotiation and publication | Work and progress | Model math, tensor graphs or transport |
| Generation | Request history, output credit, sampling and recovery | Work proposals, acceptance and snapshots | Numerical state layout or physical batching |
| Model executor | Architecture equations, logical state transitions and packed model inputs | Prepared advances and model outputs | Kernels, compiler choices, scratch or backend facts |
| Weight source | Artifact parsing, model-role mapping and import policy | Typed stored tensors and constraints | Kernel or target choices |
| State | Logical positions, visibility, sharing, commit and reclamation policy | Stable physical resource views | Tensor schedules or backend storage objects |
| Ops | Formulas, complete operations including I/O, compilation, representations, resources, submission and measurement | Typed formula handles, resources, compiled callables, completion and observations | TileLang or TVM objects in its public API, model and request policy |
| TileLang | Portable kernel language, target capabilities, compiler, lowering and runtime adapters | Compiled execution of a portable program | Model, operation-graph or inference-state meaning |

## Composition

```text
Serving
└── Service
    ├── Generation × requests
    │   └── Sampling
    └── Model executor
        ├── Description ◄── Weight source and model-role mapping
        ├── Logical state ──── physical resource views
        └── Model tensor function
             └── Ops
                  ├── inference operations and tensor compiler
                  ├── physical resources and compiled execution
                  └── portable TileLang programs
                       └── TileLang compiler and runtime ── device
```

The model function is the only numerical expression owned by Magnitude. It uses
Ops as a tensor library and carries no schedule, allocation or compiler
objects. Ops sees no request, checkpoint, container or modality policy.
TileLang sees only final portable compilation units and their operands.

## Boundary rules

| Rule | Consequence |
|---|---|
| Every model computation uses Ops | No model, operation, loader or driver in the engine imports TileLang |
| Hardware capability flows from TileLang to Ops | Magnitude is hardware-unaware; Ops does not probe or classify vendors |
| Distinct mathematics has a formula contract | Attention, recurrence, routing and experts retain meaning across physical implementation changes |
| Model topology remains model code | No Qwen-, Gemma- or modality-named concept enters the tensor compiler merely to aid matching |
| A parent operation may implement its composed formula directly | Fusion is authored against a mathematical boundary, not selected by competing graph covers |
| A formula is not a compilation boundary | Authored operations remain composable into maximal native submission units |
| A portable operation's native realization belongs to TileLang | Missing expressiveness or performance is fixed in its capability, language, lowering or runtime contract |
| One TileLang program is one native host entrypoint | TileLang encodes its device-kernel launches natively and supports generic partial binding; Magnitude never loops over them |
| Dynamic TileLang construction stays Python-native | Ops supplies an ordered ABI and composes authored schedules through TileLang's public eager construction; it never generates source text or manipulates TIR |
| Physical completion and logical acceptance are different events | Ops releases resources only after completion; Magnitude commits state only after acceptance |

The dependency boundary is enforced, not conventional:

```text
engine ──▶ ops ──▶ tilelang

forbidden: engine ──▶ tilelang
forbidden: ops ──▶ engine
forbidden: public ops API ──▶ TileLang/TVM values
```

## Composability

A boundary is not an execution barrier. A model remains formula composition while
its operations may implement a primitive or a whole composed formula. Layout,
representation and materialization follow the authored physical implementation.

Generic means reusable by semantic contract, not broad or weak. A region may be
specific enough that one architecture currently produces it. It remains a valid
tensor optimization when its applicability is stated in operations, geometry,
precision, representations and resource effects rather than a model or backend
name.

## Performance containment

| Avoidable cost | Owner of its elimination |
|---|---|
| Primitive graphs that obscure an inference algorithm | Ops operation library |
| Intermediate tensors and excess kernel boundaries | Ops operation implementation and materialization |
| Python launches or binding proportional to model depth | Ops submission planning and TileLang native multi-launch/partial binding |
| Poor tile, traversal or reduction strategy | Authored portable kernel and TileLang schedule tuning |
| Missing native instruction or target pipeline quality | TileLang lowering |
| Rebinding immutable operands or replanning per invocation | Ops compiled callable |
| Copying shared history when requests change | Magnitude logical state over Ops resource views |
| Model or request work on the device hot path | Model executor and service boundaries |

An abstraction is not accepted merely because it is clean. Formula boundaries,
materialization, dispatches, generated code and enclosing TTFT, prefill and decode
are observable and qualified. A persistent gap identifies an incomplete owner,
not permission to cross a boundary.

## Documents

| Component | Document |
|---|---|
| Ops | [tensor-system.md](tensor-system.md) |
| Inference operations | [operations.md](operations.md) |
| Portable schedules | [kernels.md](kernels.md) |
| Kernel optimization | [development/kernel-optimization.md](development/kernel-optimization.md) |
| Physical execution | [platform.md](platform.md) |
| Weight sources and representations | [weights.md](weights.md) |
| Logical KV state | [state.md](state.md) |
| Model inputs and multimodality | [models/inputs.md](models/inputs.md) |
| Model executor | [models/executor.md](models/executor.md), [models/qwen35.md](models/qwen35.md) |
| Generation | [engine/generation.md](engine/generation.md) |
| Service | [engine/service.md](engine/service.md) |
| Serving | [serving.md](serving.md) |
| Measurement | [performance.md](performance.md), [benchmark-fixtures.md](benchmark-fixtures.md) |

## Outside this design

Training, automatic differentiation and distributed tensor execution are not
part of Ops. They are not anticipated through abstractions in the
inference path.
