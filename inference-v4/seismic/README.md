# Seismic

All packages live under this subsystem directory and belong to the workspace
at `inference-v4/Cargo.toml`. Grouping directories are not nested workspaces.

| Directory | Owns |
|---|---|
| `lang/` | Checked language, semantic programs and expression arena |
| `ir/` | Executable kernel/schedule/storage definitions and checked construction |
| `compiler/` | Refinement, numerical analysis, planning, freezing and portfolio |
| `estimator/core/` | Generic prediction contract, traversal and shared service model |
| `estimator/metal/` | Pure Metal demand rules over shared IR |
| `backends/{metal,cpu,cuda}/` | Target adapters, native realization and execution |
| `runtime/` | Invocation binding, workflow admission, memory and submission |
| `api/`, `build/`, `cli/` | Public API, build integration and command line |

`ir`, estimator core and the Metal model have no compiler or native runtime
library dependency. Shared Metal vocabulary lives in `ir::metal`; there is no
second executable representation. Compiler refinement and solver entry points
and runtime admission are modules in their owning crates.

From `inference-v4`, the short host-only loops are:

```sh
cargo test -p seismic-ir -p seismic-estimator -p seismic-estimator-metal --lib --locked --offline
cargo test -p seismic-compiler --lib solve::tests --locked --offline
cargo test -p seismic-metal --lib isolation --locked --offline
cargo test -p seismic-runtime --lib workflow::tests --locked --offline
```

The Metal isolation tests construct synthetic target facts, refine checked
source, and pass one shared kernel through prediction and source emission.
They do not open a device or establish physical estimator accuracy. Native
compilation still precedes production planning. Workflow binding closes the
whole graph before allocation; admission then reserves the graph atomically,
rolls the full transaction back on failure, and retains every lease and access
permit through native completion.

See the [durable compilation architecture](../../design/inference/seismic-compilation.md)
for the subsystem contracts, ownership, and qualification boundaries.
