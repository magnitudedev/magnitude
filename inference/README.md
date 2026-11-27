# Inference

This workspace contains Magnitude's Seismic-backed inference engine and service.
The engine owns model interpretation, generation, scheduling, and state. Seismic
owns numerical compilation, device resources, and execution on CPU, Metal,
CUDA, and Vulkan. The service owns the public API, model inventory, assessment,
residency, and worker supervision.

Start with the [architecture overview](docs/overview.md), then the
[engine](docs/engine/overview.md), [Seismic](docs/seismic/overview.md), and
[distribution](docs/distribution.md) contracts.

## Layout

| Directory | Owner |
| --- | --- |
| `engine/` | Reusable engine, model families, scheduler, serving, and headless CLI |
| `service/` | HTTP service, contracts, catalog, assessment, and worker lifecycle |
| `seismic/` | Language, compiler, runtime, native libraries, device backends, and hardware research |
| `catalog/` | Model catalog and planner inputs |
| `conversion/` | Conversion of released checkpoints into the GGUFs Magnitude publishes, and their byte-exact check |
| `evaluation/` | `mrig`, every check and measurement on the hosts, and what its instruments drive: fixtures, the harness, the references and the serving benchmark (`bench`) |

## Build and test

From the repository root, build a development installation and smoke its binary,
authenticated hardware endpoint, readiness handshake, and parent-loss exit:

```sh
bun icn:build
bun run inference/scripts/smoke.ts inference/target/development/installation.json
```

From this directory, check the Rust workspace or run a focused crate test:

```sh
cargo check --workspace --all-targets
cargo test -p magnitude-scheduler
```

The product's local release acquisition test is `bun test:release-bootstrap`
from the repository root. `--cached` reuses a previously built local release and
does not validate source changes made after that build.

The engine CLI is `magnitude-engine` (`engine/cli`); the production service
binary is `magnitude-inference` (`service/server`). Platform support and
qualification floors are in [compatibility](docs/compatibility.md). A passing
local build does not establish qualification on every release host.

### Diagnosing GPU expert-cache headroom

`magnitude-engine --device-room BYTES` limits device weight placement to the
smaller of the supplied budget and the room available to the plan. It does not
cap total device memory, which also holds request state and execution buffers.
`--weight-cap BYTES` also limits host weights. Omit both flags for the stock
comparison.

To compare a change in device weight room, use the same model, input and
generation settings. Start a fresh engine for each measured run and alternate
stock and override order. Record wall time, prompt and generated token counts,
resident expert slots, and the residency watcher's per-request DMA bytes and
wait time. The request counters combine prefill and decode; summed DMA wait is
not time on the request's critical path, and a bank reclamation is not a count
of evicted or reloaded experts. Report a timing change separately unless the
cache observations support an explanation. A useful budget on one device or
workload is not a portable default.
