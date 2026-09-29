# Inference

This workspace contains Magnitude's Seismic-backed inference engine and service.
The engine owns model interpretation, generation, scheduling, and state. Seismic
owns numerical compilation, device resources, and execution on CPU, Metal,
CUDA, and Vulkan. The service owns the public API, model inventory, assessment,
residency, and worker supervision.

Start with the [architecture overview](docs/overview.md), then the
[engine](docs/engine/overview.md), [Seismic](docs/seismic/overview.md), and
[distribution](docs/distribution.md) contracts. The frozen previous implementation
is under `../old-inference/`; its behavior and independent V3 numerical fixtures
remain preservation references.

## Layout

| Directory | Owner |
| --- | --- |
| `engine/` | Reusable engine, model families, scheduler, serving, and headless CLI |
| `service/` | HTTP service, contracts, catalog, assessment, and worker lifecycle |
| `seismic/` | Language, compiler, runtime, native libraries, and device backends |
| `catalog/` | Model catalog and planner inputs |
| `benchmarks/` | Session Bench and runtime adapters |
| `validation/` | Independent numerical references and hardware qualification |

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
