# Magnitude ICN

This workspace builds the Inference Control Node. `icn-api` is the backend-neutral Axum/Utoipa
boundary and can export OpenAPI without compiling llama.cpp. `icn-core` owns the backend contract,
`icn-llamacpp` adapts the pinned Rust bindings, and `icn-server` assembles the executable.

The native dependency has two independently recorded revisions in `native-pin.toml`: the exact
`llama-cpp-rs` commit and the llama.cpp gitlink embedded by that commit. The editable binding source
is checked out at `native/llama-cpp-rs`; the inference workspace must consume its `llama-cpp-2`
crate by relative path rather than resolving a second Cargo Git checkout. Run
`bun icn:verify-native-pin` after changing either pin; the ICN-facing backend interface remains
unchanged.

## Native submodule management

The native source is nested and pinned:

```text
magnitude
└── inference/native/llama-cpp-rs       # our bindings fork
    └── llama-cpp-sys-2/llama.cpp       # exact upstream llama.cpp revision
```

We do **not** need utilityai or llama.cpp to accept our changes. Binding changes are committed and
pushed to `magnitudedev/llama-cpp-rs`. Upstream PRs are optional.

Magnitude stores only the exact bindings-fork commit, not changes made inside the submodule. The
required order for a bindings change is therefore:

1. Change and test `inference/native/llama-cpp-rs`.
2. Commit and push that change to `magnitudedev/llama-cpp-rs`.
3. Commit the updated `inference/native/llama-cpp-rs` pointer in Magnitude.

Never point Magnitude at an unpushed bindings commit; other checkouts and CI could not fetch it.

We normally do not modify llama.cpp. To upgrade it, update its nested commit pointer and commit that
pointer in our bindings fork. Create a llama.cpp fork only if we actually need native patches.

The bindings fork directly compiles its checked-in C/C++ wrapper sources, including the
`wrapper_common_fit` surface, alongside the pinned llama.cpp checkout; it does not generate or
apply a source overlay. [`parity/upstream/binding-surfaces.json`](parity/upstream/binding-surfaces.json)
is the parity-owned audit inventory that maps relevant upstream, bridge, and safe Rust surfaces. It
is not a fork build input; review it whenever either native pin or a parity-relevant safe surface
changes.

Initialize both submodules after cloning Magnitude:

```sh
git submodule update --init --recursive
```

## First five minutes

Run these commands from the Magnitude repository root.

Compile the development binary:

```sh
bun icn:build
```

The executable is now at `inference/target/debug/magnitude-icn`. Start it with the deterministic fake
backend, which does not need a model file:

```sh
bun icn:dev
```

In another terminal, check health and make a streaming completion:

```sh
curl -sS http://127.0.0.1:8080/health

curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  --data '{
    "model": "icn-fake",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

The second command prints OpenAI-compatible `data:` frames followed by `data: [DONE]`. Stop the
server with Ctrl-C.

To use a real GGUF model:

```sh
bun icn:serve -- \
  --model /absolute/path/to/model.gguf \
  --model-alias my-model \
  --bind 127.0.0.1:8080
```

Use `my-model` in the same completion request. On Apple Silicon, the pinned bindings enable their
macOS Metal backend. `--gpu-layers 0` forces CPU execution; the default attempts to offload all
layers.

## Build and verification commands

Useful commands from the monorepo root:

```sh
bun icn:check                 # type-check the Rust workspace without linking a final binary
bun icn:build                 # debug binary, fastest normal development build
bun icn:build:release         # optimized binary at inference/target/release/magnitude-icn
bun icn:build:reference       # selected pinned tests, official tools, and native oracle
bun icn:test                  # Rust API, SSE, backend, and workspace tests
bun icn:parity:validate       # validate cases, fixtures, profiles, targets, and model registry
bun icn:parity:list           # list primitive cases and implementation status
bun icn:parity:test:ts        # test reference/model/provenance scripts
bun icn:build:candidate -- --reference-manifest <path> # build the production ICN parity probe with provenance
bun icn:generate
bun icn:check-generated
bun icn:verify-native-pin
bun icn:doctor
bun icn:version
```

`bun icn:build:reference -- --backend metal --target focused-tests --target oracle` builds only
declared targets from the exact nested llama.cpp source used by the Rust bindings. Other target IDs
include `llama-bench`, `llama-batched-bench`, `llama-perplexity`, `backend-ops`, and
`quantize-perf`. The builder records source, configuration, artifact, and oracle digests; use
`--dry-run` to inspect the resolved build without compiling. Every invocation reserves a fresh
CMake tree, uses an allowlisted build environment, and records compile/link evidence for assertion
and sanitizer status; an earlier CMake cache is never reused as parity evidence.

## Primitive parity

`parity/` contains neutral cases, fixtures, profiles, the content-addressed model registry, upstream
target manifests, JSON evidence schemas, and the thin native C++ oracle. `icn-parity` validates and
runs these assets without depending on the Rust bindings fork. It supports unchanged upstream
tests, official upstream tools, and differential native-oracle/ICN-probe cases. Comparisons happen
outside both producer processes and are exact, structural, tolerance-based, capability-based, or
same-work performance ratios.

Parity execution never uses a generated chat response or HTTP exchange as primitive evidence.
The production `icn-probe` exposes the active paired operations through production-owned
`icn-llamacpp` code; descriptor status remains authoritative, with genuine artifact or production
API gaps kept `planned` or `disabled`. The `diagnostic` profile is an uncontrolled, non-gating
two-sided functional smoke. `native-diagnostic` separately runs the one-sided native C0/P0 checks
without making a candidate-parity or controlled-performance claim. Generated run directories live
under `results/parity/`.
Downloaded parity models live under `target/parity-models/`. Both locations, all native/Rust build
trees, and candidate artifacts are generated and ignored by the repository.

`bun icn:generate` runs the Rust OpenAPI exporter and regenerates `packages/icn/openapi.json` plus
the Effect Schema, HttpApi, and streaming descriptors under `packages/icn/src/generated`.
`bun icn:check-generated` performs the same derivation without writing and fails if committed output
is stale.

The `inference/` directory is also a Bun workspace, so the equivalent short forms work:

```sh
bun run --cwd inference build
bun run --cwd inference test
bun run --cwd inference dev
```
