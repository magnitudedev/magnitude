# Magnitude ICN

This workspace builds the Inference Control Node. `icn-api` is the backend-neutral Axum/Utoipa
boundary and can export OpenAPI without compiling llama.cpp. `icn-core` owns the backend contract,
`icn-llamacpp` adapts the pinned Rust bindings, and `icn-server` assembles the executable.

The native dependency has two independently recorded revisions in `native-pin.toml`: the exact
`llama-cpp-rs` commit used by Cargo and the llama.cpp gitlink embedded by that commit. The Cargo
dependency uses `rev`, never a moving branch or semver range. Run `bun icn:verify-native-pin` after
changing either pin. A future fork only requires changing the repository and revision together;
the ICN-facing backend interface remains unchanged.

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
bun icn:test                  # Rust API, SSE, backend, and workspace tests
bun icn:generate
bun icn:check-generated
bun icn:verify-native-pin
bun icn:doctor
bun icn:version
```

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
