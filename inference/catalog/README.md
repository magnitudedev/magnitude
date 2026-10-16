# Release model catalog

`models.json` owns the catalog. `models.lock.json` maps each catalog ID to the immutable Hugging
Face commit for its target package and, when separately packaged speculative decoding is declared,
the immutable commit for its draft package.

Input modality is inspected from the exact target package. Image-capable targets receive a
projector during generation: a sole repository `mmproj` is selected automatically, while
repositories with multiple candidates require an exact `projector.path` declaration. The
projector is locked as a related component of the target package.

Each variant's primary GGUF is the one file (or first shard) whose path contains its `format`.
When other repository files also contain that format, such as MTP heads, the variant declares its
exact `path` instead.

A catalog entry declares no context length. Its serving profile is the target's supported maximum
context from its GGUF metadata, which is the context the engine resolves, serves and assesses.

```sh
cargo run --release -p magnitude-service-catalog-tool -- update-lock ...   # advance the commit map
cargo run --release -p magnitude-service-catalog-tool -- build-bundle ...  # build the header bundle
cargo run --release -p magnitude-service-catalog-tool -- acquire --model qwen3.5-4b:gguf:q4 --model-store DIR
cargo run --release -p magnitude-service-catalog-tool -- dump-headers \
  --bundle target/catalog-inputs/model-planner-inputs.bundle --output catalog/headers
```

`acquire` resolves one catalog model at its locked revisions, downloads into a managed model store
only the target, drafter and projector files that are missing, verifies them through the service's
own download path, and prints their absolute paths and SHA-256 digests as JSON (the evaluation
model-reference contract). Evaluation tools obtain models only through it.

Generation resolves the pinned repositories and writes `model-planner-inputs.bundle`: the exact GGUF
header (every byte through the aligned tensor-data offset) of every package component, which is
what the engine's header-only planning and chat-template inspection read before a model is
downloaded. Repeated references to the same immutable package share one package identity and one
set of bundled headers. The bundle is derived release output and is not committed. It is the only
catalog-related file shipped alongside the service; catalog definitions and pins are compiled into
the executable.

`dump-headers` writes the bundle's headers as the JSON the engine family tests read
(`catalog/headers/`, not committed), replacing that directory. `index.json` lists every component
as `{model, role, file}`. A target is `<model>__target-<variant>.json` (the variant ID with `:`
as `_`), a separate draft package `<model>__draft.json` and a projector `<model>__projector.json`;
a draft or projector must be identical across the model's variants. Each file holds the
component's metadata and its tensor directory (`name`, GGML `type`, `shape` in GGML storage
order). Metadata arrays longer than 1024 entries, such as tokenizer vocabularies and merges, keep
only `{"head": [first 64], "length": total}`; shorter arrays, including every per-layer array,
are complete. A split target's file is its first shard, and `<file>-allshards.json` beside it
holds the first shard's metadata with every shard's tensors. Build the bundle first when it is
absent (`bun run catalog:build-bundle`).
