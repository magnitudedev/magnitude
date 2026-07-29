# Release model catalog

`models.json` is the reviewed source for Magnitude's curated local-model catalog. The catalog
generator resolves it through ICN's production repository resolver, GGUF parser, package builder,
template assessor, and native planner. It emits one reviewed release manifest:

- `generated/release-catalog.lock.json` contains the resolved catalog and hardware-independent
  model-planning metadata, including immutable Hugging Face revisions, exact source GGUF-header
  ranges, and the digest and size of every compact planner stub.

After editing the source or when intentionally updating pinned upstream revisions, run from the
repository root:

```sh
bun run icn:catalog:update
bun run icn:catalog:hydrate
bun run icn:catalog:check
```

Update synchronously conditionally revalidates every upstream repository, generates standard-GGUF
planner stubs without lexical tokenizer payloads, and compares the native planning result of each
source header and stub at every curated profile. Each primary stub contains a compressible
placeholder vocabulary of the original cardinality so llama.cpp follows its ordinary vocabulary
loading path and preserves graph-relevant metadata. Any resolution or parity failure prevents
publication. Review the source and generated diffs together. Stable upstream and source inputs
produce byte-identical output.

Hydrate downloads only the pinned source-header byte ranges from Hugging Face, verifies each
digest, reproduces and verifies each compact stub, deterministically packs the stubs, and verifies
the aggregate digest from the lock file. `check` repeats that validation offline. The bundle is not
committed to the repository.

Normal product startup never runs this workflow and never reconstructs this catalog from the
network. Release builds package the lock file and planner bundle beside ICN under `catalog/`; local
development prepares the same installation layout before starting ACN. ICN validates both files at
startup, then combines those release inputs with current hardware topology and local calibration.
Installed models are inspected from their local files. Adding or changing a curated model requires
regenerating the lock file and shipping a new release.
