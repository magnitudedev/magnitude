# Release model catalog

`models.json` is the reviewed source for Magnitude's curated local-model catalog. The catalog
generator resolves it through ICN's production repository resolver, GGUF parser, package builder,
template assessor, and native planner. It emits one reviewed release manifest:

- `generated/release-catalog.lock.json` contains the resolved catalog and hardware-independent
  model-planning metadata, including immutable Hugging Face revisions and the exact byte length and
  digest of every GGUF header required by the native planner.

After editing the source or when intentionally updating pinned upstream revisions, run from the
repository root:

```sh
bun run local-model-catalog:refresh
bun run local-model-catalog:check
```

Refresh synchronously conditionally revalidates every upstream repository, fails rather than
publishing a partial catalog, and writes the manifest atomically. Review the source and generated
diffs together. Stable upstream and source inputs produce byte-identical output.

Development and release Cargo builds materialize the planner bundle in Cargo's build-output
directory. When a matching bundle is not already present there, the build downloads only the
pinned header byte ranges directly from Hugging Face, verifies each digest, deterministically packs
them, and verifies the aggregate digest from the manifest. The resulting bundle is embedded in the
ICN binary. `HF_TOKEN` is used when present. The bundle is not committed to the repository.

Normal product startup never runs this workflow and never reconstructs this catalog from the
network. Setup combines the already embedded planning inputs with current hardware topology and
local calibration; it does not fetch model metadata. Installed models are inspected from their
local files. Adding or changing a curated model requires regenerating the manifest and shipping a
new release.
