# Release model catalog

`models.json` owns the catalog. `models.lock.json` maps each catalog ID to the immutable Hugging
Face commit for its target package and, when separately packaged speculative decoding is declared,
the immutable commit for its draft package.

Input modality is inspected from the target GGUF. Image-capable targets receive a projector during
generation: a sole repository `mmproj` is selected automatically, while repositories with multiple
candidates require an exact `projector.path` declaration. The projector is locked as a component of
the target package rather than as a separate capability flag.

```sh
bun run icn:catalog:update    # advance the commit map
bun run icn:catalog:build-bundle # build planner inputs from the pinned commits
```

Generation resolves the pinned repositories, compacts their GGUF headers, verifies native-planner
parity, and writes `model-planner-inputs.bundle`. Repeated references to the same immutable package
share one package identity and one set of bundled planner payloads. The bundle is derived release
output and is not committed. It is the only catalog-related file shipped alongside ICN; catalog
definitions and pins are compiled into the executable.
