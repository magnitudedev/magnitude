# Release model catalog

`models.json` owns the catalog. `models.lock.json` maps each catalog ID to the immutable Hugging
Face commit for its target package and, when separately packaged speculative decoding is declared,
the immutable commit for its draft package.

```sh
bun run icn:catalog:update    # advance the commit map
bun run icn:catalog:build-bundle # build planner inputs from the pinned commits
```

Generation resolves the pinned repositories, compacts their GGUF headers, verifies native-planner
parity, and writes `model-planner-inputs.bundle`. Repeated references to the same immutable package
share one package identity and one set of bundled planner payloads. The bundle is derived release
output and is not committed. It is the only catalog-related file shipped alongside ICN; catalog
definitions and pins are compiled into the executable.
