# Release model catalog

`models.json` owns the catalog. `models.lock.json` contains only a map from each catalog ID to one
immutable Hugging Face commit.

```sh
bun run icn:catalog:update    # advance the commit map
bun run icn:catalog:build-bundle # build planner inputs from the pinned commits
```

Generation resolves the pinned repositories, compacts their GGUF headers, verifies native-planner
parity, and writes `model-planner-inputs.bundle`. The bundle is derived release output and is not
committed. It is the only catalog-related file shipped alongside ICN; catalog definitions and pins
are compiled into the executable.
