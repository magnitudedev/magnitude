# Pinned upstream reference

This directory is the declarative source of truth for the native llama.cpp
reference build used by primitive parity validation.

- `targets.toml` names focused CTest suites, official tools, and the native
  oracle. Expensive backend and quantization performance executables are
  separate selections and are not part of the default build.
- `build-profiles.toml` defines the upstream-default and Cargo-equivalent
  native build lanes.
- `binding-surfaces.json` inventories the upstream, C bridge, and safe Rust
  symbols used by Magnitude. Its verifier checks declarations only; it does
  not establish behavioral parity.

Build one or more selections with:

```sh
bun run scripts/build-reference.ts --target llama-bench --target oracle
```

With no `--target`, the builder uses `default_targets`. `--target all` builds
every declared selection. CTests are built but never executed implicitly. Run
the C0 focused suite through either the builder or its qualification wrapper:

```sh
bun run scripts/build-reference.ts --target focused-tests --run-tests \
  --model-dir /path/to/registry-artifacts
bun run scripts/test-upstream.ts --model-dir /path/to/registry-artifacts
```

Both paths parse the accepted artifact from `models/registry.toml`, verify
its exact byte count and SHA-256, and stage a private create-only copy at the path expected
by upstream CTest. `--model-dir` takes precedence over
`ICN_PARITY_MODEL_DIR`. The upstream `test-download-model` fixture setup is
explicitly suppressed, so a qualification run cannot fall back to a network
download. Their evidence records the registry, cache source path, staged path,
digest, size, and copy/existing disposition. Cache artifacts are never hard-linked into the mutable
upstream build tree. Before execution, the wrapper resolves the exact declared test names and
commands through CTest's JSON inventory; after execution, it preserves JUnit evidence and requires
one executed record for every declaration.

Use a dedicated cache rather than a model already staged in a reference build:

```sh
cargo run -p icn-parity -- models --root parity fetch \
  --id stories15m-q4-0 --model-root target/parity-models
```

The cache and build tree must be disjoint, and the staged artifact must have a distinct inode. This
ensures an upstream test can never mutate the content-addressed cache through an alias.

Evidence is create-only. With no `--output-dir`, each invocation creates a
unique run directory below `results/parity/reference`. An explicit output
directory must be empty; prior manifests and inventories are never replaced.
Native build directories are also unique and fresh. The builder runs CMake/CTest under an allowlisted
environment, hashes every consumed configuration input before and after execution, and records
compiler, compile-command, link-command, assertion, and sanitizer evidence in the schema-v3 manifest.
