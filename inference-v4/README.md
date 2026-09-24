# Inference V4

Rust inference on Seismic. This workspace is under implementation; it is not yet a
replacement for V3 or a qualified release.

Start with the [architecture overview](docs/overview.md), then the
[engine](docs/engine/overview.md) and [Seismic](docs/seismic/overview.md) contracts.
These local docs are normative. Seismic is governed by the
[structured authoring spec](../specs/26-09-18/seismic-structured-authoring-spec.md);
the [implementation plan](../specs/26-09-19/seismic-structured-authoring-implementation-plan.md)
records the decisions taken where the spec is silent. Where any other engine
planning document describes Seismic differently, the structured authoring spec and
the local docs win. V3's active behavior and numerical contracts
remain the preservation reference.

## Status

- **Seismic** is a language of authored execution structure with joint selection of
  implementations, contiguous fusion groups, and numerical sites. Pipeline: check →
  structured IR → joint family → budgeted selection → instantiation → Metal
  realization → MSL → native kernel. A checked feasible witness is executable.
- **Metal, the CPU and CUDA are the backends** on this pipeline. Metal is the reference
  backend. The CPU (Cranelift) and CUDA (PTX) backends cover the same structure with scalar
  code only; both agree bit for bit with the reference interpreter on the kernel table under
  exact numerics.
- **Qwen3.5-4B runs prefill and decode on Metal through this pipeline**, with every
  kernel authored in Seismic and logits matching the V3 reference. Performance work
  is ongoing; the estimate model behind selection is unqualified and no performance
  claim is made here.
- The engine owns artifacts (MLX and GGUF import through Seismic kernels), Qwen3.5
  geometry, state, generation, chat, and serving.

## Commands

From this directory. Run one Cargo process at a time.

Check every library source for Metal:

```sh
cargo run -p seismic-cli -- check seismic-std/lib engine/lib
```

Inspect a selection (occurrences, candidates, sites, covers, seed and selected
witness, estimates, proof status), or print the MSL of the selected witness:

```sh
cargo run -p seismic-cli -- select seismic-std/lib --fn linear --shape M=4,N=5,K=64 --element T=bf16,U=q4g64,V=bf16
cargo run -p seismic-cli -- emit   seismic-std/lib --fn linear --shape M=4,N=5,K=64 --element T=bf16,U=q4g64,V=bf16
```

`--shape` binds every shape parameter of the entry and `--element` every element
parameter. `select` and `emit` use the local Metal device's limits when one opens and
documented default limits otherwise. There are no implementation, tiling, or
candidate flags: those decisions belong to selection. Related commands:
`analyze-search` (search structure of an entry, same flags), `bindings` (Rust
bindings of an entry), `print` (canonical source).

Metal kernel sweep — every standard kernel the Qwen path uses is selected, compiled,
run on Metal, and compared with the reference interpreter:

```sh
cargo test -p seismic-runtime --test kernels -- --ignored
```

Engine reference tests — the structured interpreter against the V3 reference
fixtures, under two partitions each (device-free):

```sh
cargo test -p seismic-engine --test sequence_program --test attention_step \
    --test recurrent_step --test rotary_prepare --test routed --test sampling
```

Interpreter against Metal at real Qwen3.5-4B dimensions:

```sh
cargo test -p seismic-engine --test qwen_metal_reference -- --ignored
```

Full-model prefill and decode on Metal:

```sh
cargo run --release -p seismic-engine --example qwen_baseline -- \
    ARTIFACT CONTEXT PROMPT_IDS CONTINUATION_IDS OUTPUT_JSON
```

`ARTIFACT` is the model artifact path, `CONTEXT` the context length,
`PROMPT_IDS` and `CONTINUATION_IDS` comma-separated token ids, and `OUTPUT_JSON` the
report path. The report records each entry's selection status and estimates, labelled
as estimates, beside measured times. Only a search budget is configurable.

Generated evidence belongs under the ignored `validation/results/` directory; small
identity and qualification manifests belong in source control.
