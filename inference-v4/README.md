# Inference V4

Rust inference on Seismic. This workspace is under implementation; it is not yet a
replacement for V3 or a qualified release.

Start with the [architecture overview](docs/overview.md), then the
[engine](docs/engine/overview.md) and [Seismic](docs/seismic/overview.md) contracts.
These local docs define intended architecture. The [master implementation spec](/Users/anerli/notes/specs/26-09-17/inference-v4-master.md)
and [accounting research/spec](/Users/anerli/notes/specs/26-09-17/seismic-accounting.md)
provide implementation context; older conflicting mechanisms are superseded by the
local architectural contracts. V3's active behavior and numerical contracts remain
the preservation reference.

Current packages are the audited foundation: language/checker/interpreter, Metal
emission/runtime, a shared scalar realization with native CPU/CUDA execution, derived
accounting, a device-free Rust artifact/Qwen binding layer, and development CLI. The existing Qwen decode
harness lives under `validation/qwen35-poc`; it is a development reference, not the
production engine. New owners are added as their implementations are introduced.

From this directory:

```sh
cargo test --workspace
cargo run -p seismic-cli -- check seismic-std/lib
cargo run -p seismic-cli -- account seismic-std/lib --fn projection --shape N=128,K=256
cargo run -p seismic-cli -- run seismic-std/lib --fn projection --shape N=128,K=256 --target cpu
```

Metal execution is available on macOS. CUDA scalar execution uses the installed driver
and requires an explicit block-size candidate; neither backend is performance-qualified.
`account --target cpu|cuda` also reports concrete SSA requests and scratch, separate
from physical transactions or runtime predictions. `--loads borrow-proven` is an
explicit alternative to materialized snapshots, not automatic tuning. The validation-only Qwen harness requires
`--features metal-poc`; it is excluded from ordinary portable workspace builds.

Read [validation/continuation.md](validation/continuation.md) for actual progress,
known limitations and the next concrete steps. Generated evidence belongs under the
ignored `validation/results/` directory; small identity/qualification manifests belong
in source control.
