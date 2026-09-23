# Validation generators

Fixture JSON is generated locally under ignored `results/fixtures/`. Only the
generators belong in source control. A clean checkout needs generation before
compiling tests that include fixtures:

```sh
uv run inference-v4/validation/generate_fixtures.py
uv run inference-v4/validation/generate_fixtures.py --test -- -p seismic-engine --test sampling --test vision_math --test sequence_program
```

The driver pins NumPy, uses the local `inference-v3` numerical references, and
generates all eleven codec, decoder, attention, recurrence, rotary, routing,
sampling, vision, and erf/GELU fixtures. No model download or GPU is required.
Vision and erf/GELU references use independent Python equations. `--source`
selects another V3 source directory; `--output` selects another output directory
when running generation without `--test`.

The driver completes every generator before replacing existing fixtures. The
individual reference generators record their source and generator hashes;
the erf/GELU array is produced by `qwen_vision_merger_reference.py --erf`.

Measurement reports also belong under ignored `results/`;
`v3_qwen_forward_bench.py` generates the full-model V3 measurements.

## Native V4 Session Bench

`v4_sessionbench.py` uses the unchanged V3 Session Bench fixtures, runner,
client, and report against the native V4 `magnitude-engine` binary. It starts
the server itself, checks `/health`, and obtains prompt token counts from the
server's `/v1/count` endpoint. Each `--artifact` runs sequentially in its own
Session Bench run, with two fresh server passes when `--repeat 1`. Run it with
an existing V3 checkout virtual environment containing the Session Bench
dependencies after the V4 binary is built. A lightweight environment with
versions pinned from the V3 lockfile is sufficient; a full V3 `uv sync` is not
required:

```sh
inference-v3/.venv/bin/python inference-v4/validation/v4_sessionbench.py \
  --source inference-v3 \
  --binary inference-v4/target/release/magnitude-engine \
  --results inference-v4/validation/results \
  --artifact /absolute/path/Qwen3.5-4B-Q4_K_M.gguf \
  --artifact /absolute/path/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --suite context --context 16384 --workload prose --repeat 1
```

Use `--suite single` for a short smoke run. Use `--context 65536` with the
`context` suite for the longer context pass and `--workload retrieval`
for the pinned RULER-derived retrieval fixture. `--storage-gib` and `--method`
pass directly to the V4 binary. The result's `v4-launcher.json` records the
V4 source hash and per-file hashes (Rust, Seismic, Metal, Cargo inputs,
validation Python, and the complete vendored native template build tree,
including C++ sources, headers, and provenance), plus a separate V3 Session
Bench source hash and per-file hashes (runner, client, report, fixtures,
fixture lock data, `pyproject.toml`, and `uv.lock`). It also records the V4
binary hash and size, requested context, installed runner dependency versions,
the adapter identity, local artifact path, and count endpoint. The launcher
checks both source sets and the binary for changes during each run. The V3
target schema still calls the engine `magnitude`; the runtime evidence
identifies the native V4 implementation.

## Locked catalog tensor types

`catalog-tensor-type-audit.md` records the E0.5 conclusions. Regenerate its
ignored per-file evidence from immutable catalog revisions with:

```sh
uv run --with huggingface-hub inference-v4/validation/audit_catalog_tensor_types.py
```
