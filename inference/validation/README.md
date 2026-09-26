# Validation generators

Historical reference JSON can be generated locally under ignored
`results/fixtures/`. Current engine tests do not consume these files. The
generators remain available for independent numerical comparisons:

```sh
uv run inference/validation/generate_fixtures.py
```

The driver pins NumPy and reads the V3 numerical reference source from pinned
Git commit `1fb31d00c548b2da9b5c496ffc7f7df6155c7dda`. It does not need
the `inference-v3/` working tree. It generates ten codec, decoder, attention,
recurrence, rotary, routing, sampling, vision, and erf/GELU fixtures. No model
download or GPU is required.
Vision and erf/GELU references use independent Python equations. `--source`
selects another V3 source directory; `--output` selects another output directory
when running generation without `--test`.

The driver completes every generator before replacing existing fixtures. The
individual reference generators record their source and generator hashes;
the erf/GELU array is produced by `qwen_vision_merger_reference.py --erf`.

Measurement reports also belong under ignored `results/`;
`v3_qwen_forward_bench.py` generates the full-model V3 measurements on a GGUF
file: decode after `--context` tokens of history, prefill chunks (`--prefill`)
and concurrent decode (`--sequences`), selected with `--cells
decode,prefill,concurrent`, each with one per-kernel profile. Run one cell
family per process when the families need different context capacities or
sequence counts; each run writes one JSON file (`--output x.json`, or a
directory receiving `result.json`).

## Session Bench

Session Bench, including the native engine adapter (`--engine magnitude`), is
the `inference/benchmarks` project. For example, a short native smoke run:

```sh
uv run --project inference/benchmarks session-bench run \
  --target magnitude=/absolute/path/Qwen3.5-4B-Q4_K_M.gguf \
  --suite single --context 512 --workload prose-repeat
```

## Locked catalog tensor types

`catalog-tensor-type-audit.md` records the E0.5 conclusions. Regenerate its
ignored per-file evidence from immutable catalog revisions with:

```sh
uv run --with huggingface-hub inference/validation/audit_catalog_tensor_types.py
```
