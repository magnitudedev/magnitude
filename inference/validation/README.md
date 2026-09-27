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

## Model-family references

Independent float32 references of the in-scope model families, written from the released modeling
code (transformers 5.17.0) and configs over the exact dequantized GGUF weights:
`llama_reference.py` (MiniCPM5), `lfm2_reference.py`, `laguna_reference.py`, `muse_reference.py`,
`gemma4_reference.py`, `nemotron_h_reference.py`, and `dflash_reference.py` (DFlash/DSpark drafts over
any of them). Shared pieces: `reference_gguf.py` (GGUF read/dequantize/write, local or HF byte
range), `reference_model.py` (float32 ops, on-demand weights), `reference_cli.py`.

```sh
uv run inference/validation/lfm2_reference.py check-hf --model M.gguf --device cuda    # vs transformers
uv run inference/validation/lfm2_reference.py check-weights --model M.gguf --checkpoint HF_DIR
uv run inference/validation/dflash_reference.py check --draft D.gguf --target T.gguf    # vs released draft code
uv run inference/validation/generate_fixtures.py --set families                         # synthetic variation fixtures
uv run inference/validation/precision/model_base.py --model M.gguf --tokenizer ORG/REPO@SHA --label L --chunks 171,171,170
uv run inference/validation/layer_slice.py --repository ORG/NAME --revision SHA --path SHARD-1.gguf --output S.gguf
```

`handoffs/26-09-27/model-families/references.md` records the verification results, the fixture
format and the differences found between definitions.

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
