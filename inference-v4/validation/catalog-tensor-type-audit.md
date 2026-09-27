# Locked catalog tensor-type audit

Audit date: 2026-09-21. Blueprint package: E0.5. Dependency row: D11.

## Scope and evidence

The audit resolved all 21 models and 55 variants in
`inference/catalog/models.json` against the exact commits in
`inference/catalog/models.lock.json`. It used the production catalog's format
selector rules, expanded every selected shard set, included declared projector
and file-draft components, and inspected 178 distinct GGUF files.
The audited input digests are:

- `models.json`: `1403129a16e0d37737f5012e113b59618810943bce1fab8ed585289b01f19f5a`
- `models.lock.json`: `49fdf9909ad22132e0113f567ba82c60caddb1395cf7014bd633978ac207a451`

This is deliberately a family-neutral container audit. It records raw GGUF
tensor encodings for target, projector, and draft inventories; it does not
interpret architecture metadata or assign family-specific semantic weight
roles. Those responsibilities remain in each selected family adapter.

All 178 immutable remote files were available and their GGUF tensor directories
were parsed successfully. Local artifact presence was not assessed and is not
used as evidence; this report establishes locked-source header facts, not local
installation presence. The generated per-file evidence is
`validation/results/catalog-tensor-types.json` and is intentionally ignored.

Reproduce from the repository root (network access is required):

```sh
uv run --with huggingface-hub inference-v4/validation/audit_catalog_tensor_types.py
```

The auditor exits nonzero if a locked artifact is unavailable, a tensor type is
unknown, a declared component is absent, or a catalog selector/shard set is
ambiguous or incomplete.

## Result

The complete observed type set is:

`BF16, F16, F32, I32, IQ3_S, IQ4_NL, IQ4_XS, MXFP4, NVFP4, Q1_0, Q3_K, Q4_0, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0`.

The current inference-v4 GGUF reader accepts `F32`, `F16`, `Q8_0`, `Q4_K`,
`Q5_K`, `Q6_K`, and `IQ4_XS`. D11 therefore has eleven concrete gaps:

| Missing type | Locked-artifact evidence |
| --- | --- |
| `BF16` | 49 files across target, projector, and draft roles; 17 catalog models |
| `I32` | DeepSeek V4 Flash target |
| `IQ3_S` | Qwen3.8 27B target |
| `IQ4_NL` | Qwen3.8 27B and Qwen3.8 Flash Next targets |
| `MXFP4` | Nemotron 3 Super and Ultra targets; DeepSeek V4 Flash draft |
| `NVFP4` | Nemotron 3.5 Lightning target and draft |
| `Q1_0` | Bonsai 8B target |
| `Q3_K` | Qwen3.8 27B target |
| `Q4_0` | All five Gemma 4 QAT targets |
| `Q5_0` | Nemotron 3.5 Lightning target |
| `Q5_1` | Nemotron 3 Super and Qwen3.8 Flash Next targets |

D11 must add common reader representations and shared import entries for all
eleven types, not BF16 alone. EA.3 is not complete until every type above
imports from every locked component role in which it occurs. This audit found
no unknown numeric GGML type IDs and no unavailable locked artifacts.
