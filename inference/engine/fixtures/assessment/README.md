# Assessment fixtures

Real GGUF headers with the verdict the engine gives each one. The engine and the service both
test against this set.
When a change adds support for something a fixture uses, the same change flips that fixture's
expected verdict.

The engine's test is `inference/engine/tests/assessment_fixtures.rs`. It assesses every fixture
through `magnitude_engine::assessment::assess_model` on the CPU backend.

## Headers

`headers/<repository file name>.gz` is one gzipped GGUF header: every byte through the aligned
tensor-data offset, with no tensor payload. Decompress it under its repository file name, so a
split target's later shards are found beside the first shard by the split naming convention.

Headers are trimmed so the whole set stays small. Trimming keeps every metadata key, the
vocabulary's length and token kinds, and the whole tensor directory. It empties
`tokenizer.ggml.merges`, and it blanks the text and score of every normal token that no
special-token key names. Assessment reads none of what is cut, and verdicts were checked to be
identical before and after trimming. The tokenizer digest in a fixture's `ModelFingerprint` is the
trimmed table's, so it does not match the full file.

To add a header, run
`cargo run -p magnitude-service-catalog-tool -- fixture-header --repository REPO --revision REVISION --path PATH --model-store DIR`
in `inference/`. It reads the header with range requests only (never the whole file), then trims
and gzips it into `headers/`. `HF_TOKEN` is sent when set. A header from Ollama's registry
(`ollama-<model>-<tag>.gguf.gz`) is the model layer's blob read the same way and trimmed the same
way. The manifest lists Hugging Face
files only, so such a header serves the service's tests alone; `ollama-qwen3.6-35b-a3b.gguf.gz` is
`registry.ollama.ai/library/qwen3.6:35b-a3b`, blob
`sha256:d372de8e934898a59e6ccfabc3368474711384d8f1fd4d22d87a3f0a45400cdc`.

## Weight samples

`weight-samples/<repository file name without .gguf>.json` is one real file's weight sample, as the
service's `WeightSample` serializes it (tensor name to its first 64 values), for the service's weight-lineage tests. To add one,
run `cargo run -p magnitude-service-catalog-tool -- weight-sample --name NAME SOURCE` in
`inference/`, where `SOURCE` is `hub --repository REPO --revision REVISION --path PATH --model-store DIR`,
`ollama --model MODEL --digest sha256:DIGEST --model-store DIR` or `local PATH`; it reads the
header and then only the sampled values, with range requests. Sources
(`ollama-qwen3.6-35b-a3b` is the blob named above):

| Sample | Source |
| --- | --- |
| `Qwen3.8-27B-UD-Q4_K_XL` | `unsloth/Qwen3.8-27B-GGUF@27af057ecb382ddfea5d12837360a8980560e3ed` |
| `Qwen3.6-27B-Q4_K_M` | `lmstudio-community/Qwen3.6-27B-GGUF@58c6607d9c4cae8b071b3781c73be633fb3dee36` |
| `Qwen3.5-27B-UD-Q4_K_XL` | `unsloth/Qwen3.5-27B-GGUF@3221f178a6b842d04f1fb42f1c413534adcc0a6a` |
| `Qwen3.6-35B-A3B-UD-Q4_K_XL` | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF@5bc3e238d916f48a861bac2f8a1990a0e9b7e98d` |
| `Qwen3.5-35B-A3B-UD-Q4_K_XL` | `unsloth/Qwen3.5-35B-A3B-GGUF@bc014a17be43adabd7066b7a86075ff935c6a4e2` |
| `MiniCPM5-2B-Q4_K_M` | `openbmb/MiniCPM5-2B-GGUF@2079a22f3beaa4e306449978533478fe0522f4b3` |
| `MiniCPM5-2B-heretic-abliterated-Q5_K_M` | `Abiray/MiniCPM5-2B-heretic-abliterated-GGUF@c153cc60bed0471d7d26feb02f999a887851a81b` |
| `gemma-4-12B-it-qat-UD-Q4_K_XL` | `unsloth/gemma-4-12B-it-qat-GGUF@980b060c40a8539ac159e0501a3e0f66a6365af3` |
| `gemma-4-12B-it-Q4_K_M` | `lmstudio-community/gemma-4-12B-it-GGUF@40b870babe39398ce917bbf4b4ff9b5a1f12710e` |
| `Qwen3.5-9B-UD-Q4_K_XL` | `unsloth/Qwen3.5-9B-MTP-GGUF@9716a636ee4bddc3fed678220b7a33dd2a4160ae` |
| `Qwen_Qwen3.5-9B-IQ4_XS` | `bartowski/Qwen_Qwen3.5-9B-GGUF@182be2fd6c7bc44887d88a91cb03ff009cc9f549` |
| `Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-MTP-Q4_K_M` | `DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF@d5f3139249b7ee2535c20ab81125c9a6ab0c4915` |

## `manifest.json`

```json
{
  "fixtures": [
    {
      "name": "qwen35-mismatched-dflash",
      "description": "Developer text.",
      "target": [{ "header": "headers/A.gguf.gz", "repository": "owner/repo", "revision": "<commit>", "path": "A.gguf" }],
      "draft": { "header": "headers/B.gguf.gz", "repository": "owner/repo", "revision": "<commit>", "path": "B.gguf" },
      "method": "dflash",
      "expected": {
        "target": "supported",
        "drafter": { "rejected": [{ "TargetMismatch": { "detail": "draft width 4096" } }] }
      }
    }
  ]
}
```

- `target` lists the target's components, first shard first. A split target lists every shard.
- `draft` is an optional separate drafter.
- Each component names its committed `header` and where it came from: `repository`, `revision`
  (an immutable commit) and the repository `path`.
- `method` is optional: `auto` (the default), `plain`, `mtp`, `dflash`, `dspark` or `dflash2`.
- `expected.target` is `"supported"` or `{ "unsupported": [reasons] }`.
  - `"supported"` means `Fits` or `DoesNotFit`, whichever this machine gives. Memory fit depends on
    the hardware, so it is not pinned.
  - `unsupported` lists every reason, in the engine's order.
- `expected.drafter` applies only to a supported target. It is `"none"`, `{ "bound": method }`
  (`mtp`, `dflash`, `dspark` or `dflash2`) or `{ "rejected": [reasons] }`. When it is omitted, the
  drafter is `"none"`.
- Reasons use the serde form of the engine's `UnsupportedReason`, for example
  `{ "WeightTypes": { "types": [{ "ggml_type": "I32", "tensor_count": 3 }] } }`,
  `{ "Architecture": { "architecture": "deepseek4" } }` or `"Unreadable"`.
- A `TargetMismatch` detail is developer text. The expected detail only has to occur in the actual
  one. Every other reason must be equal.

Every `Unsupported` verdict and every rejected drafter carries at least one reason.
