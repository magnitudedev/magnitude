# Assessment fixtures

Real GGUF headers with the verdict the engine gives each one. The engine and the service both
test against this set (see `design/model-management/assessment-and-ranking.md`, Engine boundary).
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

To add a header, run `python3 -I fetch_header.py REPO REVISION PATH`. It fetches with ranged GETs
only (the first few MB, never the whole file), then trims and gzips into `headers/`. `HF_TOKEN` is
sent when set. A header from Ollama's registry (`ollama-<model>-<tag>.gguf.gz`) is the model
layer's blob read the same way and trimmed by the same `trim`. The manifest lists Hugging Face
files only, so such a header serves the service's tests alone; `ollama-qwen3.6-35b-a3b.gguf.gz` is
`registry.ollama.ai/library/qwen3.6:35b-a3b`, blob
`sha256:d372de8e934898a59e6ccfabc3368474711384d8f1fd4d22d87a3f0a45400cdc`.

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
  `{ "WeightTypes": { "types": [{ "ggml_type": "Q2_K", "tensor_count": 147 }] } }`,
  `{ "Architecture": { "architecture": "deepseek4" } }` or `"Unreadable"`.
- A `TargetMismatch` detail is developer text. The expected detail only has to occur in the actual
  one. Every other reason must be equal.

Every `Unsupported` verdict and every rejected drafter carries at least one reason.
