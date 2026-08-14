# Inference benchmark

Library-first Bun benchmark for instrumented, llama.cpp-compatible Chat Completions engines. The
benchmark downloads pinned BFCL V4 records, compiles one immutable semantic workload plan, and
replays the same messages, tools, schedules, and serving policy against every target.

Fetch and inspect the corpus:

```sh
bun benchmark corpus fetch
bun benchmark corpus status
```

List or prefetch an immutable model profile:

```sh
bun benchmark models list
bun benchmark models fetch qwen3.6-35b-a3b
```

Model downloads use the standard Hugging Face cache (`$HF_HOME/hub`, otherwise
`~/.cache/huggingface/hub`) and are reused by Magnitude. The built-in profile pins the repository,
commit, GGUF file, byte size, and SHA-256. `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` is used for gated
repositories.

Inspect the deterministic workload without running inference:

```sh
bun benchmark explain agent-core qwen3.6-35b-a3b \
  --profile standard \
  --context 32768
```

Run one existing endpoint:

```sh
bun benchmark run agent-core qwen3.6-35b-a3b \
  --endpoint http://127.0.0.1:8080 \
  --context 32768 \
  --profile smoke
```

Compare managed ICN and llama.cpp:

```sh
bun benchmark compare agent-core qwen3.6-35b-a3b \
  --icn-executable inference/target/benchmark-release/bin/magnitude-icn \
  --llama-executable /path/to/llama-server \
  --profile standard
```

`compare` resolves the profile, downloads it once if necessary, verifies it, and identifies the
same GGUF for ICN and llama.cpp by SHA-256. A direct immutable Hugging Face resolve URL can be used in
place of a profile. `--model-path /models/model.gguf` remains an optional offline/development
override; it is not part of the normal workflow.

Both managed targets use the same port sequentially, so only one model is loaded at once. The model
stays loaded throughout that target's entire evaluation. Memory is sampled from the managed process
tree; no server metrics or memory endpoint is used.

`--context` is logical context per sequence and `--sequences` is the serving capacity. Both values
are part of the plan digest. The llama.cpp adapter multiplies them for its shared `--ctx-size`;
ICN's reported allocation must match them before measurement starts.

Every target must return standard terminal usage, including cached prompt tokens, and the
llama.cpp-compatible `timings` object. The benchmark rejects missing or inconsistent evidence; it
never estimates token counts, prefill time, or decode time from text length or client latency.
Workload checkpoints are semantic depths and concurrency levels, not predicted token counts.
Reports use each target's terminal counts for actual work and label a comparison as a product
comparison if corresponding requests render different prompt-token counts.

## Workload coverage

| Workload | Responsiveness | Prefill | Decode | Memory | Distribution |
| --- | --- | --- | --- | --- | --- |
| Single request | Baseline TTFT | Isolated input rate | Isolated output rate | Baseline and peak | Variance and failures |
| Sequential session | TTFT by depth | Rate by context depth | Rate by depth | History growth | Turn-level tails |
| Independent concurrency | TTFT under load | Prefill under load | Decode under load | Batch peak | Throughput and fairness |
| Forked concurrency | Branch TTFT | Prefix reuse and uncached rate | Branch decode rate | Prefix and branch footprint | Throughput and fairness |
| Concurrency pressure | TTFT by concurrency | Prefill degradation | Decode degradation | Footprint by concurrency | Saturation and tails |
| Memory pressure | TTFT with resident histories | Prefill under memory load | Decode under memory load | Peak and retained scaling | OOM and failure onset |

For a local GGUF, the trained context limit and chat-template digest are read directly from model
metadata. `--context` is an optional upper bound. Existing endpoints still require the exact GGUF
through the model profile or `--model-path`; it establishes artifact identity and must match the
artifact served by the endpoint.

For reproducible custom target sets, use a JSON configuration:

```json
{
  "suite": "agent-core",
  "profile": "standard",
  "model": {
    "id": "qwen3.6-35b-a3b",
    "artifactPath": "/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
    "contextLimit": 32768
  },
  "targets": [
    {
      "kind": "existing",
      "id": "local-server",
      "endpoint": "http://127.0.0.1:8080",
      "servedModel": "qwen3.6-35b-a3b",
      "parallelSequences": 4
    }
  ],
  "output": "benchmark-results/result.json"
}
```

```sh
bun benchmark execute benchmark.json
```

The public library exports `prepareCorpus`, `compileTrialPlan`, `evaluate`, `compare`, target
builders, analysis functions, and report renderers from `src/index.ts`.
