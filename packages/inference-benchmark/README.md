# Inference benchmark

Library-first Bun benchmark for instrumented, llama.cpp-compatible Chat Completions engines. The
benchmark downloads pinned BFCL V4 records, compiles one immutable semantic workload plan, and
replays the same messages, tools, schedules, and serving policy against every target.

Fetch and inspect the corpus:

```sh
bun benchmark corpus fetch
bun benchmark corpus status
```

Experiments are TypeScript modules that bind immutable model artifacts to engines. Inspect and
prepare one before running it:

```sh
bun benchmark experiments list
bun benchmark experiments show packages/inference-benchmark/experiments/qwen36-q8-mlx-lm-vs-llama-cpp.experiment.ts
bun benchmark prepare packages/inference-benchmark/experiments/qwen36-q8-mlx-lm-vs-llama-cpp.experiment.ts
```

Model downloads use the standard Hugging Face cache (`$HF_HOME/hub`, otherwise
`~/.cache/huggingface/hub`) and are reused by Magnitude. Reusable TypeScript model definitions pin
the logical upstream model and every engine-loadable artifact. GGUF artifacts pin the repository,
commit, filename, byte size, and SHA-256; multi-file MLX artifacts use a committed complete-file
lock. `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` is used for gated repositories.

Create a complete MLX lock when registering a new artifact:

```sh
bun benchmark models lock-mlx <repository> <revision> <bits> <group-size> <output.lock.json>
bun benchmark models lock-mlx-unquantized <repository> <revision> <bfloat16|float16> <output.lock.json>
```

Inspect the deterministic workload without running inference:

```sh
bun benchmark plan packages/inference-benchmark/experiments/qwen36-q8-mlx-lm-vs-llama-cpp.experiment.ts
```

Run the prepared experiment and inspect its immutable evidence:

```sh
bun benchmark run packages/inference-benchmark/experiments/qwen36-q8-mlx-lm-vs-llama-cpp.experiment.ts
bun benchmark runs list
bun benchmark runs show <run-id>
bun benchmark runs watch <run-id>
```

Start the local dashboard over those same experiments and run files:

```sh
bun benchmark dashboard
```

The dashboard is served at `http://127.0.0.1:5187`; its API is at
`http://127.0.0.1:4897`. It has no separate configuration or database.

Experiment variants support managed ICN, managed llama.cpp, the UV-frozen MLX-LM adapter, and
conforming existing endpoints. MTP experiments additionally use a UV-frozen MLX-VLM adapter because
released MLX-LM cannot load the Qwen or Gemma MTP architectures used here. Managed targets use the
same port sequentially, so only one model is loaded at once. The model stays loaded throughout that
target's entire evaluation. Memory is sampled from the managed process tree; no server metrics or
memory endpoint is used.

Speculative decoding is an explicit engine setting. Comparable variants must all declare the same
mode: either `{ kind: "none" }` or MTP with an immutable drafter artifact and a shared candidate-token
limit. MTP runs are rejected unless every request returns native drafted and accepted-token counters
and the target demonstrates nonzero drafting. This prevents launch configuration from being mistaken
for evidence that speculation actually ran.

`requestPolicy.contextTokensPerSequence` is logical context per sequence and
`requestPolicy.parallelSequences` is serving capacity. Both values are part of the plan digest. The
llama.cpp adapter multiplies them for its shared `--ctx-size`; ICN's reported allocation must match
them before measurement starts.

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
| Context scaling | TTFT by prompt size | Full-prompt rate by size | Decode by size | Peak by size | Long-context degradation |

For a focused long-context comparison, a `context-sweep` suite stacks canonical completed BFCL
interactions until the shared serialized messages and tools are closest to each configured character
target. Token targets use a declared approximation such as 3.5 characters per token; both engines
receive exactly the same content, and reports use each engine's terminal prompt-token count as the
actual context length. Context-scaling reports keep BFCL correctness separate from measurement:
semantically invalid completions remain visibly invalid, but complete native timing evidence is
still summarized. Protocol and transport failures are never measured.

Experiments may set `requestPolicy.requestTimeoutMs` for trials expected to exceed the five-minute
default, such as 200K-context prefills.

For a GGUF artifact, the trained context limit and chat-template digest are read directly from model
metadata. The experiment's context policy is an upper bound. Existing endpoints still reference an
exact artifact in the TypeScript experiment; that artifact establishes the claimed model identity.

Reusable model definitions separate a logical model from its concrete artifacts. Experiments import
those artifacts and bind each one to an engine:

```ts
export default defineExperiment({
  id: "qwen36-q4-engine-comparison",
  title: "Qwen3.6 35B-A3B 4-bit engine comparison",
  suite: agentCore({ profile: "standard" }),
  requestPolicy: {
    contextTokensPerSequence: 32768,
    parallelSequences: 4,
    maxOutputTokens: 256,
    temperature: 0,
    topP: 1,
    seed: 42,
    enableThinking: false,
  },
  variants: [
    {
      id: "llama-cpp-q4",
      artifact: qwen36.artifacts.llamaQ4,
      engine: llamaCpp({ /* engine-only settings */ }),
    },
    {
      id: "mlx-lm-q4",
      artifact: qwen36.artifacts.mlx4,
      engine: mlxLm({ /* engine-only settings */ }),
    },
  ],
  execution: { blocks: 2, variantOrder: "balanced" },
})
```

The public library exports `prepareCorpus`, `compileTrialPlan`, `evaluate`, `compare`, target
builders, analysis functions, report renderers, experiment builders, preparation, and run lifecycle
operations from `src/index.ts`.
