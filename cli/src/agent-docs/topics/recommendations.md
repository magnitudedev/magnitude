# Model recommendations

`magnitude catalog recommendations` suggests local-model configurations that Magnitude expects to
work well on this computer. It considers whether a configuration is compatible with the machine
and likely to fit in memory, then weighs model intelligence, expected generation speed, and how
faithfully the local artifact preserves the source model.

The default `Balanced` view aims for a useful compromise between speed and intelligence. The
other preferences shift that tradeoff; they are not separate model catalogs. Use `faster` or
`smarter` for normal requests to lean toward speed or intelligence while still meaningfully
considering both. If a user asks for “fast and smart options,” these are the two views to compare.

`fastest` and `smartest` are extremes. `fastest` prioritizes speed and gives intelligence only
limited consideration, so use it only when the user clearly cares little about intelligence and
wants maximum speed. `smartest` prioritizes intelligence and gives speed only limited
consideration, so use it only when the user clearly accepts slow generation in exchange for maximum
intelligence. A recommendation is a starting point for choosing with the user, not a claim that one
model is best for every workload.

## What the displayed properties tell you

- **Speed** is Magnitude's estimate of generation throughput on this computer. The displayed range
  shows how performance may change between shorter and longer contexts; it is not a confidence
  interval. It is a hardware-aware prediction rather than a benchmark run of the downloaded model,
  and real speed can vary with workload and system activity.
- **Memory** is the estimated memory needed while the model is running, not its download size. A
  model that normally fits can still need other memory-intensive applications to be closed before
  it loads.
- **Context** is the amount of conversation and working material available to the local serving
  configuration. It is not necessarily the model architecture's absolute maximum. Magnitude may
  choose a smaller context for compact models because longer context uses more memory and can slow
  generation, especially on resource-constrained computers.
- **Intelligence** is the model's Artificial Analysis Intelligence Index score. Magnitude displays
  it with a percent sign, but it is an index score—not a probability or the percentage of questions
  the model answers correctly. Quantized variants of the same source model generally share this
  model-level score.
- **Accuracy** describes how faithfully the local artifact is expected to preserve the source model
  after quantization. It does not mean factual accuracy and is separate from Intelligence.
- **Acceleration** identifies the speculative-decoding method Magnitude has prepared for that
  configuration. See `magnitude docs speculative-methods` for what `None`, `MTP`, `DFlash`, and
  `DSpark` mean and how Magnitude sets them up.
- **Capabilities** identify supported features such as vision, tool use, structured output, and
  reasoning. These can matter more than a small speed or intelligence difference when the user has
  a specific task in mind.
- **ID** is the exact configuration identifier required by later commands. Preserve it exactly;
  the friendly model name is not a command argument.

Use these signals together. A faster model may feel better for quick iteration, while a more
intelligent model may be worth waiting for on difficult coding or reasoning tasks. Memory and
context determine whether the model is practical on this computer, artifact accuracy indicates how
much quality the local format may give up, and capabilities determine whether it can do the job at
all.

## Artificial Analysis intelligence and reference points

The [Artificial Analysis Intelligence Index](https://artificialanalysis.ai/evaluations/artificial-analysis-intelligence-index)
is an independently run composite of evaluations spanning areas such as mathematics, science,
coding, knowledge, long-context work, and agentic tasks. Higher scores generally indicate stronger
performance across that mixture, but the number is a broad comparison signal rather than a promise
about one particular task.

The frontier models below are useful reference points when a user wants to understand what a local
model's Intelligence score means. For example, a local model near 52 is in the same general score
region as GPT-5.6 Luna in this snapshot, while a score near 61 is around GPT-5.6 Sol. This does not
mean the models behave identically; it gives the user a familiar yardstick for the index.

These scores are a September 1, 2026 snapshot of Artificial Analysis Intelligence Index v4.1.1:

| Reference model | Evaluated configuration | Score |
| --- | --- | ---: |
| [Claude Opus 5](https://artificialanalysis.ai/models/claude-opus-5) | Adaptive reasoning, max effort | 63 |
| [Claude Fable 5](https://artificialanalysis.ai/models/claude-fable-5) | Adaptive reasoning, max effort, Opus 4.8 fallback | 62 |
| [GPT-5.6 Sol](https://artificialanalysis.ai/models/gpt-5-6-sol) | Max effort | 61 |
| [GPT-5.6 Terra](https://artificialanalysis.ai/models/gpt-5-6-terra) | Max effort | 57 |
| [GPT-5.6 Luna](https://artificialanalysis.ai/models/gpt-5-6-luna) | Max effort | 52 |

Compare scores from the same index version and similar reasoning settings when possible. Artificial
Analysis may revise its methodology or results over time. Magnitude ships reviewed scores with its
catalog rather than refreshing them from the network during recommendation, and clearly labels an
authored estimate when a directly measured score is unavailable.
