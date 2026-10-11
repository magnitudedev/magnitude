---
applies_to:
  - inference/evaluation/**
---

# Evaluation

Evaluation tests and measures models. It answers three questions:
- **Correctness:** kernels against synthetic tests; models against an independent reference on the same input.
- **Speed:** at the engine and serving levels.
- **Change:** the first two compared across builds.

Kernel unit and golden tests are not evaluation; they live with the kernels and use synthetic tensors.

## Ownership

| Component | Owns |
|---|---|
| Catalog | Model identity and acquisition: target, variant, drafter and projector, downloaded and verified at locked commits into a model store |
| Fixtures | Every model input |
| Harness | The only in-process engine driver: rendering, timing, output distributions, drafting, tuning traces |
| References | Independent float32 family implementations producing true outputs |
| Bench | Serving over HTTP against our engine and comparison engines |
| Compare | Build-to-build and engine-to-reference deltas |
| Remote | Running any evaluation job on a host |

Models are always selected by catalog reference (`<model>:<variant>`). The drafter and projector are the catalog's own for that model, never overridden. No evaluation tool keeps its own model registry.

## Inputs

**A fixture is a deterministic chat conversation, and it is the only model input.** Raw-text corpora, token lists and random tokens are not accepted anywhere.
- **Rendering:** fixtures are rendered only by the engine's own serving path (the model's chat template and tokenizer), and every rendered output records its exact tokens and template identity.
- **References** consume rendered tokens and never tokenize on their own. A reference and our engine therefore always evaluate identical input.
- **Families:** tool sessions with completed history, prose continue, prose repeat, retrieval and code. Sources are pinned and hash-locked, and builders are deterministic.
- **Quality** is measured by teacher forcing on scored assistant turns. Serving requests use a declared conversation prefix with family-specific answer validation.
- **Sizing:** fitting a fixture to a context length uses the target model's rendered length and takes whole conversational units. Rebuilding the same size with the same fixture and model reproduces identical tokens.

## References

A reference is identified by model, rendered input and implementation. Before use, every reference records dense perplexity and top-1 accuracy against the actual next tokens on the scored spans. A reference outside its bounds fails and must not be used to judge anything. Its failure means the input is outside the model's working regime, not that an implementation under test is wrong.

Quality is reported as a distribution over all scored positions: mean, median, p90, trimmed (top 0.1% removed), p99 and maximum KL, the mean excluding the ten worst positions, the count above 0.5 nats, and top-1 agreement. The mean excluding ten positions is null when no positions remain. These diagnostic summaries retain the full mean and tail rather than hiding sensitive positions. A single position is diagnostic evidence, never a quality measurement or a ratio between models.

## Measurement boundaries

A faster end-to-end answer does not establish faster model execution, and a faster kernel does not establish better serving.

| Boundary | Measures | Appropriate control |
|---|---|---|
| Operator / storage | Completed tensor work, state movement and allocation | Same inputs and state with an independent implementation |
| Model executor | Prefill or decode from a fixed state | Upstream program at matching geometry and output demands |
| Generation | Plain or speculative advancement, acceptance and repair | Independent requests and the same target algorithm |
| Engine | Scheduling, admission, batching, retention and delivery | Matched offered traffic through our engine and an upstream generator |
| Serving | Rendering, HTTP, semantic output and sessions | Actual stock servers through the shared client |

Rules for every measurement:
- Complete asynchronous work inside timing. Keep loading, reset and correctness checks outside it unless the claim includes them.
- Execution parity requires matched rendered input and generated work. Different generated prose or tool encodings are real serving differences, not evidence of slower execution.
- No time or token count is fabricated from text or client latency. Engine counters and terminal usage must agree.
- A correctness gate is never relaxed to make a timing admissible. Unsupported batching or capabilities fail explicitly rather than being measured as something else.
- **Tuning:** first-load tuning runs under an explicit budget. Fresh tuning ignores stored tuning results and keeps compiled kernels. The tuning trace and outcome are part of every measured run.

## Comparison

A comparison measures two builds:
- **Same conditions:** the same measurement code, on the same host, under one lock.
- **Alternation:** speed rounds alternate which side runs first, so drift affects both alike.

**Identical rendered input is a precondition.** If the builds render a fixture to different tokens, that is reported as a finding, and quality is compared on one side's exact tokens rather than silently on different input.

**Verdicts are judged against measured noise:**
- A speed cell improves or regresses only beyond each side's measured spread and the declared bar.
- A quality metric does so only when paired resampling over positions excludes no change and the relative change exceeds the bar.
- Everything else is reported as within noise.

A cell measured on only one side, or a failed step, is a failure, never an omission.

## Tuning traces

Tuning decisions, including precision, are recorded in each run's tuning trace (contract "Tuning trace") and queried directly. Each unit's decision is one self-contained record: its share, the yardstick at each served point and how it was obtained, the chosen configuration and why, the winner's coverage check, and every candidate at every point with its time, error ratio, rounding-level ratio and verdict.

Development probes check the precision model against the real engine:
- **Survey:** every admissible configuration with its error measurements next to its time.
- **Precision probe:** the logits of a pinned selection on rendered fixtures, against the defaults and, optionally, a reference with passing sanity evidence for those exact fixtures. It records measured KL in nats and the trace's dimensionless squared error ratios separately. Those ratios are not a prediction or bound in nats without a measured resolution-noise effect. Missing precision measurements remain null. The model-input rule applies: only rendered chat fixtures.

## Serving benchmark

Each adapter owns its engine's preparation, launch, readiness and cleanup. Shared code owns scheduling, HTTP/SSE, validation and reporting.
- An adapter that can configure serving context sets it to the launched block's context and requires readiness to report that value.
- Traffic shapes (single, session, concurrency, fork, memory) describe offered traffic. History sharing is not evidence of retained-prefix reuse.
- Sampling is greedy, with seed 42 where supported and model-selected thinking disabled.
- **Validation:**
  - Tool calls are matched as a multiset with complete assignment.
  - Retrieval answers are exact string mappings, reported as whole-answer and partial-field accuracy.
  - Truncation, protocol errors, timeouts and cancellation never enter performance summaries, but always enter accuracy denominators.

## Remote execution

Any job can run locally or on a host named by ssh alias or pool.
- Connection details come from ssh configuration. A local, untracked hosts file records only pools, cache roots and model stores. No machine-specific configuration is committed; without the file, every host uses default locations.
- On a host, the job runs as a child of one agent that holds an operating-system file lock.
  - The lock is released when the agent exits.
  - The agent exits when the controlling session's heartbeat stops, and it has a hard maximum hold time.
  - A lock can therefore never outlive its work, and work never outlives its lock.
- Synchronization, building, model acquisition, thermal settling and measurement all happen under the lock.
- Measurement starts only after temperatures are flat and the machine is idle.

## Evidence

Every run writes one result record:
- **Identity:** build (commit plus uncommitted digest), catalog model and artifacts, fixture set and rendered-input digests, host hardware facts, and the reproduction command.
- **Measurements:** speed, drafting, quality, tuning and serving, each present only when measured.
- **Traces:** a whole-run temperature trace and a tuning trace.

Further rules:
- **Hardware facts** are typed; unavailable values are null, and failed probes keep their errors.
- **Temperature** readings are per-sensor and timestamped, from setup through cleanup. Aggregates exclude gaps, and missing probes are explicit errors, never zeros.
- **Failed controls** and raw observations are kept alongside successful results.
- **Comparison** consumes result records only.
