# Composite inference benchmark

This directory contains the versioned, endpoint-neutral workload used by `benchmark-runner`.
The runner sends controlled streaming requests through `/v1/chat/completions`; it never calls an
ICN backend directly.

## Performance properties

The benchmark measures the complete inference service rather than isolated llama.cpp calls. Its
cases are derived from the independent properties that determine endpoint performance:

1. **Fixed request cost** — HTTP handling, queue admission, chat rendering, tokenization, sampler
   and parser setup, and stream creation.
2. **Prefill efficiency** — the rate and latency of evaluating input tokens.
3. **Decode efficiency** — the rate and latency of generating tokens one step at a time.
4. **Context-depth cost** — the effect of an already populated KV context on generation.
5. **Batch utilization** — how effectively homogeneous prompt or generation work is combined
   across active requests.
6. **Scheduling interference** — how prompt evaluation and generation affect each other when both
   are runnable.
7. **Prefix reuse** — whether sequential or concurrently resident requests avoid repeated prompt
   work.
8. **Application-path cost** — chat templates, tool schemas, tool-call parsing, and multi-turn
   continuation around inference.
9. **Operational recovery** — whether cancellation releases sequence and queue capacity promptly.

Latency, throughput, fairness, memory, cache counts, and errors are measurements of these
properties. They are not separate reasons to add more prompts. Model startup, model quality,
random conversational traffic, and configuration sweeps are intentionally outside the core suite.

## Why these cases are minimal

Each experiment must isolate a property through a contrast while holding the other work constant.
The suite does not take a Cartesian product of every prompt size, output size, concurrency, arrival
delay, and cache state. A new core case is justified only when it answers a performance question
that the existing contrasts cannot answer.

The ordinary workload uses two fixed prompt-size classes (`P_s` and `P_l`) and two exact answer
sizes (`O_s` and `O_l`). Fully rendered prompt counts depend on the model's tokenizer and chat
template, so qualification records the actual counts and strict comparisons require them to match
across targets. That produces the smallest useful serial surface. The same fixtures are then reused
for concurrency and scheduling cases so those experiments do not introduce a different task or
response distribution.

## Scenarios

| ID | Trial | Controlled contrast | Property isolated |
| --- | --- | --- | --- |
| E1-SS | One `P_s` request copying `O_s` | Smallest complete request | Fixed-path sensitivity and serial baseline |
| E1-LS | One `P_l` request copying `O_s` | E1-SS with prompt work increased and output fixed | Prefill cost |
| E1-SL | One `P_s` request copying `O_l` | E1-SS with output increased and prompt work near-matched | Sustained decode cost |
| E1-LL | One `P_l` request copying `O_l` | E1-SL with deeper starting context | Context-depth and prefill/decode interaction |
| E2-prefill | Fixed populations of E1-LS at concurrency `1, 2, 4, ...` | Same total work at each declared population | Prefill batching and scaling |
| E2-decode | Fixed populations of E1-SL at concurrency `1, 2, 4, ...` | Same total work at each declared population | Decode batching and scaling |
| E3 | Start E1-SL; submit E1-LS after the first request emits its first semantic output | Isolated E1 baselines versus an event-aligned overlap | Prefill/decode interference and scheduling fairness |
| E4-exact | Cold request A followed by the identical A | Exact repeat versus cold work | Sequential exact-prefix reuse |
| E4-partial | Cold A followed by A′ with the same long prefix and a different short suffix | Partial reuse versus exact and unrelated arms | Sequential partial-prefix reuse |
| E4-unrelated | Equal-sized U followed by V, diverging near the beginning | Negative control for warmup and connection reuse | Cost unrelated to prefix reuse |
| E5-shared | Four simultaneous requests with one long common prefix and distinct suffixes | Same arrival barrier, sizes, and output work as E5-independent | Concurrent shared-prefix handling |
| E5-independent | Four simultaneous requests diverging near the beginning | Negative control for ordinary batching | Concurrent work without prefix sharing |
| E6 | Forced edit-tool call, fixed tool result, then exact acknowledgement | One valid tool and one valid argument tuple | Tool/template/parser and multi-turn application path |
| E7 | Start E1-SL, cancel after eight generated tokens, immediately run E1-SS | Recovery request versus its E1-SS baseline | Cancellation cleanup and capacity recovery |

E2 is a fixed-work closed-loop trial: at concurrency `C`, it admits a frozen `N = kC` requests,
replacing a completed request until all `N` have been admitted, then drains them. It does not use a
fixed-duration window, because that would leave different unfinished work on faster and slower
targets. E3 uses an observed output event rather than a millisecond delay so both endpoints enter
the intended prefill-during-decode situation regardless of their absolute speed.

## Prompt and response contracts

E1–E5 and E7 use the same mechanical copying task. A request contains a literal, generated
`ANSWER_BLOCK`; the only valid response is the exact bytes inside that block. Short and long answer
blocks are fixed artifacts, not text invented by the model. Filler records change rendered prompt
length without changing the task or answer distribution.

```text
ANSWER_BLOCK_BEGIN
amber cedar cobalt delta ember fable granite harbor iris juniper
amber cedar cobalt delta ember fable granite harbor iris juniper
...
ANSWER_BLOCK_END

Copy exactly the bytes between ANSWER_BLOCK_BEGIN and ANSWER_BLOCK_END.
```

The published fixture's answer blocks and output-token counts are qualified together against its
benchmark model before release; another model is a separate cohort and must qualify the same
artifacts before its timings can support a strict comparison. The generation limit is the qualified answer-token
count. A measured request is valid only when its response bytes equal the answer block, its
generated-token count equals the declared work, and its finish behavior matches the contract.
Greedy decoding and matched model/template settings reduce numerical variation, but the prompt does
not ask the model to “be deterministic.” Determinism comes from there being one literal answer and
from rejecting every other response.

E6 uses a synthetic file containing exactly one `OLD_STATUS` occurrence. `tool_choice` forces
`replace_text`, and the JSON schema restricts `path`, `old_text`, and `new_text` to the sole valid
tuple. The runner returns a fixed tool result without mutating the filesystem. The second response
must equal `EDIT_APPLIED`; containment, explanations, additional tool calls, and retries fail the
contract.

`ignore_eos: true` is used for fixed-token synthetic workloads, and `cache_prompt: false` prevents
reuse outside E4/E5. Cache cases enable reuse deliberately. Tool responses use normal EOS behavior
because their natural semantic completion is part of the contract. These llama.cpp-compatible
controls equalize execution work; they do not substitute for response validation.

## Measurements and validity

The runner records client-observed response headers, first semantic output, every SSE event, stream
completion, and scenario makespan. When exposed, it also records server prompt, decode, cached-token,
queue, sampling, and parser timings. Process RSS is sampled for managed local PIDs.

Primary interpretations are serial TTFT and latency, prompt and decode rates, fixed-work goodput,
concurrency scaling, mixed-work inflation, evaluated prompt work saved, tool transaction latency,
and recovery latency. A timing comparison is invalid when prompt work, generated work, response
contract, schedule, cache state, or required provenance differs. Raw repetitions are retained;
outliers are not silently removed.

## Running the benchmark

Profiles are data, not registered code. Every `*.toml` file under `benchmark/profiles/` is
discovered and validated automatically. `cases` accepts exact selectors such as `E1/ss` or
experiment wildcards such as `E1/*`; the same selector mechanism applies across E1-E7. Adding,
renaming, or removing a profile therefore changes only the profile file. The CLI requires an
explicit `--profile` and has no compiled-in profile default. Parameters that apply only to E2 or to
controlled adaptive stopping are omitted elsewhere and rejected if supplied where they have no
effect. Pairing is not a profile choice: every two-target comparison is matched automatically.

Validate the assets from `inference/`:

```sh
cargo run -p benchmark-runner -- validate --root benchmark
```

Run one endpoint:

```sh
cargo run -p benchmark-runner -- run \
  --root benchmark \
  --profile development \
  --kind icn \
  --endpoint http://127.0.0.1:8080 \
  --model my-model \
  --output results/benchmark/local
```

Compare ICN and llama.cpp:

```sh
cargo run -p benchmark-runner -- compare \
  --root benchmark \
  --profile controlled \
  --candidate-kind icn \
  --candidate-endpoint http://127.0.0.1:8080 \
  --candidate-model my-model \
  --reference-kind llama-cpp \
  --reference-endpoint http://127.0.0.1:8081 \
  --reference-model my-model \
  --model-sha256 MODEL_SHA256 \
  --controlled-host \
  --exclusive-device \
  --set context_size=8192 \
  --set max_sequences=4 \
  --output results/benchmark/controlled
```

`smoke` and `development` are diagnostic. A controlled result is valid only when the recorded
model, template, effective configuration, cache state, host, and completed token work support the
claim. Every comparison runs matched blocks, derives the initial target order from the run ID, and
then alternates it to keep AB/BA exposure balanced. Controlled comparisons continue from the
profile repetition count up to `max_repetitions` until all primary ratio intervals meet the
precision target; stopping is checked only after a balanced AB/BA block. Raw
request events and every repetition are retained in `run.json`. API keys are read from an
environment variable and are never serialized into evidence.

`variance-short` runs only E1-SS after three warmups and records 20 repetitions. It is intended to
measure single-request noise without allowing the longer E1 arms to alter server state. With
`compare`, those repetitions are automatically matched across the two live targets and their order
alternates. Warmups run once per target before the first block.

Standalone one-target results can be joined later for regression or cross-host comparison:

```sh
cargo run -p benchmark-runner -- compare-evidence \
  --candidate results/benchmark/current/run.json \
  --reference results/benchmark/baseline/run.json \
  --output results/benchmark/regression
```
