# Experimental pipeline: review and qualification

Prepared locally on 2026-10-03; this reviewer package is not public yet. Raw logs,
private paths and machine identities remain outside the branch. Evidence filenames
below are local record identifiers, not downloadable links. No evidence was uploaded.

## Review order

Review these eight boundaries in order (the five implementation commits remain intact):

1. **Global layer identity and logical publication** — [stage identity](../state/src/stage.rs),
   [layout](../state/src/layout.rs), [ordered publication](../state/src/pipeline.rs).
   Check that global identity never becomes a local ordinal, and all stages preflight
   before any accepted state publishes.
2. **Backend-neutral placement and concrete two-CUDA admission** — [placement](../executor/src/placement.rs),
   [projection](../executor/src/pipeline/model.rs), [admission](../executor/src/pipeline/execution.rs).
   Check complete original-block coverage, backend-neutral intent and explicit exactly-two-CUDA admission before opening devices.
3. **Stage-local ownership, assigned imports and resource admission** — [stage](../executor/src/pipeline/stage.rs),
   [planning](../executor/src/planning/resources.rs), [resource owner](../executor/src/domain/pipeline_resources.rs).
   Check direct assigned-role imports, independent budgets and exact opened-device/arena allocation adoption.
4. **Ordinary target lifecycle and joint reconciliation** — [family](../executor/src/domain/pipeline_family.rs),
   [target](../executor/src/domain/pipeline_target.rs), [reconciliation](../executor/src/domain/reconcile.rs).
   Check reservation restoration before submission, discard after physical failure and joint logical acceptance.
5. **Physical execution and activation handoff** — [transfer](../executor/src/pipeline/transfer.rs),
   [execution](../executor/src/programs/native_target/pipeline.rs), [shared target](../executor/src/programs/native_target.rs).
   Check producer/consumer completion, bounded geometry/ownership, and shared state binding/readout semantics.
6. **Engine/worker/serving selection and default-path preservation** — [construction](../src/execution/pipeline.rs),
   [worker](../src/worker/mod.rs), [session](../src/worker/session.rs), [Owner](../scheduler/src/owner.rs), [CLI](src/main.rs).
   Check per-device preparation, bounded control yields, typed failure propagation and unchanged ordinary selection.
7. **Independent numerical and lifecycle controls** — [domain tests](../executor/src/domain/domain_tests.rs),
   [numerical qualifier](../executor/src/pipeline/qualification.rs), [worker qualifier](../src/execution/pipeline/qualification.rs).
   Check independent unsliced control, full state/logit observations and same-process recovery/reload; ignored tests are not CI passes.
8. **Contracts and reproduction instructions** — [design contract](../../../design/inference/engine/pipeline-execution.md),
   [CLI guide](README.md) and the commands below. Check that guarantees and evidence remain within the admitted profile.

## Endpoint provenance

Hardware-qualified source: `15c6f264e812e30e9a989ce6786308d1ec2cd46f`
(tree `10eab3cec03d53c77da8fcab0981959a814ae6c0`).
Reviewed implementation: `ba112f763a3bb97d07f87cf9b5f869f10c766b55`
(tree `32ac923ec4644d90c90d72a92dc6b67e6a60a43f`).
Pinned base: `cffe46e77af5f45ce99565f096d545443dbbd3d0`.
The former history is preserved locally as `review-start-15c6f264-20261003`.

The direct two-ref diff was inspected in full: three files, six additions and five
removals. This is an endpoint-tree comparison, independent of commit reorganization.

| File/region | Actual change | Effect on execution/build/test | Evidence disposition |
|---|---|---|---|
| `executor/src/placement.rs`, type doc comment | “Ordered pipeline groups” → “Ordered execution groups” | Documentation metadata only; no executable tokens, assertions or attributes changed | Reuse all relevant gates |
| `design/inference/engine/pipeline-execution.md`, opening paragraph | Same terminology correction | No executable or build input | Reuse |
| `cli/README.md`, server/curl example | Literal port `38892` → quoted caller `$PORT` in both commands; available-port setup instruction added | Changes the documented endpoint invocation, not CLI implementation; server and client still address the same caller-selected port | Reuse historical API results at their recorded ports; verify revised command syntax/source separately |

Every other tracked input is identical, including kernels, manifests, build scripts,
feature gates, test helpers/assertions and qualifier invocations. The one changed Rust
line is a doc comment. No GPU rerun is required by this comparison. Hardware tests
ran on the earlier source, **not** on the reviewed commit; source equivalence for
these gates does not make distinct binaries identical.

### Gate matrix

All retained hardware results below come from the `cleanup-20261002` campaign.
The numerical/worker source association is supported by the retained pre-campaign
`semantic-head.txt`, exact `final-semantic.patch` (byte-identical to base→tested diff),
`final-validation.sh`, final source records and API launch records. Their test logs
record executable filenames, not SHA256 digests. Do not invent missing binary hashes.

| Gate | Disposition | Actual source/build and decisive result | Local evidence identifiers |
|---|---|---|---|
| Qwen3.5 numerical | PREVIOUSLY PASSED — REUSE SUBSTANTIATED BY SOURCE COMPARISON | Tested source; unoptimized test profile, `experimental-pipeline-cuda`; `magnitude_executor-2a34ce68ffcd9f0d` (no recorded binary SHA). 374 bit-exact groups at cuts 12/16, both prompts; full logits/features/KV/recurrent state/positions/feedback, 32 generated candidate tokens; abs/relative error and nonfinite counts zero | `final-numerical.txt`, `.exit`, `final-validation.sh` |
| 35B single-card refusal and paired API | PREVIOUSLY PASSED — REUSE SUBSTANTIATED BY SOURCE COMPARISON | Tested source; unoptimized dev normal experimental build. Both single-card starts refuse with typed `LoadFailed(InsufficientMemory)` before resident target imports; split 21/40 serves two fresh 16-token requests with deterministic text | `pr1-large-single-capacity-result.json`, `pr1-large-single-capacity-gpu0.txt`, `-gpu1.txt`, `pr1-large-api-paired-launch.json`, `-request-0.json`, `-request-1.json`, `large-api.exit` |
| Ordinary public API generation | PREVIOUSLY PASSED — REUSE SUBSTANTIATED BY SOURCE COMPARISON | Tested source; same normal binary. Small model matches ordinary single-device control for both prompts; 35B exercises ordinary Chat Completions serving | `pr1-small-api-{control,paired}-launch.json`, `-request-{0,1}.json`, `small-api.exit` |
| Live cancellation and fresh recovery | PREVIOUSLY PASSED — REUSE SUBSTANTIATED BY SOURCE COMPARISON | Tested source; worker test profile and normal API binary. Worker stop/drain and fresh generation pass; small/large HTTP streams disconnect after content, then fresh deterministic requests pass | `final-worker.txt`, `.exit`, `pr1-{small,large}-api-paired-concurrent-refusal.json`, `-disconnect-recovery.json` |
| Partial-stage failure after prefix/transfer | PREVIOUSLY PASSED — REUSE SUBSTANTIATED BY SOURCE COMPARISON | Tested source; **separate** unoptimized dev `pipeline-fault-injection` build. Position 2: prefix completed, activation transferred, suffix not submitted; HTTP 500, no successful token response, worker unload, allocation release; later normal-build fresh recovery passes | `fault-build.txt`, `.exit`, `pr1-suffix-fault-api-{launch,request,discarded,shutdown}.json`, `-server.txt`, `suffix-fault-api.exit`, `recovery/pr1-small-api-*.json`, `final-fresh-recovery.exit` |
| Same-process unload/reload and graceful shutdown | PREVIOUSLY PASSED — REUSE SUBSTANTIATED BY SOURCE COMPARISON | Tested source; `magnitude_engine-c654ce07d4a2a7af` test executable (no recorded SHA), experimental feature. Reload in one worker process preserves readiness census and generation; worker shutdown passes. Separate API shutdowns exit 0 | `final-worker.txt`, `.exit`, `pr1-{small,large}-api-paired-shutdown.json` |
| Default single-device regression | RERUN ON REVIEWED TREE | Retained review campaign, unoptimized tests, no experimental feature: state 87, engine 19, scheduler 38, serving 79, CLI 3; default all-target check passes. Small-model single-device API hardware control remains prior tested-source evidence | `review-20261003/{state,engine,scheduler,serving,cli,default-check}.{txt,exit}`, `validate.sh`, `final-tree.txt` |
| Full executor suite | KNOWN BASELINE FAILURE | Reviewed tree: 156 passed, **2 failed**, 1 ignored; absent catalog-header fixtures. Configured NVRTC rerun supersedes setup failures, not the two fixture failures | `review-20261003/executor-configured.{txt,exit}` |
| Ignored manual qualifiers in ordinary CI; hardware loss/mid-kernel fault; independent single-card 35B numerical control | NOT RUN / EVIDENCE GAP | Ignored does not mean passed. These controls were not run or cannot fit on one card; no such qualification is claimed | No successful evidence |

“RERUN ON REVIEWED TREE” refers to retained review-campaign execution on its
recorded tree, not a new run during this documentation audit. No hardware or test
suite was newly run in this audit; documentation checks and passive CLI help were.
The two fixture failures are
`assessment::catalog::every_admitted_catalog_target_derives_complete_terms` and
`assessment::catalog::separate_drafts_plan_and_charge_every_graph_class_from_headers`.

Normal API binary SHA256: `9a2da9008794bdc5712cd0268040738b87ffe64b606d88f41cde53baf3c44600`.
Development fault API binary SHA256: `11228a9cae1b5237d75d117d7d392db5be8af802332d3d6f773a8bcec55f431c`.
The single-card capacity runner recorded source and command but no per-run binary
hash; its binary is associated with the normal build by campaign command order,
not independently digest-attested. Recovered file hashes in the private local
index identify retained evidence today, not contemporaneous executable attestations.

### Artifacts, capacity and limits

| Qualification artifact | Recorded identity |
|---|---|
| `unsloth/Qwen3.5-4B-MTP-GGUF`, `Qwen3.5-4B-UD-Q8_K_XL.gguf` | Snapshot `86835bf9949e4d14d6860f7910b1340ad4f271a9`; 6,065,971,520 bytes; SHA256 `5ede74f75757ab29acfe76b99f57b0ba6d5bedf7ac921e516654bcb7055adafd` |
| `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, `Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf` | Snapshot `5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`; 39,099,447,584 bytes; SHA256 `6c6b816537abad90b250a0972b345466028d861ddfe316d5f0de31ca6440f781` |

Two RTX 3090s each have 24,576 MiB physical VRAM. Labels below were resolved from
original launch selectors and census UUIDs: prefix was physical index 0, suffix
index 1; the census listed suffix first. Do not use census vector order as stage order.

| 35B stage | Blocks | Single-card plan required / available (bytes) | Readiness model holdings | Context | Compute |
|---|---|---:|---:|---:|---:|
| Prefix | [0,21) | 40,029,885,251 / 22,759,656,653 | 19,357,865,344 | 101,564,736 | 17,489,522 |
| Suffix | [21,40) | 40,029,885,251 / 22,766,439,629 | 18,848,112,768 | 96,686,960 | 17,529,462 |

These are separate measurements: artifact file bytes are not resident model bytes;
model holdings exclude context/compute and are not physical VRAM or total readiness.
Auxiliary holdings are zero; no HostRam **model census** entry exists. HostRam
headroom entries do exist in the observation response. Original records support
both facts. No pooled ceiling or whole-model-first GPU0 import is claimed.

Small-model full numerical equivalence is separate from large-model deterministic
serving (`1, 2, 3, 4, 5, 6`, 16 tokens/request). There is no independent one-card
35B numerical control, performance claim, hardware-loss or mid-kernel-failure
qualification. The fault test injects a pre-suffix-submission refusal after real
prefix writes/transfer; physical writes are not rolled back. The admitted profile
is plain text, dense KV, one active request, prefill rows 1–2 and decode row 1;
retention/resume, media, draft/lookahead and unsupported placements refuse.

Decisive excerpts (paths/device identities omitted; results unchanged):

```text
PASS complete two-CUDA cut=16 bit_exact_features_logits_KV_recurrent_positions_feedback=true program_family=true normal_domain=true serving=false
PASS normal ExecutionOwner/Worker/Session paired generation, two-domain observation, retention/concurrency refusal without output backpressure, ordered stop/recovery, same-process unload/reload census and generation, shutdown; HTTP_API=false
internal engine failure: two-stage CUDA execution: qualification-injected suffix failure after completed prefix
```

## Reproduction (Bash, from repository root)

Install Rust 1.91.1, a CUDA-capable driver and the repository's development NVRTC
runtime with its matching builtins. Follow [inference development](../../README.md).
Set `SEISMIC_NVRTC_DIRECTORY` to that runtime directory when running outside an
installation layout; an absent NVRTC runtime is a setup failure. Supply cached
complete artifacts, a writable kernel cache, two exact Seismic `cuda:<uuid>`
selectors, their **Seismic CUDA catalog ordinals** (not assumed nvidia-smi order),
a split and a currently free loopback port. No command downloads models.
Run GPU campaigns serially on available devices; record driver/runtime identities.
Use a fresh evidence directory and only terminate processes started by these commands.

```bash
: "${SMALL_MODEL:?}" "${LARGE_MODEL:?}" "${KERNEL_CACHE:?}"
: "${PREFIX_CUDA:?}" "${SUFFIX_CUDA:?}" "${CUDA_ORDINALS:?}" "${PORT:?}"
export OUT="$(mktemp -d)"
export CARGO_TARGET_DIR="$OUT/normal-target"
export CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0
unset MAGNITUDE_PIPELINE_FAIL_AFTER_PREFIX_POSITION
export TOKEN_FIXTURE="$OUT/token-fixture.json"
git rev-parse HEAD 'HEAD^{tree}' > "$OUT/source.txt"
sha256sum "$SMALL_MODEL" "$LARGE_MODEL" > "$OUT/artifacts.sha256"
```

The original private fixture contains unrelated model-definition metadata. The
following is its exact consumed `vocabulary`, `stop_tokens`, `cases` projection
(all texts and token arrays unchanged), sufficient for both current qualifiers.
The original file's SHA256 recovered in this audit is
`51fd6ff1a490ee345b99a2acacf4e32867a272d8d0fe37c039364f31a062291f`;
there was no contemporaneous per-run fixture digest. The projection including its
trailing newline has SHA256 `ac0faa54ceb59fd32ffe37fa241ee587396ec761f9bd7f4f987f897e06281c78`.

```bash
cat > "$TOKEN_FIXTURE" <<'JSON'
{"vocabulary":248320,"stop_tokens":[248044,248046],"cases":[{"text":"<|im_start|>user\nWrite the numbers from one to twenty, separated by commas. Do not explain.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n","tokens":[248045,846,198,7734,279,4947,494,799,310,16570,11,18101,539,73982,13,3054,524,10033,13,248046,198,248045,74455,198,248068,271,248069,271],"teacher_tokens":[16,11,220,17,11,220,18,11,220,19,11,220,20]},{"text":"<|im_start|>user\nI am checking a local language model. Please write the integers from 1 to 30 in order, separated by commas. Start immediately with 1 and include every integer. Do not explain or summarize.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n","tokens":[248045,846,198,40,1044,12910,264,2136,3992,1558,13,5044,3165,279,24959,494,220,16,310,220,18,15,303,1906,11,18101,539,73982,13,4980,6849,440,220,16,321,2830,1396,7308,13,3054,524,10033,466,59987,13,248046,198,248045,74455,198,248068,271,248069,271],"teacher_tokens":[16,11,220,17,11,220,18,11,220,19,11,220,20]}]}
JSON
sha256sum "$TOKEN_FIXTURE" > "$OUT/fixture.sha256"
```

### Numerical and same-process lifecycle controls

```bash
export MAGNITUDE_PIPELINE_MODEL="$SMALL_MODEL"
export MAGNITUDE_PIPELINE_TOKEN_FIXTURE="$TOKEN_FIXTURE"
export MAGNITUDE_PIPELINE_CACHE="$KERNEL_CACHE"
export MAGNITUDE_PIPELINE_CUDA_ORDINALS="$CUDA_ORDINALS"
export MAGNITUDE_PIPELINE_CUTS=12,16
cargo +1.91.1 test --manifest-path inference/Cargo.toml -p magnitude-executor \
  --lib --features experimental-pipeline-cuda \
  complete_model_pipeline_matches_ordinary_control \
  -- --ignored --nocapture --test-threads=1 > "$OUT/numerical.txt" 2>&1
export MAGNITUDE_PIPELINE_CUT=16
cargo +1.91.1 test --manifest-path inference/Cargo.toml -p magnitude-engine \
  --lib --features experimental-pipeline-cuda \
  paired_worker_generates_and_observes_both_owned_domains \
  -- --ignored --nocapture --test-threads=1 > "$OUT/worker.txt" 2>&1
```

Both must exit 0, one test passed; numerical output must contain both cut PASS
lines and all zero-error comparison groups. Worker output must report ordered
stop/recovery, same-process unload/reload census/generation and shutdown. Structural
CI tests or an HTTP process restart cannot substitute for either control.

### Capacity and ordinary API

```bash
cargo +1.91.1 build --manifest-path inference/Cargo.toml -p magnitude-engine-cli \
  --features experimental-pipeline-cuda
export ENGINE_BIN="$CARGO_TARGET_DIR/debug/magnitude-engine"
sha256sum "$ENGINE_BIN" > "$OUT/normal-binary.sha256"
"$ENGINE_BIN" --help
# Verify the supplied port before each launch; release the probe before serving.
python3 - "$PORT" <<'PYPORT'
import socket,sys
with socket.socket() as s: s.bind(('127.0.0.1',int(sys.argv[1])))
PYPORT
for selector in "$PREFIX_CUDA" "$SUFFIX_CUDA"; do
  "$ENGINE_BIN" --model "$LARGE_MODEL" --no-projector --served-model pipeline-test \
    --host 127.0.0.1 --port "$PORT" --context-tokens 256 --prefill-tokens 2 \
    --method plain --lookahead off --kv-codec dense --cache-dir "$KERNEL_CACHE" \
    --device "$selector" > "$OUT/capacity-${selector#cuda:}.txt" 2>&1
  printf '%s\n' "$?" > "$OUT/capacity-${selector#cuda:}.exit"
done
```

Both large-artifact single-device launches should exit nonzero with typed
`LoadFailed(InsufficientMemory)` and required/available values, before resident
target imports or HTTP serving. Exit 0 is unexpected on this two-24-GiB setup;
a toolchain/setup failure is not capacity proof. Capacity values depend on the
actual device budget. Do not run this loop under `set -e` (nonzero exits are expected).

For paired serving set `MODEL=$LARGE_MODEL`, `SPLIT=21`, `MAX_TOKENS=16`.
For the small paired case use `MODEL=$SMALL_MODEL`, `SPLIT=16`, `MAX_TOKENS=8`;
also run its ordinary single-device control by replacing the last selection
argument with `--device "$PREFIX_CUDA"`. Compare both original prompts from the
fixture as fresh API requests, with `reasoning_effort:"none"`, seed 1 and
`cache_prompt:false`, as in the original campaign.

```bash
: "${MODEL:?}" "${SPLIT:?}" "${MAX_TOKENS:?}"
export RUN_OUT="$(mktemp -d "$OUT/api-XXXXXX")"
python3 - "$PORT" <<'PYPORT'
import socket,sys
with socket.socket() as s: s.bind(('127.0.0.1',int(sys.argv[1])))
PYPORT
MAGNITUDE_TRACE_PIPELINE=1 "$ENGINE_BIN" --model "$MODEL" --no-projector \
  --served-model pipeline-test --host 127.0.0.1 --port "$PORT" \
  --context-tokens 256 --prefill-tokens 2 --method plain --lookahead off \
  --kv-codec dense --cache-dir "$KERNEL_CACHE" \
  --experimental-pipeline "$PREFIX_CUDA,$SUFFIX_CUDA,$SPLIT" \
  > "$RUN_OUT/server.txt" 2>&1 &
server_pid=$!
# Wait for /health HTTP 200; allow preparation/tuning to finish.
until curl --fail --silent "http://127.0.0.1:$PORT/health" > "$RUN_OUT/health.json"; do
  kill -0 "$server_pid" || { wait "$server_pid"; exit 1; }
  sleep 1
done
curl --fail-with-body "http://127.0.0.1:$PORT/v1/memory" > "$RUN_OUT/memory.json"
for request in 1 2; do
  curl --fail-with-body "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"pipeline-test","messages":[{"role":"user","content":"Write the numbers from one to twenty, separated by commas. Do not explain."}],"temperature":0,"seed":1,"reasoning_effort":"none","max_tokens":'"$MAX_TOKENS"',"cache_prompt":false}' \
    > "$RUN_OUT/request-$request.json"
done
```

Expect HTTP 200, the requested completion count and `finish_reason:"length"` for
the recorded cases, identical fresh text and separate device-local model censuses.
Never ignore EOS to manufacture this result. Small-model API equality is a text
control, not a full-logit proof. Read `/v1/memory` by device identity, not list order.

For disconnect/recovery the following transcribes the original campaign's HTTP
steps using Python's standard library. It reads a live content delta, checks
concurrency refusal, closes that stream, and checks a fresh request. Save the
observations alongside the requests above. The worker qualifier independently
checks cancellation/drain and same-process reload.

```bash
python3 - "$PORT" "$RUN_OUT" "$MAX_TOKENS" <<'PYHTTP'
import json,sys,time,urllib.request,urllib.error,pathlib
base="http://127.0.0.1:"+sys.argv[1]+"/v1/chat/completions"
out=pathlib.Path(sys.argv[2]); max_tokens=int(sys.argv[3])
prompt="Write the numbers from one to twenty, separated by commas. Do not explain."
def body(text,**kw):
    return dict(model="pipeline-test",messages=[dict(role="user",content=text)],
                temperature=0,seed=1,reasoning_effort="none",cache_prompt=False,**kw)
def send(value):
    return urllib.request.urlopen(urllib.request.Request(base,data=json.dumps(value).encode(),
        headers={"Content-Type":"application/json","Connection":"close"}),timeout=30)
stream=send(body("Write the numbers from one to one hundred, separated by commas. Do not explain.",
                 max_tokens=200,stream=True))
try:
    while True:
        line=stream.readline().decode()
        assert line,"stream closed before content"
        if line.startswith("data: ") and line.strip()!="data: [DONE]":
            choices=json.loads(line[6:]).get("choices",[])
            if choices and choices[0].get("delta",{}).get("content"): break
    try:
        with send(body(prompt,max_tokens=max_tokens)) as response:
            raise AssertionError("concurrent request unexpectedly admitted")
    except urllib.error.HTTPError as error:
        (out/"concurrent-refusal.json").write_bytes(error.read())
finally:
    stream.close()
time.sleep(.5)
with send(body(prompt,max_tokens=max_tokens)) as response:
    recovery=json.load(response)
(out/"disconnect-recovery.json").write_text(json.dumps(recovery,indent=2)+"\n")
assert recovery["choices"][0]["message"]==json.loads((out/"request-1.json").read_text())["choices"][0]["message"]
PYHTTP
```

The fresh request uses the same caller-supplied token limit. Requesting `cache_prompt:true`
should refuse retention. Finish each server session with:

```bash
kill -TERM "$server_pid"
wait "$server_pid"
```

Expect exit 0, worker Shutdown and owned allocations released relative to the
opened-device/cache baseline. Persistent HTTP runtime overhead is not model storage.

### Partial-stage failure and fresh normal recovery

Build in a separate target directory to retain the normal binary:

```bash
CARGO_TARGET_DIR="$OUT/fault-target" cargo +1.91.1 build \
  --manifest-path inference/Cargo.toml -p magnitude-engine-cli \
  --features pipeline-fault-injection
sha256sum "$OUT/fault-target/debug/magnitude-engine" > "$OUT/fault-binary.sha256"
export ENGINE_BIN="$OUT/fault-target/debug/magnitude-engine"
export MAGNITUDE_PIPELINE_FAIL_AFTER_PREFIX_POSITION=2
```

Repeat the paired launch/request/shutdown block with the small model and split 16,
a freshly checked free port. The block creates a fresh `RUN_OUT` directory. The prompt must
visit source position 2 after warm-up. Expect trace
`prefix_completed=true suffix_submitted=false logical_positions=[2, 2]`, HTTP 500
with “qualification-injected suffix failure after completed prefix”, no `choices`,
then `/health` and `/v1/memory` HTTP 503 while HTTP remains alive. Check model
allocation release before graceful shutdown. The warm-up itself is not injected.

Then `unset MAGNITUDE_PIPELINE_FAIL_AFTER_PREFIX_POSITION`, restore
`ENGINE_BIN="$OUT/normal-target/debug/magnitude-engine"`, choose a free port and repeat small-model ordinary/paired API controls. Expect
fresh successful output and graceful shutdown. This is a separate normal binary
campaign, not a successful fault-build run.
