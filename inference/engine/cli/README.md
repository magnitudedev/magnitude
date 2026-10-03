# Standalone engine: explicit experimental pipeline

See the [review guide, audited qualification and self-contained reproduction](PIPELINE-REVIEW.md).

The default build and invocation remain single-device. To explicitly select the
experimental two-CUDA executor, build with:

```sh
cargo build --manifest-path inference/Cargo.toml -p magnitude-engine-cli \
  --features experimental-pipeline-cuda
```

Supply the artifact, two **exact** Seismic CUDA selectors (`cuda:<uuid>`), and a
nonempty decoder boundary yourself. No device, cut or model-fit policy selects
this mode. It is mutually exclusive with `--device`.

```sh
"$ENGINE_BIN" --model "$MODEL" --no-projector --served-model pipeline-test \
  --port "$PORT" --context-tokens 256 --method plain --kv-codec dense \
  --lookahead off --prefill-tokens 2 --cache-dir "$KERNEL_CACHE" \
  --experimental-pipeline "$PREFIX_CUDA,$SUFFIX_CUDA,$SPLIT"

curl --fail-with-body "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"pipeline-test","messages":[{"role":"user","content":"Write the numbers from one to twenty, separated by commas. Do not explain."}],"temperature":0,"max_tokens":8,"cache_prompt":false}'
```

Set `PORT` to an available local port and `ENGINE_BIN` to the repository-built binary,
not an installed release. The experimental
profile is native plain text, dense KV, no projector/draft/lookahead, one active
request, one/two-row prefill and one-row decode. Prompt retention, resume, media
and overlapping requests are refused. Disable reasoning through the model's
supported request control when comparing deterministic text outputs; do not
ignore EOS. `/v1/memory` reports each GPU separately. `MAGNITUDE_TRACE_PIPELINE=1`
adds per-step device/range and PCIe activation timing diagnostics, not benchmark
claims. Weights and persistent state never use CPU offload or pooled VRAM.

Preparation/tuning is independent on each named device. The boundary transfers
bounded activations through host staging; NVLink and peer access are not required.
Unsupported placement, structure or capacity fails rather than falling back.
SIGINT/SIGTERM gracefully drains requests and closes the execution worker.

For manual partial-step failure qualification only, a separate development build
with `--features pipeline-fault-injection` enables
`MAGNITUDE_PIPELINE_FAIL_AFTER_PREFIX_POSITION=N`. It injects a suffix refusal
**after** completed prefix GPU writes and activation transfer at that source
position. Choose a position after warm-up that the prompt actually visits (for
example 2 for a multi-row prompt with two-row prefill). The ordinary experimental
build does not include this seam. Expect an admitted request failure and model
unload, never a successful token for that incomplete step; completed writes are
not rolled back. Inspect allocation recovery against the opened-device/cache
baseline, then reload in a non-injected run to test recovery.

Manual qualification requires cached artifacts and a token-ID fixture containing
`vocabulary`, `stop_tokens`, and two `cases` with `text`, `tokens` and
`teacher_tokens`. Use the original qualified fixture unchanged. Set paths to the
small Qwen model, that fixture, and a writable kernel cache; no test downloads
anything. Set `SEISMIC_NVRTC_DIRECTORY` to the development NVRTC runtime directory
if the binary layout does not supply it. From the repository root:

```sh
export MAGNITUDE_PIPELINE_MODEL="$SMALL_MODEL"
export MAGNITUDE_PIPELINE_TOKEN_FIXTURE="$TOKEN_FIXTURE"
export MAGNITUDE_PIPELINE_CACHE="$KERNEL_CACHE"
export MAGNITUDE_PIPELINE_CUDA_ORDINALS=0,1
export MAGNITUDE_PIPELINE_CUTS=12,16
cargo +1.91.1 test --manifest-path inference/Cargo.toml -p magnitude-executor \
  --lib --features experimental-pipeline-cuda \
  complete_model_pipeline_matches_ordinary_control \
  -- --ignored --nocapture --test-threads=1

export MAGNITUDE_PIPELINE_CUT=16
cargo +1.91.1 test --manifest-path inference/Cargo.toml -p magnitude-engine \
  --lib --features experimental-pipeline-cuda \
  paired_worker_generates_and_observes_both_owned_domains \
  -- --ignored --nocapture --test-threads=1
```

The numerical test compares the candidate's ordinary domain path with an
independent unsliced control, covering full logits/features, KV/recurrent state,
accepted positions and feedback, including 32 generated candidate tokens over
both cuts. The engine test covers normal worker generation, live controls and
same-process unload/reload. Run GPU campaigns serially. Public API qualification
uses the serving command above: first both single-device invocations with the
large artifact (typed capacity refusal before imports), then explicit cut 21 and
two fresh 16-token requests. Check both per-device memory reports, cancellation
and fresh recovery, concurrent/retention refusal, drain and allocation release.
Use the separate fault build for partial-stage discard; this does not qualify
hardware loss or mid-kernel failure. See the [design contract](../../../design/inference/engine/pipeline-execution.md)
for the placement boundary and ownership guarantees.

The qualified workstation cases are complete Qwen3.5-4B UD-Q8_K_XL (independent
ordinary control, cuts 12 and 16) and Qwen3.6-35B-A3B UD-Q8_K_XL (explicit cut 21,
public serving). These are reproduction cases, not runtime eligibility rules or
placement policy. A larger-than-one-card model has no one-card numerical control;
its evidence is typed per-card capacity refusal and actual paired serving.
