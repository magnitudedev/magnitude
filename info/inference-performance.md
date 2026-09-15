# Model performance workflow

The agent runs measurements. The user opens a model overview of evidence already
published. Benchmark recipes, production formula traces, boundary fixtures and
reference-engine observations share one history store.

The durable contract is [formula execution and measurement](../design/inference/formula-execution.md).

## Open the model overview

From `inference-v3`, using its installed environment:

```bash
.venv/bin/python -m ops.lab browse runs/model-performance.sqlite
```

`magnitude-performance` is the equivalent installed command. Pick a model; its
recorded engines, hardware conditions and history are underneath it. Select an
execution for metrics, correctness, formula hierarchy and provenance. `r` refreshes
published evidence. Browsing opens SQLite read-only and never starts GPU work.

A stable model identity groups implementations, artifacts and machines. It does
not make different quantization formats, prompt/state realizations or hardware
comparable. The view keeps these conditions separate. It cannot infer performance
at unmeasured contexts or on another device.

## Publish normal benchmark runs

The existing benchmark command takes the shared store and stable model identity:

```bash
.venv/bin/python -m session_bench run \
  --target omlx=/absolute/path/to/mlx-model \
  --suite single --context 16k --repeat 2 \
  --model-identity qwen35-35b-a3b \
  --evidence-store runs/model-performance.sqlite
```

Use the same identity for that model's other engines/artifacts. The request recipe
is the actual benchmark `Plan` and request; the performance layer does not render
another prompt. Client latency, source prefill/decode timers and their exact token
counts are retained with their original boundaries. HTTP equality does not prove
identical tokenization or starting state. Serving validation is recorded separately
from independent numerical correctness.

The GGUF V3 entrypoint accepts the same publication flags:

```bash
.venv/bin/python -m performance.serving \
  --target /absolute/path/to/model.gguf --suite single --context 16384 --repeat 2 \
  --model-identity qwen35-35b-a3b \
  --evidence-store runs/model-performance.sqlite
```

## Discover and measure a scope

A factory returns an existing `Configuration`, or yields one from a context manager
when its artifact source needs to stay open. It supplies the production fixture,
normal compile options and device factory. No competing model or kernel registry
is introduced.

For example, `examples.qwen_artifact_lab:configuration` wraps the existing GGUF
feed-forward fixture. This fixture has **synthetic hidden inputs**, not captured
model activations. Save its arguments as `arguments.json`:

```json
{"model":"/absolute/path/to/model.gguf","rows":2048,"layer":0,"maximum_bytes":4294967296}
```

Discover formula occurrences and semantic hashes without compilation:

```bash
.venv/bin/python -m ops.lab describe \
  --factory examples.qwen_artifact_lab:configuration --arguments arguments.json
```

A `MeasurementRequest` JSON supplies:

- `context`: stable model identity/label, the existing benchmark recipe or explicitly
  synthetic recipe, artifact identity, numerical contract and engine identity.
- `arguments`: the same factory arguments used in discovery.
- `occurrences` and `semantics`: selected occurrence numbers and their returned
  hashes. A changed trace cannot silently redirect a request to another scope.
- `protocol`, if overriding the factory's protocol: sample/warmup counts, numerical
  tolerances, native capture capacity and input preparation. Omission preserves the
  factory's tolerances; explicit changes define different conditions.
- `repeats`: repeated measurements in one resident worker.
- `save_boundaries`: optional directory for portable boundary snapshots.
- `retained_boundaries`: optional mapping from occurrence to a saved snapshot path.
- `characterize`: explicit resource calibration/load request, false by default.

The typed request contract is `ops.lab.execution.MeasurementRequest`. Actual hardware,
host, compiled implementation and boundary realization replace the request's initial
context values at execution. Artifact/model identity and recipe remain the caller's
responsibility: use the benchmark/artifact provenance, never an invented alias for
an incompatible model.

```bash
.venv/bin/python -m ops.lab run request.json \
  --factory examples.qwen_artifact_lab:configuration \
  --store runs/model-performance.sqlite --bundle runs/scope-evidence.json
```

One worker remains alive for the request. Phase/job records separate preparation,
compilation, checks, conditioning and samples. Results survive failure; requested
bundle export also runs after a failed measurement if the store exists.

For iteration across edits, use `configuration.open()` as a Python context manager.
Keep it open and call `lab.measure(target).result.result()` after each edit. The
worker refreshes authored operation dependencies and reuses independent boundaries.
Closing the process ends device residency. Disk snapshots avoid upstream computation
in a new process, but still require validation, upload and affected compilation.

## Capture actual production inputs and contributions

`engine.models.qwen35.inspection.inspect_forwards` instruments the existing runtime.
An `observed(invocation, observation)` callback receives the natural forward boundary
at retirement. `publish_forward(store, context, invocation, observation)` publishes
that observation and the production formula hierarchy. The caller drives ordinary
engine execution and retires each forward as usual.

A separate `captured(invocation)` callback can call `invocation.fixture(decode_weight)`
before submission to snapshot actual inputs and state. `decode_weight` provides the
independent artifact reference for immutable weights. Keep the artifact owner alive
until all fixture users close. Do capture and timing in separate runs; reading back
tensors must not contaminate the timing baseline.

For that full production fixture, choose `protocol.inputs = "production"`. The worker
executes upstream operations once and retains the chosen boundary. Its preparation
may expose formerly fused intermediates and is never reported as in-parent timing.
Each selected invocation starts with fresh mutable state. A selected-kernel edit
reuses independent upstream work; changing captured dependencies invalidates it.
An explicitly restored snapshot remains a fixed historical input, with provenance.

Snapshots validate contents, tensor schemas and immutable bindings. Compatible shared
state is preserved; unsupported alias views fail explicitly. They never deserialize
executable Python. Large model weights are referenced separately rather than copied
into every snapshot.

Native timing records capture-relative endpoints, dispatch identity, formula origins
and an exclusive owner when compiled order/count/symbol checks support the mapping.
Unsupported or mismatched attribution remains unavailable. GPU busy time is an
interval union; overlapping kernel durations and isolated child times are not added
into parent latency. Compiler source artifacts accompany isolated measurements when
the compiler exposes them; they are not a substitute for unavailable machine ISA.

## Compare a concrete algorithm hypothesis

Prepare both candidates on the **same retained boundary and live device**, outside
timing. The agent explicitly prepares each authored implementation using normal
production compilation; `compare_prepared` does not select kernels:

```python
from ops.lab.comparison import compare_prepared
from ops.lab.evidence import PairedComparison

runs = compare_prepared(
    {"baseline": baseline, "candidate": candidate},
    context=context, store=store, protocol=protocol, blocks=6,
)
paired = PairedComparison.from_runs(runs[0], runs[1], "elapsed", "complete-operation")
print(paired.deltas)  # Baseline minus candidate, seconds, one value per pair.
```

The caller owns both preparations and closes them afterward. Blocks alternate order,
reset state per invocation, record pair identities and retain all samples. Compare
paired deltas and variation before measuring the containing layer/forward. Parent
impact must be measured in the actual composition; an isolated saving is a hypothesis
about end-to-end impact, not an additive prediction.

`measure_invalid=true` explicitly permits timing numerical mismatches. These results
remain failed for numerical qualification and never enter best-correct history. Unsafe
state behavior or execution failure stops the comparison and preserves incomplete
evidence. Missing useful-work analysis or calibration does not prevent valid timing.

Ceilings retain their assumptions. The empirical formula resource model uses measured
resource rates and ideal overlap; it is not a proven absolute hardware ceiling.
Unsupported latency bounds remain unavailable. A memory footprint bound is not a
latency bound, and efficiency above a probe reference challenges that reference.

## Bring in reference evidence

Native benchmark adapters publish enclosing request observations directly. For a
narrower independently instrumented region, publish a `RunEvidence` with an
`ExternalMapping`: source region, formula scopes, mapping revision, supporting
evidence and relationship (`equivalent`, `corresponding`, or `enclosing`). Declared
contract differences are mandatory for `corresponding`. Keep fused regions combined;
do not apportion their timer across formulas by guessed percentages.

The model view displays mapped reference regions at matching formula contracts,
including hardware and differences. A mapping does not manufacture a same-contract
A/B comparison. Engine-specific instrumentation still has to establish that mapping;
opaque HTTP timing cannot automatically expose internal operations.

## Use the two M4 Pros

Use ordinary SSH and file transfer. Set up each checkout/environment and model paths
explicitly. Send the small request and any boundary snapshots, run the same command
there, then transfer and import the evidence bundle. For example, with the checkout
at the shown path on the remote host:

```bash
scp request.json m4-pro-01:/tmp/scope-request.json
ssh m4-pro-01 'cd ~/magnitude-mlx/inference-v3 && .venv/bin/python -m ops.lab run /tmp/scope-request.json --factory examples.qwen_artifact_lab:configuration --store runs/scope.sqlite --bundle runs/scope-evidence.json'
scp m4-pro-01:~/magnitude-mlx/inference-v3/runs/scope-evidence.json /tmp/m4-pro-01-evidence.json
.venv/bin/python -m ops.lab import runs/model-performance.sqlite /tmp/m4-pro-01-evidence.json
```

Substitute `m4-pro-02` for an independent investigation. Adapt request artifact and
snapshot paths to the destination. For ordinary session runs, export afterward with
`python -m ops.lab export STORE BUNDLE`. Import checks integrity, references and
immutable conflicts before an atomic merge. Reimporting the same bundle is harmless.

Run one investigation per machine; keep each A/B pair on its own host. A faster
candidate on one Pro is evidence for a hypothesis on the Max, not a measured Max
speedup. No remote scheduler or coordination service is needed.
