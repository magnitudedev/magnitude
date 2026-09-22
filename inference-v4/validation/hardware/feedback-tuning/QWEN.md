# Actual Qwen3.5-4B feedforward replay

This experiment uses the cached `mlx-community/Qwen3.5-4B-4bit` artifact,
revision `0e7ffd5c629ef7719d4cbc04069232580bfa9d9c`, and inputs captured from
a real MLX model forward. It replaces synthetic elementwise/stencil workloads
as the representative tuning test.

`qwen_capture.py` runs a 32-token prefill and one subsequent decode token,
capturing the input to every post-attention normalization. It constructs
independent NumPy references for the computation in
[`qwen_dense_suffix`](../../../engine/lib/dense_suffix.seismic): RMS normalization,
gate and up projection, SwiGLU, down projection, and residual addition.
All 32 layers use their own real weights and captured inputs. Matrix dimensions
are 9216×2560 for gate/up and 2560×9216 for down. Weights are affine Q4 with
group size 64 and BF16 coefficients; activations round to BF16 at each declared
publication boundary and residual addition returns F32.
The observer reads the artifact's separate code/scale/bias planes. Seismic's
importer instead repacks these into canonical 36-byte group packets; this
prototype does not benchmark that packed storage layout.

One candidate observation replays **all 32 FFNs**, including 96 projections
over 2,264,924,160 quantized weights (1,274,019,840 bytes including coefficients).
The FFNs have captured inputs, with explicit graph dependencies preserving
layer order; the intervening attention/recurrent mixers are not replayed.
This is a real model-component workload, not a complete
decoder/token-latency benchmark.

## Candidate space

`qwen_worker.py` generates native Metal through MLX's custom-kernel API. These
are experimental implementations of the production computation; they are not
emitted by Seismic. The same generic `tune.py` controller sees seven opaque axes:

| Axis | Values |
|---|---|
| Gate/up/SwiGLU fusion | Separate projections; dual projection; dual projection plus SwiGLU |
| Gate/up outputs per SIMD group | 1, 2, 4, 8 |
| Gate/up SIMD groups per threadgroup | 1, 2, 4, 8 |
| Gate/up reduction-loop unroll | 1, 2, 4 |
| Down outputs per SIMD group | 1, 2, 4, 8 |
| Down SIMD groups per threadgroup | 1, 2, 4, 8 |
| Down reduction-loop unroll | 1, 2, 4 |

There are 6,912 coordinates and at most 48 custom projection source artifacts.
Launch-group choices share source artifacts. A policy is shared across layers
with the same geometry; it does not select an independent policy for every
layer. All coordinates preserve the same per-lane dot-product order; they
still reassociate the reduction relative to the ordered Seismic reference.

## Measurement and numerical contract

The objective is wall time to construct, submit, and complete the whole replay,
including Python/MLX overhead. It is not the GPU-only endpoint used in the
synthetic Swift experiment. Every timed sample creates fresh output graphs;
weights and captured inputs remain resident. All intermediates are retained
and evaluated for both custom and MLX paths.

Every observation checks normalized inputs, gate, up, product, down, and final
residual outputs against the independent references, for all layers. Admission
requires finite values, normalized RMS error ≤0.005 and maximum absolute error
divided by reference RMS ≤0.05 **for every stage of every layer**. These fixed
experimental thresholds were declared before testing candidates. They do not
replace Seismic's numerical admissibility analysis or qualify arbitrary inputs.

The controller's clock includes worker startup, mapping/loading the existing
weights and references, artifact construction, first execution, all warm and
timed observations, output checks, search, and finalist validation. The model
capture/reference generator is separately timed offline setup. No weights are
downloaded or copied into the experiment directory. Existing system/compiler
caches are not cleared.

MLX compiles custom kernels lazily. `first_invocation_ms` includes JIT if needed,
graph construction, and first complete execution. For compatibility with the
original controller, `compile_ms` contains this first-invocation duration when
new custom artifacts appear; **it is not isolated compiler time**. The raw
record carries an explicit `compile_metric` explaining this distinction.

## Reproduction

Use an existing environment with MLX, MLX-LM, and NumPy. On `m4-pro-01` the
environment is `/Users/ec2-user/magnitude-mlx/inference-v2/.venv/bin/python`.
Run from the experiment scratch directory so the default `qwen-data` path
resolves, or set `QWEN_FEEDBACK_DATA` explicitly.

```sh
python qwen_capture.py --model /path/to/cached/model/snapshot --output qwen-data
python qwen_suite.py --mode campaigns --regime decode --seeds 3 --output qwen-results
python qwen_suite.py --mode reference --regime decode --output qwen-results
python qwen_suite.py --mode confirm --regime decode --output qwen-results
python qwen_controls.py --summary qwen-results/decode-summary.json --output qwen-results/controls.json
```

GPU experiments run sequentially. `reference` screens all 6,912 coordinates
with one warm sample after the first invocation. `confirm` compares its top 24,
the online selections, and matched-precision MLX in eight randomized blocks,
with five samples per observation. These offline assessments are outside the
one-minute tuning budgets. A finite noisy screen is not a proof of the exact
optimum.

The matched MLX control uses F32 activations and F32 copies of the coefficient
planes inside `quantized_matmul`, then publishes BF16. Its additional
coefficient bytes and casts are part of that implementation's cost. Ordinary
BF16 MLX is measured separately and checked against the same reference; a
failure is recorded rather than weakening the experimental contract.

`qwen_controls.py` also replays the selected decode policy on the actual
32-token prefill activations. This is a transfer test, not a prefill tuning
campaign. The current custom family uses SIMD matvec-style reductions and
does not contain a tiled matrix-instruction implementation.

## Five-minute convergence run

The same controller accepts `--budget 300`; it reserves ten seconds for final
validation and returns when that validation completes. The recorded run uses
three complete timed replays per search observation and checks every stage.

```sh
python tune.py --worker qwen-results/qwen-observer --fixture decode \
  --algorithm evolution --seed 0 --budget 300 --trials 3 --target-ms 2 \
  --output qwen-serial-results/decode-evolution-300s.jsonl
python qwen_convergence.py --mode validate \
  --input qwen-serial-results/decode-evolution-300s.jsonl \
  --output qwen-serial-results/convergence.json
python qwen_convergence.py --mode plot \
  --input qwen-serial-results/convergence.json \
  --output qwen-serial-results/convergence.png
```

The plot's main curve is retrospective: at each time checkpoint, take the six
best candidates observed by then and independently compare those nominees in
ten randomized blocks. Their best remeasured mean estimates what finalizing
the shortlist at that checkpoint could have returned. It is not an online
validated incumbent maintained during the campaign. All offline checks are
excluded from the tuning clock. The raw running-minimum curve is also shown,
so selection noise is visible rather than presented as improvement.
