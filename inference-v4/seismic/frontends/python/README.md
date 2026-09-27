# Seismic Python frontend

Calls authored `.seismic` functions through the public Rust `seismic` API.
Python handles data, experiments, and reporting. There is no Python tracing,
autodiff, subprocess compiler, or inference-engine dependency.

## Build and install

From `inference-v4`, using the repository's Rust toolchain:

```sh
uv venv --python 3.12 .venv-python
source .venv-python/bin/activate
uv pip install maturin numpy ml_dtypes pytest
maturin develop --release --manifest-path seismic/frontends/python/Cargo.toml
pytest seismic/frontends/python/tests
python seismic/frontends/python/examples/basics.py --device cpu
```

The distribution is `magnitude-seismic`; the import is `seismic`. Python 3.10+
is required; the exercised configuration is CPython 3.12 on macOS arm64.
Normal workspace default builds do not build this extension.
To distribute a wheel, use `maturin build --release --manifest-path ...` and
install its wheel in a clean environment. A release build matters for benchmarks.
Backend availability and preparation errors are surfaced without fallback.

```python
import numpy as np
import seismic as sm

module = sm.load("kernels.seismic", std=False)
device = sm.device("cpu")
kernel = module.affine.prepare(
    device=device,
    evaluation=sm.Feedback(search_seconds=0),
)
x = sm.asarray(np.arange(8, dtype=np.float32), device=device)
y = kernel(x, scale=2.0)
print(y.numpy())
```

`prepare` defaults to analytical evaluation and exact precision. Explicit
zero-budget feedback is useful when a backend has no analytical profile; it
publishes the existing core's initial legal implementation. It does not imply
that a meaningful timing search was performed.

## Surface

| Operation | API |
|---|---|
| Loading | `load`, `load_source`, `Module.save`, `module[name]`, `module.functions` |
| Devices | `devices`, `device`, capabilities, memory usage/limit |
| Host transfers | `asarray`, `zeros`, `from_bytes`, `Tensor.numpy`, `item`, `tobytes` |
| Resident buffers | `copy`, `copy_from`, `write_bytes`, reshape, leading slices |
| Execution | `Function.prepare`, `prepare_native`, positional/keyword `Kernel` calls |
| Ownership | `move(tensor)`; admission commits, live views block moves |
| Feedback | `Function.scope`, `Interval`, `start_feedback`, `continue_for`, `close` |
| Grouped calls | `Workflow.enqueue`, `Workflow.run`, pending tensor/scalar values |
| Testing | `testing.assert_close`, `testing.check(...).assert_passed()` |
| Timing | `benchmark(kernel, args=..., setup=..., warmup=3, repeat=20)` |

Dense elements are `float32`, `float16`, `bfloat16` (`ml_dtypes`), `int32`,
`uint32`, and `bool_`. Host imports copy; readback returns an independent array.
NumPy dtype is preserved unless an explicit conversion is requested. Python
sequences infer float32/int32/bool. Encoded representations use `element(name)`
and canonical bytes, with layout checked by Seismic. Resident tensor arguments
never implicitly upload or cast. `asarray(existing, copy=False)` returns the
same object when device and dtype match.

Calls are synchronous. The extension releases the GIL during preparation,
execution, and transfers. Operations sharing an allocation serialize through
its access gate; unrelated allocations have separate gates. A moved tensor's
Python aliases all become invalid. A failed admission leaves moves uncommitted.
Once admitted, consumed tensors stay consumed even if execution later fails.

Feedback sessions publish immutable kernel snapshots; closing a session does
not invalidate them. A scope guides search without narrowing the callable
shape domain. `inspect.signature(kernel)` describes source argument binding.
Use `module[name]` when an entry collides with a Python attribute.

Workflow drafts are single-use and native kernels cannot be enqueued. Pending
leading slices currently require explicit nonnegative start/stop, step one,
and one slice operation. Final outputs must be complete pending leaves, not
slices. The runtime reports scalar edges that require a host boundary.

The bounded observer currently supports ordinary kernels with independent,
dense input allocations. Aliased inputs, encoded inputs, direct-native kernels,
and source executions without a comparable complete oracle outcome report
`unsupported`. Memory/work exhaustion reports `resource_limit`. Neither status
passes `assert_passed`. Checks use private inputs, including private owned moves,
and compare final input state as well as return values.

Benchmark timing covers the Python call through native completion and result
construction. It excludes setup, preparation, readback, and result disposal.
Mutable/owned arguments require `setup()` returning `(args, kwargs)` every time.

## Examples

- `examples/basics.py`: load, prepare, call, compare, check, benchmark.
- `examples/mnist.py --smoke --batch-size 3`: deterministic small MLP reference
  checks, explicit gradients, one SGD update, and a complete training workflow.
  Add `--full-size` to exercise 784 → 128 → 10 with synthetic inputs.
- `examples/mnist.py --data mnist.npz --weights weights.npz --device metal`:
  a resident-weight 784 → 128 → 10 model, partial final minibatches, intermediate
  and logits comparisons, and reported test accuracy with weight SHA256.
- Add `--epochs 1` for explicit Seismic SGD and a reported mean-loss curve.
- `--route composed` explicitly selects the source-composed forward/step. The
  current compiler fails preparation when publishing nested-call tensor results.
  The default `--route workflow` does not attempt or fall back from composition.

Known core regressions, including CPU top-level sequential checked access, are
recorded in `bugs/26-09-22/seismic-python-workload-blockers.md` at the repository root.

`prepare_mnist.py` converts locally acquired MNIST IDX/IDX.gz files and optionally
creates reference-pretrained weights. It requires all four dataset paths:

```sh
python examples/prepare_mnist.py \
  --train-images train-images-idx3-ubyte.gz --train-labels train-labels-idx1-ubyte.gz \
  --test-images t10k-images-idx3-ubyte.gz --test-labels t10k-labels-idx1-ubyte.gz \
  --output mnist.npz --weights weights.npz --epochs 5 --seed 7
```

Run that command from this frontend directory. No imports or examples silently
download data. Real accuracy requires real MNIST data and identified weights;
the synthetic smoke result is not a claim about MNIST accuracy. NumPy comparisons
use explicit tolerances because independently written floating computations can
have different evaluation order even under exact source semantics.

## PyTorch epoch comparison

`examples/compare_mnist.py` runs the same 784→128→10 float32 MLP with
cross-entropy and plain SGD using Seismic workflows and eager PyTorch autograd.
It checks forward values, loss, every parameter gradient, and one weight update
against NumPy, warms both batch geometries, resets weights, then trains both
implementations on identical seeded epoch orders. PyTorch is an optional demo
dependency; it is not imported by the frontend.

From `inference-v4`, with the development virtual environment active:

```sh
uv pip install torch pyarrow pillow
maturin develop --release --manifest-path seismic/frontends/python/Cargo.toml
hf download ylecun/mnist --repo-type dataset \
  --revision 77f3279092a1c1579b2250db8eafed0ad422088c \
  --include '*.parquet' --local-dir /tmp/mnist-hf
python seismic/frontends/python/examples/prepare_mnist.py \
  --hf-parquet /tmp/mnist-hf --output /tmp/mnist.npz
python seismic/frontends/python/examples/compare_mnist.py \
  --data /tmp/mnist.npz --device metal --epochs 3 --output /tmp/comparison.json
```

The converter also accepts the four local IDX files documented above. Downloads
are explicit; neither importing Seismic nor running the comparison downloads data.
Use `--device cpu` for CPU versus CPU, or `--limit 129 --epochs 1` for a quick
validation including a partial minibatch. Subset epochs are labeled by their
actual example count and must not be presented as full MNIST epochs.

Timing includes Python minibatch slicing, workflow construction or autograd,
forward/backward, SGD, allocation, and completion. Data shuffling, uploads,
preparation, correctness checks, warmup, and final test evaluation are outside the
epoch timer. Upload and setup costs are reported separately. Seismic calls are
synchronous; PyTorch is synchronized at epoch boundaries using
[`torch.mps.synchronize`](https://docs.pytorch.org/docs/stable/generated/torch.mps.synchronize.html).
There is no artificial per-step synchronization on PyTorch and no loss readback
inside the timing loop. Execution order alternates by epoch. Output includes each
epoch's seconds and examples/second, medians, test accuracy, dataset/source hashes,
and software/device metadata.

This compares the available implementations: Seismic's exact, explicitly authored
scalar reductions and gradients against PyTorch's optimized dense operations and
autograd. It is not a comparison of equally optimized matrix kernels, nor of
`torch.compile`. The Seismic path uses the explicit workflow because source-composed
result publication currently has the compiler limitation described above. Always
build the native extension with `--release` before publishing timings; the script
does not infer the build profile of an installed extension.
