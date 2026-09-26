import gc
import inspect
import os
from concurrent.futures import ThreadPoolExecutor

import ml_dtypes
import numpy as np
import pytest

import seismic as sm

SOURCE = """\
fn affine[N](x: &tensor[N] f32, scale: f32) -> tensor[N] f32 where N >= 0:
    let mut y = zeros_like(x)
    parallel for i in 0..N:
        y[i] = x[i] * scale + 1.0
    return y

fn write[N](x: &tensor[N] f32, y: &mut tensor[N] f32):
    parallel for i in 0..N:
        y[i] = x[i] + 1.0

fn consume[N](x: tensor[N] f32) -> tensor[N] f32:
    let mut y = x
    parallel for i in 0..N:
        y[i] = y[i] + 1.0
    return y

fn scalars(a: f32, b: i32, c: u32, flag: bool) -> (f32, i32, u32, bool):
    return a, b, c, flag
"""


@pytest.fixture(scope="session")
def dev():
    return sm.device(os.environ.get("SEISMIC_TEST_DEVICE", "cpu"))


@pytest.fixture(scope="session")
def module():
    return sm.load_source(SOURCE, name="test.seismic", std=False)


@pytest.fixture(scope="session")
def kernels(module, dev):
    return {
        name: fn.prepare(device=dev, evaluation=sm.Feedback(0))
        for name, fn in module.functions.items()
    }


def test_source_and_bundle(tmp_path, module):
    p = tmp_path / "program.seismic"
    p.write_text(SOURCE)
    a = sm.load(p, std=False)
    b = sm.load(tmp_path, std=False)
    assert a.identity == b.identity
    bundle = tmp_path / "snapshot.seismicbundle"
    a.save(bundle)
    assert sm.load(bundle).identity == a.identity
    p.write_text(SOURCE.replace("+ 1.0", "+ 2.0"))
    assert sm.load(p, std=False).identity != a.identity
    assert sm.load(bundle).identity == a.identity
    data = bytearray(bundle.read_bytes())
    data[-1] ^= 1
    bundle.write_bytes(data)
    with pytest.raises(sm.BundleError):
        sm.load(bundle)
    assert list(inspect.signature(module.affine).parameters) == ["x", "scale"]
    with pytest.raises(sm.SourceError):
        sm.load_source("fn invalid(:\n", std=False)


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float16, ml_dtypes.bfloat16, np.int32, np.uint32, np.bool_]
)
def test_dense_roundtrip(dtype, dev):
    a = np.array([[0, 1], [1, 0]], dtype=dtype)
    x = sm.asarray(a, device=dev)
    assert x.tobytes() == a.tobytes()
    assert x.numpy().dtype == a.dtype
    np.testing.assert_array_equal(x.numpy(), a)
    assert sm.asarray(x, copy=False) is x
    y = sm.asarray(x, copy=True)
    assert y is not x
    host = x.numpy()
    host.fill(0)
    assert x.tobytes() == a.tobytes()


def test_bits_layout_and_casts(dev):
    words = np.array([0x80000000, 0x7FC01234, 0x3F800000], dtype=np.uint32)
    x = sm.asarray(words.view(np.float32), device=dev)
    assert x.tobytes() == words.tobytes()
    a = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
    np.testing.assert_array_equal(sm.asarray(a, device=dev).numpy(), a)
    np.testing.assert_array_equal(sm.asarray(a.astype(">f4"), device=dev).numpy(), a)
    with pytest.raises(TypeError):
        sm.asarray(np.ones(2), device=dev)
    with pytest.raises(ValueError):
        sm.asarray(a, device=dev, copy=False)
    with pytest.raises(OverflowError):
        sm.asarray([2**40], device=dev)
    with pytest.raises(OverflowError):
        sm.asarray([1e30], dtype=sm.float16, device=dev)


def test_views_and_overlap(dev):
    x = sm.asarray(np.arange(6, dtype=np.float32), device=dev)
    x[1:5].copy_from(x[:4])
    np.testing.assert_array_equal(x.numpy(), [0, 0, 1, 2, 3, 5])
    view = x[2:4]
    del x
    gc.collect()
    np.testing.assert_array_equal(view.numpy(), [1, 2])
    assert view.reshape((1, -1)).shape == (1, 2)
    with pytest.raises(sm.TensorError):
        view.reshape((3,))
    with pytest.raises(TypeError):
        view[::2]


def test_call_binding(kernels, dev):
    x = sm.asarray(np.arange(4, dtype=np.float32), device=dev)
    k = kernels["affine"]
    np.testing.assert_array_equal(k(x, 2).numpy(), np.arange(4) * 2 + 1)
    np.testing.assert_array_equal(k(scale=2, x=x).numpy(), k(x, 2).numpy())
    for invoke in [
        lambda: k(x),
        lambda: k(x, 2, scale=3),
        lambda: k(x, scale=2, unknown=1),
    ]:
        with pytest.raises(TypeError):
            invoke()
    s = kernels["scalars"](1.5, -7, 2**32 - 1, True)
    assert s == (np.float32(1.5), np.int32(-7), np.uint32(2**32 - 1), np.bool_(True))
    with pytest.raises(OverflowError):
        kernels["scalars"](0, 2**32, 0, False)


def test_moves_and_rollback(kernels, dev):
    x = sm.asarray([1.0, 2.0], device=dev)
    intent = sm.move(x)
    assert x.shape == (2,)
    with pytest.raises(TypeError):
        kernels["consume"](x)
    view = x[:1]
    with pytest.raises(sm.InvocationError):
        kernels["consume"](intent)
    np.testing.assert_array_equal(x.numpy(), [1, 2])
    del view
    gc.collect()
    y = kernels["consume"](intent)
    np.testing.assert_array_equal(y.numpy(), [2, 3])
    with pytest.raises(sm.TensorError):
        x.numpy()
    z = kernels["consume"](sm.move(y))
    np.testing.assert_array_equal(z.numpy(), [3, 4])
    a = sm.asarray([1.0, 2.0], device=dev)
    with pytest.raises(sm.InvocationError):
        kernels["write"](a, a)
    np.testing.assert_array_equal(a.numpy(), [1, 2])


def test_workflow(kernels, dev):
    x = sm.asarray([1.0, 2.0], device=dev)
    w = sm.Workflow(device=dev)
    a = w.enqueue(kernels["affine"], x, 2)
    b = w.enqueue(kernels["consume"], sm.move(a))
    with pytest.raises(sm.WorkflowError):
        w.enqueue(kernels["affine"], a, 2)
    y, z = w.run((b, b))
    assert y is z
    np.testing.assert_array_equal(y.numpy(), [4, 6])
    with pytest.raises(sm.WorkflowError):
        w.run(b)
    w = sm.Workflow(device=dev)
    target = sm.zeros((2,), dtype=sm.float32, device=dev)
    w.enqueue(kernels["write"], x, target)
    assert w.run() is None
    np.testing.assert_array_equal(target.numpy(), [2, 3])


def test_feedback_session(module, dev):
    scope = module.affine.scope(
        dimensions={"N": sm.Interval(1, 8)}, scalars={"scale": 2.0}
    )
    with module.affine.start_feedback(
        device=dev, options=sm.Feedback(0, scope=scope)
    ) as session:
        first = session.kernel
        second = session.continue_for(0)
        assert session.kernel is second
        assert "elapsed" in session.report
    with pytest.raises(ValueError):
        session.continue_for(0)
    x = sm.asarray([2.0], device=dev)
    np.testing.assert_array_equal(first(x, 3).numpy(), second(x, 3).numpy())
    with pytest.raises(ValueError):
        module.affine.scope(dimensions={"bad": 1})
    with pytest.raises(ValueError):
        module.affine.scope(dimensions={"N": sm.Interval(9, 1)})


def test_observer_and_benchmark(kernels, dev):
    x = sm.asarray([1.0, 2.0], device=dev)
    target = sm.zeros((2,), dtype=sm.float32, device=dev)
    report = sm.testing.check(kernels["write"], args=(x, target))
    report.assert_passed()
    np.testing.assert_array_equal(target.numpy(), [0, 0])
    report = sm.testing.check(kernels["consume"], args=(sm.move(x),))
    report.assert_passed()
    assert x.shape == (2,)
    report = sm.testing.check(kernels["affine"], args=(x, 2), memory_bytes=1)
    assert report.status == "resource_limit"
    with pytest.raises(sm.testing.CheckResourceLimit):
        report.assert_passed()
    with pytest.raises(ValueError):
        sm.benchmark(kernels["write"], args=(x, target), repeat=2)
    setups = []

    def setup():
        setups.append(1)
        target.copy_from(np.zeros(2, dtype=np.float32))
        return (x, target), {}

    measured = sm.benchmark(kernels["write"], setup=setup, warmup=1, repeat=2)
    assert len(setups) == 3 and len(measured.samples) == 2
    assert measured.median >= 0


def test_comparison_contract():
    sm.testing.assert_close(
        np.array([1.0], np.float32), np.array([1.01], np.float32), atol=0.02
    )
    with pytest.raises(AssertionError):
        sm.testing.assert_close(np.array([1]), np.array([[1]]))
    with pytest.raises(AssertionError):
        sm.testing.assert_close(np.uint64(2**63), np.uint64(2**63 + 1))
    with pytest.raises(AssertionError):
        sm.testing.assert_close(-0.0, 0.0)
    sm.testing.assert_close(-0.0, 0.0, preserve_signed_zero=False)
    with pytest.raises(AssertionError):
        sm.testing.assert_close(np.nan, np.nan)
    sm.testing.assert_close(np.nan, np.nan, equal_nan=True)
    with pytest.raises(TypeError):
        sm.testing.assert_close(np.float64(1), np.float64(1), ulps=0)
    sm.testing.assert_close(np.array([], np.float32), np.array([], np.float32))


def test_concurrent_calls(kernels, dev):
    x = sm.asarray(np.arange(16, dtype=np.float32), device=dev)
    with ThreadPoolExecutor(2) as pool:
        results = list(
            pool.map(lambda scale: kernels["affine"](x, scale).numpy(), [2.0, 3.0])
        )
    for scale, result in zip([2.0, 3.0], results):
        np.testing.assert_array_equal(result, np.arange(16) * scale + 1)


def test_scope_and_policy_validation(module, dev):
    with pytest.raises(ValueError, match="unknown numerical subject"):
        module.affine.prepare(
            device=dev,
            evaluation=sm.Feedback(0),
            precision=sm.Precision.bounded(
                sm.Tolerance(atol=0.01), outputs={"missing": sm.Tolerance()}
            ),
        )
    with pytest.raises(OverflowError):
        sm.Tolerance(ulps=2**70)
    with pytest.raises(OverflowError):
        sm.Feedback(0, seed=2**70)
    with pytest.raises(ValueError):
        sm.load([])
    scope = module.affine.scope(dimensions={"N": 4})
    with pytest.raises(ValueError, match="another entry"):
        module.consume.prepare(device=dev, evaluation=sm.Feedback(0, scope=scope))


def test_failed_workflow_retains_external_owner(kernels, dev):
    x = sm.asarray([1.0, 2.0], device=dev)
    w = sm.Workflow(device=dev)
    pending = w.enqueue(kernels["consume"], sm.move(x))
    view = x[:1]  # Appeared after enqueue; run must revalidate under its access gates.
    with pytest.raises(sm.InvocationError):
        w.run(pending)
    np.testing.assert_array_equal(x.numpy(), [1, 2])
    del view
    with pytest.raises(sm.WorkflowError):
        w.run(pending)
    first = sm.Workflow(device=dev)
    second = sm.Workflow(device=dev)
    a = first.enqueue(kernels["affine"], x, 2)
    with pytest.raises(sm.WorkflowError):
        second.enqueue(kernels["affine"], a, 2)
    b = second.enqueue(kernels["affine"], x, 3)
    np.testing.assert_array_equal(second.run(b).numpy(), [4, 7])


def test_native_snapshot_and_route(tmp_path):
    if not any(d.selector.startswith("metal:") for d in sm.devices()):
        pytest.skip("Metal device is unavailable")
    source = """fn native_add[N](x: &tensor[N] f32) -> tensor[N] f32:
    let mut y = zeros_like(x)
    parallel for i in 0..N:
        y[i] = x[i] + 1.0
    return y
native native_add for metal from "add.metal":
    launch native_add:
        threadgroups (ceil_div(N, 32), 1, 1)
        threads_per_threadgroup (32, 1, 1)
"""
    metal = """kernel void native_add(
 device const float *x [[buffer(SEISMIC_BUFFER_X)]],
 device float *y [[buffer(SEISMIC_RESULT_0_BUFFER)]],
 constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
 uint i [[thread_position_in_grid]]) {
 if (i < SEISMIC_DIM_N) { y[i] = x[i] + 1.0; }
}
"""
    src = tmp_path / "native.seismic"
    asset = tmp_path / "add.metal"
    src.write_text(source)
    asset.write_text(metal)
    old = sm.load(src, std=False)
    bundle = tmp_path / "native.seismicbundle"
    old.save(bundle)
    asset.write_text(metal.replace("+ 1.0", "+ 2.0"))
    fresh = sm.load(src, std=False)
    assert old.identity != fresh.identity
    restored = sm.load(bundle)
    assert restored.identity == old.identity
    dev = sm.device("metal")
    x = sm.asarray([1.0, 2.0], device=dev)
    old_kernel = restored.native_add.prepare_native(device=dev)
    new_kernel = fresh.native_add.prepare_native(device=dev)
    np.testing.assert_array_equal(old_kernel(x).numpy(), [2, 3])
    np.testing.assert_array_equal(new_kernel(x).numpy(), [3, 4])
    with pytest.raises(sm.WorkflowError):
        sm.Workflow(device=dev).enqueue(old_kernel, x)
    assert sm.testing.check(old_kernel, args=(x,)).status == "unsupported"
    with pytest.raises(ValueError):
        sm.load_source(source, std=False)


def test_standard_library_and_empty_arrays(dev):
    module = sm.load_source("fn scalar(x: f32) -> f32:\n    return x\n")
    assert "linear" in module.functions and "scalar" in module.functions
    empty = sm.asarray(np.empty((0, 3), dtype=np.float32), device=dev)
    assert empty.numpy().shape == (0, 3)
    assert empty.reshape((0, 1, 3)).nbytes == 0
    with pytest.raises(ValueError):
        empty.reshape((0, -1))


def test_narrow_scalars_tuple_input_and_ranges(dev):
    source = """fn narrow(a: f16, b: bf16) -> (f16, bf16):
    return a, b
fn tuple_arg(pair: (f32, i32)) -> (i32, f32):
    let (a, b) = pair
    return b, a
fn range_arg[N](x: &tensor[N] f32, span: range[N]) -> range[N]:
    return span
"""
    module = sm.load_source(source, std=False)
    prepare = lambda f: f.prepare(device=dev, evaluation=sm.Feedback(0))
    a = np.array([0x8000], dtype=np.uint16).view(np.float16)[0]
    b = np.array([0x7FC1], dtype=np.uint16).view(ml_dtypes.bfloat16)[0]
    x, y = prepare(module.narrow)(a, b)
    assert x.tobytes() == a.tobytes() and y.tobytes() == b.tobytes()
    assert prepare(module.tuple_arg)((np.float32(1.5), -7)) == (
        np.int32(-7),
        np.float32(1.5),
    )
    kernel = prepare(module.range_arg)
    tensor = sm.asarray([1.0, 2.0, 3.0], device=dev)
    assert kernel(tensor, range(1, 3)) == range(1, 3)
    with pytest.raises(TypeError):
        kernel(tensor, range(0, 3, 2))
