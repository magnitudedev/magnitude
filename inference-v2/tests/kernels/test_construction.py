"""Construction checks cover binding, dependency closure and executable cache identity."""

from dataclasses import replace

import mlx.core as mx
import pytest

from magnitude_engine.kernels.core import Input, KernelPlan, Launch, Output, Program, Scalar, Source
from magnitude_engine.kernels.core import plan as plan_module
from magnitude_engine.kernels.core.assembly import assemble
from performance.assembly import source_key

PROGRAM: Program | None = None


def invoke(x, y, scale=2.0, reverse=False):
    assert PROGRAM is not None
    inputs = (Input("x", x), Input("y", y))
    return KernelPlan(
        PROGRAM,
        inputs[::-1] if reverse else inputs,
        (Output("result", x.shape, x.dtype),),
        Launch((x.size, 1, 1), (32, 1, 1)),
        template=(("T", x.dtype),),
        constants=(Scalar("SCALE", scale),),
    ).run()[0]


@pytest.fixture
def sources(tmp_path, monkeypatch):
    (tmp_path / "body.metal").write_text(
        "uint i = thread_position_in_grid.x; result[i] = twice(x[i]) + T(SCALE) * y[i];"
    )
    helper = tmp_path / "helper.metal"
    helper.write_text("template <typename T> T twice(T x) { return x + x; }")
    monkeypatch.setattr(plan_module, "files", lambda package: tmp_path)

    def bind():
        return Program("construction_test", Source("body.metal", (Source("helper.metal"),)))

    monkeypatch.setattr(__import__(__name__, fromlist=["PROGRAM"]), "PROGRAM", bind())
    return tmp_path, bind


def test_named_bindings_and_scalar_specializations(sources):
    x, y = mx.arange(37, dtype=mx.float32), mx.ones(37)
    assert mx.array_equal(invoke(x, y), 2 * x + 2).item()
    assert mx.array_equal(invoke(x, y, reverse=True), 2 * x + 2).item()
    assert mx.array_equal(invoke(x, y, scale=3.0), 2 * x + 3).item()
    compiled = mx.compile(lambda a, b: invoke(a, b, scale=3.0))
    assert mx.array_equal(compiled(x, y), 2 * x + 3).item()


def test_source_snapshots_and_transitive_fingerprints(sources, monkeypatch):
    directory, bind = sources
    x, y = mx.ones(3), mx.ones(3)
    assert mx.array_equal(invoke(x, y), mx.full((3,), 4.0)).item()
    before, files = source_key((invoke,))
    assert files["magnitude_engine.kernels/helper.metal"]
    (directory / "helper.metal").write_text(
        "template <typename T> T twice(T x) { return x + x + x; }"
    )
    # A live executable continues to identify and execute its bound source snapshot.
    assert source_key((invoke,))[0] == before
    assert mx.array_equal(invoke(x, y), mx.full((3,), 4.0)).item()
    monkeypatch.setattr(__import__(__name__, fromlist=["PROGRAM"]), "PROGRAM", bind())
    assert source_key((invoke,))[0] != before
    assert mx.array_equal(invoke(x, y), mx.full((3,), 5.0)).item()


def test_shared_source_dependency_is_emitted_once(sources):
    directory, _ = sources
    (directory / "left.metal").write_text("// left")
    (directory / "right.metal").write_text("// right")
    helper = Source("helper.metal")
    body = Source(
        "body.metal",
        (Source("left.metal", (helper,)), Source("right.metal", (helper,))),
    )
    result = assemble(Program("diamond", body))
    assert [p for p, _ in result.sources] == [
        "helper.metal",
        "left.metal",
        "right.metal",
        "body.metal",
    ]
    assert result.header.count("T twice") == 1


def test_invalid_plan_rejected_before_device_submission(sources):
    assert PROGRAM is not None
    x = mx.ones(1)
    plan = KernelPlan(
        PROGRAM,
        (Input("x", x),),
        (Output("result", (1,), mx.float32),),
        Launch((1, 1, 1), (32, 1, 1)),
    )
    with pytest.raises(ValueError, match="unique"):
        replace(plan, inputs=(Input("x", x), Input("x", x)))
    with pytest.raises(ValueError, match="identifier"):
        replace(plan, inputs=(Input("x; invalid", x),))
    with pytest.raises(ValueError, match="nonnegative"):
        replace(plan, outputs=(Output("result", (-1,), mx.float32),))
    with pytest.raises(ValueError, match="positive"):
        Launch((0, 1, 1), (32, 1, 1))
    with pytest.raises(ValueError, match="1024"):
        Launch((1, 1, 1), (1024, 2, 1))
    with pytest.raises(ValueError, match="finite"):
        Scalar("EPS", float("nan"))
    with pytest.raises(ValueError, match="relative"):
        Source("../body.metal")


def test_dependency_order_participates_in_executable_identity(sources, monkeypatch):
    directory, _ = sources
    (directory / "first.metal").write_text("#define VALUE 1\n")
    (directory / "second.metal").write_text("#undef VALUE\n#define VALUE 2\n")
    first, second = Source("first.metal"), Source("second.metal")
    module = __import__(__name__, fromlist=["PROGRAM"])
    monkeypatch.setattr(module, "PROGRAM", Program("order", Source("body.metal", (first, second))))
    before = source_key((invoke,))[0]
    monkeypatch.setattr(module, "PROGRAM", Program("order", Source("body.metal", (second, first))))
    assert source_key((invoke,))[0] != before


def test_scalar_cache_preserves_integer_arithmetic_and_signed_zero(sources, monkeypatch):
    directory, _ = sources
    (directory / "body.metal").write_text(
        "uint i = thread_position_in_grid.x; result[i] = 1 / SCALE;"
    )
    program = Program("scalar_types", Source("body.metal"))
    monkeypatch.setattr(__import__(__name__, fromlist=["PROGRAM"]), "PROGRAM", program)
    x = mx.ones(1)
    assert invoke(x, x, scale=2).item() == 0
    assert invoke(x, x, scale=2.0).item() == 0.5
    assert (
        assemble(program, (Scalar("SCALE", 0.0),)).header
        != assemble(program, (Scalar("SCALE", -0.0),)).header
    )
