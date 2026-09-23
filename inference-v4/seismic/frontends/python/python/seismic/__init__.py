"""Call authored Seismic functions from Python. Computation remains in .seismic source."""

from __future__ import annotations

import inspect
import json
import math
import operator
import os
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

import ml_dtypes
import numpy as np

from . import _native


class SeismicError(RuntimeError):
    """A failure reported by the shared Seismic core."""


class SourceError(SeismicError):
    pass


class BundleError(SeismicError):
    pass


class TargetError(SeismicError):
    pass


class PreparationError(SeismicError):
    pass


class InvocationError(SeismicError):
    pass


class ExecutionError(SeismicError):
    pass


class TensorError(SeismicError):
    pass


class WorkflowError(SeismicError):
    pass


class InternalError(SeismicError):
    pass


def _invoke(fn, *args):
    try:
        return fn(*args)
    except _native.PanicException as exc:
        raise InternalError(f"Seismic internal invariant failed: {exc}") from exc
    except RuntimeError as exc:
        if len(exc.args) != 2:
            raise
        kind, message = exc.args
        cls = {
            "TypeError": TypeError,
            "ValueError": ValueError,
            "KeyError": KeyError,
            "OSError": OSError,
        }.get(kind, globals().get(kind, SeismicError))
        error = cls(message)
        error.code = kind
        raise error from exc


@dataclass(frozen=True)
class Element:
    name: str
    dense: str | None

    def __repr__(self):
        return f"seismic.{_PUBLIC_DTYPES.get(self.name, self.name)}"


_PUBLIC_DTYPES = {
    "f32": "float32",
    "f16": "float16",
    "bf16": "bfloat16",
    "i32": "int32",
    "u32": "uint32",
    "bool": "bool_",
}
_NUMPY_DTYPES = {
    "f32": np.dtype("float32"),
    "f16": np.dtype("float16"),
    "bf16": np.dtype(ml_dtypes.bfloat16),
    "i32": np.dtype("int32"),
    "u32": np.dtype("uint32"),
    "bool": np.dtype("bool"),
}


def element(name: str) -> Element:
    return Element(*_invoke(_native.element_info, name))


float32, float16, bfloat16, int32, uint32, bool_ = (element(n) for n in _PUBLIC_DTYPES)


def _dtype(value) -> Element:
    if isinstance(value, Element):
        if value.dense is None:
            raise TypeError("encoded elements require from_bytes")
        return element(value.name)
    dtype = np.dtype(value).newbyteorder("=")
    for name, candidate in _NUMPY_DTYPES.items():
        if dtype == candidate:
            return element(name)
    raise TypeError(
        f"unsupported dtype {dtype}; explicitly convert to a supported dtype"
    )


@dataclass(frozen=True)
class DeviceInfo:
    selector: str
    name: str
    backend: str


class Device:
    def __init__(self, inner):
        self._inner = inner

    @property
    def name(self):
        return self._inner.name

    @property
    def backend(self):
        return self._inner.backend

    @property
    def capabilities(self):
        return tuple(self._inner.capabilities)

    def memory_usage(self):
        charged, limit, pool_charged = self._inner.memory_usage()
        return MappingProxyType(
            {"charged": charged, "limit": limit, "pool_charged": pool_charged}
        )

    def set_memory_limit(self, value):
        if value is not None:
            value = _natural(value)
        _invoke(self._inner.set_memory_limit, value)

    def __repr__(self):
        return repr(self._inner)


def devices() -> tuple[DeviceInfo, ...]:
    return tuple(DeviceInfo(*v) for v in _invoke(_native.devices))


def device(selector: str | DeviceInfo | Device) -> Device:
    if isinstance(selector, Device):
        return selector
    if isinstance(selector, DeviceInfo):
        selector = selector.selector
    return Device(_invoke(_native.device, selector))


def _natural(value):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("expected an integer, not a boolean")
    value = operator.index(value)
    if value < 0:
        raise ValueError("expected a nonnegative integer")
    if value > 2**64 - 1:
        raise OverflowError("integer exceeds the Seismic uint64 domain")
    return value


def _shape(shape):
    if isinstance(shape, (int, np.integer)):
        shape = (shape,)
    return tuple(_natural(v) for v in shape)


def _host_array(data, dtype=None):
    if isinstance(data, (np.ndarray, np.generic)):
        array = np.asarray(data)
        target = _dtype(array.dtype) if dtype is None else _dtype(dtype)
    else:
        array = np.asarray(data)
        if dtype is None:
            if not array.size:
                raise TypeError("empty sequences require dtype")
            target = {"b": bool_, "i": int32, "u": int32, "f": float32}.get(
                array.dtype.kind
            )
            if target is None:
                raise TypeError(
                    "expected rectangular numeric data and a supported dtype"
                )
        else:
            target = _dtype(dtype)
    target_dtype = _NUMPY_DTYPES[target.name]
    if array.dtype == target_dtype:
        return np.require(array, requirements=["C"]), target
    if array.dtype.kind in "OcSUV" and array.dtype != _NUMPY_DTYPES["bf16"]:
        raise TypeError(
            "object, complex, string, and structured inputs are unsupported"
        )
    if target.name in ("i32", "u32"):
        bounds = np.iinfo(target_dtype)
        if (
            np.any(~np.isfinite(array))
            or np.any(array < bounds.min)
            or np.any(array > bounds.max)
            or np.any(array != np.trunc(array))
        ):
            raise OverflowError(f"values are not representable as {target_dtype}")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = array.astype(target_dtype)
    if target.name in ("f32", "f16", "bf16") and np.any(
        np.isfinite(array) & ~np.isfinite(converted)
    ):
        raise OverflowError(f"finite value overflows {target_dtype}")
    return np.require(converted, requirements=["C"]), target


class Tensor:
    def __init__(self, inner):
        self._inner = inner

    @property
    def shape(self):
        return tuple(_invoke(lambda: self._inner.shape))

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return math.prod(self.shape)

    @property
    def element(self):
        return element(_invoke(lambda: self._inner.element))

    @property
    def dtype(self):
        value = self.element
        return value if value.dense is not None else None

    @property
    def device(self):
        return Device(_invoke(lambda: self._inner.device))

    @property
    def nbytes(self):
        return _invoke(lambda: self._inner.nbytes)

    def numpy(self):
        dtype = self.dtype
        if dtype is None:
            raise TypeError("encoded tensors require an authored decode function")
        return (
            np.frombuffer(self.tobytes(), dtype=_NUMPY_DTYPES[dtype.name])
            .reshape(self.shape)
            .copy()
        )

    def item(self):
        if self.size != 1:
            raise ValueError("item requires exactly one element")
        return self.numpy().item()

    def tobytes(self):
        return _invoke(self._inner.read)

    def write_bytes(self, data):
        _invoke(self._inner.write, bytes(data))

    def copy(self):
        return Tensor(_invoke(self._inner.copy))

    def copy_from(self, source):
        if isinstance(source, Tensor):
            if source.shape != self.shape or source.element != self.element:
                raise TensorError("copy_from requires matching shape and element")
            _invoke(self._inner.copy_from, source._inner)
        else:
            array, dtype = _host_array(source)
            if array.shape != self.shape or dtype != self.dtype:
                raise TensorError("copy_from does not cast or broadcast")
            self.write_bytes(array.tobytes())

    def reshape(self, shape):
        if isinstance(shape, (int, np.integer)):
            shape = (shape,)
        if any(isinstance(n, (bool, np.bool_)) for n in shape):
            raise TypeError("boolean shape")
        shape = tuple(operator.index(v) for v in shape)
        if shape.count(-1) > 1 or any(n < -1 for n in shape):
            raise ValueError("invalid reshape dimensions")
        if -1 in shape:
            known = math.prod(n for n in shape if n != -1)
            if not known or self.size % known:
                raise ValueError("reshape dimension cannot be inferred uniquely")
            shape = tuple(self.size // known if n == -1 else n for n in shape)
        return Tensor(_invoke(self._inner.reshape, shape))

    def __getitem__(self, key):
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("only leading-axis slices with step one are supported")
        if not self.ndim:
            raise IndexError("cannot slice a scalar tensor")
        start, stop, _ = key.indices(self.shape[0])
        return Tensor(_invoke(self._inner.slice, start, max(start, stop)))

    def __bool__(self):
        raise TypeError("tensor truth values require explicit host inspection")

    def __repr__(self):
        try:
            return f"Tensor(shape={self.shape}, element={self.element.name}, device={self.device.backend})"
        except TensorError:
            return "Tensor(<moved>)"


def asarray(data, *, dtype=None, device=None, copy=None) -> Tensor:
    if copy is not None and copy is not True and copy is not False:
        raise TypeError("copy must be True, False, or None")
    if isinstance(data, Tensor):
        if dtype is not None and _dtype(dtype) != data.dtype:
            raise TypeError(
                "resident dtype conversion requires an authored function or explicit numpy() conversion"
            )
        target = data.device if device is None else globals()["device"](device)
        same = _invoke(data._inner.same_device, target._inner)
        if same:
            return data.copy() if copy is True else data
        if copy is False:
            raise ValueError("changing devices requires copying")
        return from_bytes(
            data.tobytes(), shape=data.shape, element=data.element, device=target
        )
    if device is None:
        raise TypeError("host array construction requires device")
    if copy is False:
        raise ValueError("host array import requires copying")
    array, target = _host_array(data, dtype)
    return from_bytes(array.tobytes(), shape=array.shape, element=target, device=device)


def zeros(shape, *, dtype, device) -> Tensor:
    return Tensor(
        _invoke(
            _native.Tensor.zeros,
            globals()["device"](device)._inner,
            _dtype(dtype).name,
            _shape(shape),
        )
    )


def from_bytes(data, *, shape, element, device) -> Tensor:
    name = element.name if isinstance(element, Element) else element
    return Tensor(
        _invoke(
            _native.Tensor.from_bytes,
            globals()["device"](device)._inner,
            name,
            _shape(shape),
            bytes(data),
        )
    )


@dataclass(frozen=True)
class _Move:
    tensor: Tensor


def move(tensor: Tensor):
    if not isinstance(tensor, (Tensor, Pending)):
        raise TypeError("move requires a Seismic tensor or pending tensor")
    return _Move(tensor)


@dataclass(frozen=True)
class Tolerance:
    atol: float = 0.0
    rtol: float = 0.0
    relative_floor: float = 0.0
    ulps: int | None = None

    def __post_init__(self):
        for name in ("atol", "rtol", "relative_floor"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError("tolerances must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if self.ulps is not None:
            object.__setattr__(self, "ulps", _natural(self.ulps))


@dataclass(frozen=True)
class Precision:
    kind: str
    tolerance: Tolerance = Tolerance()
    outputs: Any = None
    preserve_nan: bool = True
    preserve_infinity: bool = True
    preserve_signed_zero: bool = True
    preserve_subnormal: bool = True

    def __post_init__(self):
        if self.kind not in ("exact", "bounded", "unconstrained"):
            raise ValueError("unknown precision kind")
        if not isinstance(self.tolerance, Tolerance):
            raise TypeError("precision requires a Tolerance")
        values = dict(self.outputs or {})
        if any(not isinstance(v, Tolerance) for v in values.values()):
            raise TypeError("output overrides require Tolerance values")
        object.__setattr__(self, "outputs", MappingProxyType(values))

    @staticmethod
    def exact():
        return Precision("exact")

    @staticmethod
    def bounded(
        tolerance,
        *,
        outputs=None,
        preserve_nan=True,
        preserve_infinity=True,
        preserve_signed_zero=True,
        preserve_subnormal=True,
    ):
        return Precision(
            "bounded",
            tolerance,
            outputs,
            preserve_nan,
            preserve_infinity,
            preserve_signed_zero,
            preserve_subnormal,
        )

    @staticmethod
    def unconstrained():
        return Precision("unconstrained")

    def _config(self):
        return {
            "kind": self.kind,
            **asdict(self.tolerance),
            "outputs": {n: asdict(v) for n, v in self.outputs.items()},
            **{
                n: getattr(self, n)
                for n in (
                    "preserve_nan",
                    "preserve_infinity",
                    "preserve_signed_zero",
                    "preserve_subnormal",
                )
            },
        }


@dataclass(frozen=True)
class Analytical:
    def _config(self):
        return {"kind": "analytical"}


@dataclass(frozen=True)
class Interval:
    lower: Any
    upper: Any


@dataclass(frozen=True)
class Feedback:
    search_seconds: float
    scope: Any = None
    seed: int = 0
    experiment_memory_bytes: int = 268435456
    reference_work_limit: int = 1000000

    def __post_init__(self):
        seconds = float(self.search_seconds)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid search duration")
        object.__setattr__(self, "search_seconds", seconds)
        for name in ("seed", "experiment_memory_bytes", "reference_work_limit"):
            object.__setattr__(self, name, _natural(getattr(self, name)))

    def _config(self):
        return {
            "kind": "feedback",
            **{k: v for k, v in vars(self).items() if k != "scope"},
        }


def _argument(ty, value):
    kind = ty["kind"]
    if kind == "unit":
        if value is not None:
            raise TypeError("unit parameter requires None")
        return None
    if kind == "tuple":
        if not isinstance(value, tuple) or len(value) != len(ty["items"]):
            raise TypeError("tuple shape mismatch")
        return tuple(_argument(t, v) for t, v in zip(ty["items"], value))
    if kind == "tensor":
        if isinstance(value, _Move):
            return _native.move_tensor(value.tensor._inner)
        if not isinstance(value, Tensor):
            raise TypeError("tensor arguments require seismic.asarray first")
        return value._inner
    if kind == "range":
        if not isinstance(value, range) or value.step != 1:
            raise TypeError("range parameter requires range(start, stop) with step one")
        return _native.Scalar("range", _natural(value.start), _natural(value.stop))
    if kind == "index":
        return _native.Scalar("index", _natural(value))
    name = ty["dtype"]
    if name == "bool" and not isinstance(value, (bool, np.bool_)):
        raise TypeError("boolean parameter requires bool")
    if name in ("i32", "u32"):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("integer parameter cannot accept bool")
        value = operator.index(value)
        bounds = np.iinfo(_NUMPY_DTYPES[name])
        if not bounds.min <= value <= bounds.max:
            raise OverflowError(f"value is not representable as {name}")
    array, _ = _host_array(value, element(name))
    if array.shape != ():
        raise TypeError("scalar parameter requires a scalar")
    return _native.Scalar(name, int.from_bytes(array.tobytes(), "little"))


def _result(ty, value):
    kind = ty["kind"]
    if kind == "unit":
        return None
    if kind == "tuple":
        return tuple(_result(t, v) for t, v in zip(ty["items"], value))
    if kind == "tensor":
        return Tensor(value)
    name, word, end = value
    if name == "range":
        return range(word, end)
    if name == "index":
        return word
    dtype = _NUMPY_DTYPES[name]
    return np.frombuffer(word.to_bytes(dtype.itemsize, "little"), dtype=dtype)[0]


class Function:
    def __init__(self, inner):
        self._inner = inner
        self._schema = json.loads(inner.signature)
        self.__name__ = self._schema["name"]
        self.__signature__ = inspect.Signature(
            [
                inspect.Parameter(
                    p["name"],
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=p["type"],
                )
                for p in self._schema["parameters"]
            ],
            return_annotation=self._schema["result"],
        )

    @property
    def signature(self):
        return self.__signature__

    @property
    def elements(self):
        return tuple(self._schema["elements"])

    @property
    def dimensions(self):
        return tuple(self._schema["dimensions"])

    @property
    def numerical_subjects(self):
        return tuple(self._schema["numerical_subjects"])

    def scope(
        self, *, dimensions=None, scalars=None, range_starts=None, range_ends=None
    ):
        parameters = {p["name"]: p["type"] for p in self._schema["scope_parameters"]}
        constraints = []
        for kind, values in (
            ("dimensions", dimensions),
            ("scalars", scalars),
            ("range_starts", range_starts),
            ("range_ends", range_ends),
        ):
            for name, v in (values or {}).items():
                if kind == "dimensions":
                    if name not in self.dimensions:
                        raise ValueError(f"unknown dimension {name}")
                    ty = {"kind": "index"}
                else:
                    if name not in parameters:
                        raise ValueError(f"unknown scalar or range {name}")
                    ty = parameters[name]
                    if kind.startswith("range_"):
                        if ty["kind"] != "range":
                            raise TypeError(f"{name} is not a range")
                        ty = {"kind": "index"}
                a, b = (v.lower, v.upper) if isinstance(v, Interval) else (v, v)
                constraints.append((kind, name, _argument(ty, a), _argument(ty, b)))
        return _invoke(self._inner.scope, constraints)

    def prepare(self, *, device, elements=None, precision=None, evaluation=None):
        precision = precision or Precision.exact()
        evaluation = evaluation or Analytical()
        return self._prepare(device, elements, precision, evaluation, False)

    def start_feedback(self, *, device, options, elements=None, precision=None):
        if not isinstance(options, Feedback):
            raise TypeError("options must be Feedback")
        dev = globals()["device"](device)
        precision = precision or Precision.exact()
        elements = {
            name: (v.name if isinstance(v, Element) else element(v).name)
            for name, v in (elements or {}).items()
        }
        config = json.dumps(
            {"precision": precision._config(), "evaluation": options._config()},
            allow_nan=False,
        )
        session, initial = _invoke(
            self._inner.start_feedback, dev._inner, elements, config, options.scope
        )
        return FeedbackSession(session, Kernel(initial, self, dev, precision, False))

    def prepare_native(self, *, device, elements=None):
        return self._prepare(device, elements, Precision.exact(), Analytical(), True)

    def _prepare(self, dev, elements, precision, evaluation, native):
        dev = device(dev)
        elements = {
            name: (v.name if isinstance(v, Element) else element(v).name)
            for name, v in (elements or {}).items()
        }
        config = json.dumps(
            {"precision": precision._config(), "evaluation": evaluation._config()},
            allow_nan=False,
        )
        inner = _invoke(
            self._inner.prepare,
            dev._inner,
            elements,
            config,
            native,
            getattr(evaluation, "scope", None),
        )
        return Kernel(inner, self, dev, precision, native)

    def __call__(self, *args, **kwargs):
        raise TypeError("prepare this function for a device before calling it")

    def __repr__(self):
        return f"Function {self.__name__}{self.signature}"


class Kernel:
    def __init__(self, inner, function, device, precision, native):
        self._inner, self.function, self.device = inner, function, device
        self.precision, self.native = precision, native
        self.__signature__ = function.signature
        self.__name__ = function.__name__

    @property
    def signature(self):
        return self.__signature__

    @property
    def preparation_report(self):
        return MappingProxyType(
            {
                "seconds": self._inner.preparation_seconds,
                "route": "native" if self.native else "compiled",
                "build_profile": _native.build_profile,
            }
        )

    def _arguments(self, args, kwargs):
        bound = self.signature.bind(*args, **kwargs)
        return tuple(
            _argument(p["type"], bound.arguments[p["name"]])
            for p in self.function._schema["parameters"]
        )

    def __call__(self, *args, **kwargs):
        encoded = self._arguments(args, kwargs)
        raw = _invoke(self._inner.call, encoded)
        return _result(self.function._schema["result"], raw)


class FeedbackSession:
    def __init__(self, inner, kernel):
        self._inner, self.kernel = inner, kernel

    @property
    def report(self):
        return MappingProxyType(json.loads(self._inner.report))

    def continue_for(self, seconds):
        inner = _invoke(self._inner.continue_for, seconds)
        old = self.kernel
        self.kernel = Kernel(inner, old.function, old.device, old.precision, False)
        return self.kernel

    def close(self):
        _invoke(self._inner.close)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Pending:
    def __init__(self, inner):
        self._inner = inner
        self._schema = json.loads(inner.signature)

    def __bool__(self):
        raise TypeError("pending values cannot be inspected before run")

    def __array__(self, *args, **kwargs):
        raise TypeError("pending values cannot be converted before run")

    def __iter__(self):
        raise TypeError("pending values cannot be iterated")

    def __getitem__(self, key):
        if not isinstance(key, slice) or key.step not in (None, 1) or key.stop is None:
            raise TypeError(
                "pending leading slices require an explicit nonnegative stop and step one"
            )
        return Pending(
            _invoke(self._inner.slice, _natural(key.start or 0), _natural(key.stop))
        )


class Workflow:
    def __init__(self, *, device):
        self.device = globals()["device"](device)
        self._inner = _native.Workflow(self.device._inner)
        self._closed = False

    def enqueue(self, kernel, /, *args, **kwargs):
        if self._closed:
            raise WorkflowError("workflow is closed")
        if not isinstance(kernel, Kernel):
            raise TypeError("enqueue requires a prepared kernel")
        bound = kernel.signature.bind(*args, **kwargs)

        def argument(ty, v):
            if isinstance(v, Pending):
                return v._inner
            if isinstance(v, _Move) and isinstance(v.tensor, Pending):
                return v.tensor._inner.moved()
            if ty["kind"] == "tuple":
                if not isinstance(v, tuple) or len(v) != len(ty["items"]):
                    raise TypeError("tuple mismatch")
                return tuple(argument(t, x) for t, x in zip(ty["items"], v))
            return _argument(ty, v)

        encoded = tuple(
            argument(p["type"], bound.arguments[p["name"]])
            for p in kernel.function._schema["parameters"]
        )

        def wrap(v):
            if isinstance(v, tuple):
                return tuple(wrap(x) for x in v)
            return None if v is None else Pending(v)

        return wrap(_invoke(self._inner.enqueue, kernel._inner, encoded))

    def run(self, outputs=None):
        if self._closed:
            raise WorkflowError("workflow is closed")
        self._closed = True

        def encode(v):
            if v is None:
                return None
            if isinstance(v, tuple):
                return tuple(encode(x) for x in v)
            if not isinstance(v, Pending):
                raise TypeError("run outputs must be pending values")
            return v._inner

        try:
            encoded = encode(outputs)
        except Exception:
            self._inner.close()
            raise
        raw = _invoke(self._inner.run, encoded)
        cache = {}

        def decode(p, v):
            if p is None:
                return None
            if isinstance(p, tuple):
                return tuple(decode(x, y) for x, y in zip(p, v))
            if id(p) not in cache:
                cache[id(p)] = _result(p._schema, v)
            return cache[id(p)]

        return decode(outputs, raw)


class Module:
    def __init__(self, inner):
        self._inner = inner
        self._functions = {
            name: Function(_invoke(inner.function, name)) for name in inner.names()
        }

    @property
    def identity(self):
        return self._inner.identity

    @property
    def functions(self):
        return MappingProxyType(self._functions)

    def __getitem__(self, name):
        return self._functions[name]

    def __getattr__(self, name):
        try:
            return self._functions[name]
        except KeyError:
            raise AttributeError(name) from None

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self._functions))

    def save(self, path):
        _invoke(self._inner.save, os.fspath(path))


def load(path_or_paths, *, std=True) -> Module:
    if isinstance(path_or_paths, (str, os.PathLike)):
        path_or_paths = [path_or_paths]
    return Module(_invoke(_native.load, [os.fspath(p) for p in path_or_paths], std))


def load_source(text, *, name="<string>", std=True, base_dir=None) -> Module:
    return Module(
        _invoke(
            _native.load_source,
            text,
            name,
            std,
            None if base_dir is None else os.fspath(base_dir),
        )
    )


from . import testing as testing
from .benchmark import benchmark as benchmark
