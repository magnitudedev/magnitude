from collections.abc import Callable, Mapping, Sequence
from inspect import Signature
from os import PathLike
from typing import Any

from numpy.typing import ArrayLike, NDArray

from . import testing as testing
from .benchmark import BenchmarkResult

class SeismicError(RuntimeError): ...
class SourceError(SeismicError): ...
class BundleError(SeismicError): ...
class TargetError(SeismicError): ...
class PreparationError(SeismicError): ...
class InvocationError(SeismicError): ...
class ExecutionError(SeismicError): ...
class TensorError(SeismicError): ...
class WorkflowError(SeismicError): ...
class InternalError(SeismicError): ...

class Element:
    name: str
    dense: str | None

float32: Element
float16: Element
bfloat16: Element
int32: Element
uint32: Element
bool_: Element

def element(name: str) -> Element: ...

class DeviceInfo:
    selector: str
    name: str
    backend: str

class Device:
    @property
    def name(self) -> str: ...
    @property
    def backend(self) -> str: ...
    @property
    def capabilities(self) -> tuple[str, ...]: ...
    def memory_usage(self) -> Mapping[str, int | None]: ...
    def set_memory_limit(self, value: int | None) -> None: ...

def devices() -> tuple[DeviceInfo, ...]: ...
def device(selector: str | DeviceInfo | Device) -> Device: ...

class Tensor:
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def element(self) -> Element: ...
    @property
    def dtype(self) -> Element | None: ...
    @property
    def device(self) -> Device: ...
    @property
    def nbytes(self) -> int: ...
    def numpy(self) -> NDArray[Any]: ...
    def item(self) -> Any: ...
    def tobytes(self) -> bytes: ...
    def write_bytes(self, data: bytes | bytearray | memoryview) -> None: ...
    def copy(self) -> Tensor: ...
    def copy_from(self, source: Tensor | ArrayLike) -> None: ...
    def reshape(self, shape: int | Sequence[int]) -> Tensor: ...
    def __getitem__(self, key: slice) -> Tensor: ...

def asarray(
    data: Tensor | ArrayLike,
    *,
    dtype: Any = None,
    device: str | Device | None = None,
    copy: bool | None = None,
) -> Tensor: ...
def zeros(
    shape: int | Sequence[int], *, dtype: Any, device: str | Device
) -> Tensor: ...
def from_bytes(
    data: bytes | bytearray | memoryview,
    *,
    shape: Sequence[int],
    element: Element | str,
    device: str | Device,
) -> Tensor: ...

class _Move: ...

def move(tensor: Tensor | Pending) -> _Move: ...

class Tolerance:
    atol: float
    rtol: float
    relative_floor: float
    ulps: int | None
    def __init__(
        self,
        atol: float = 0.0,
        rtol: float = 0.0,
        relative_floor: float = 0.0,
        ulps: int | None = None,
    ) -> None: ...

class Precision:
    kind: str
    @staticmethod
    def exact() -> Precision: ...
    @staticmethod
    def bounded(
        tolerance: Tolerance,
        *,
        outputs: Mapping[str, Tolerance] | None = None,
        preserve_nan: bool = True,
        preserve_infinity: bool = True,
        preserve_signed_zero: bool = True,
        preserve_subnormal: bool = True,
    ) -> Precision: ...
    @staticmethod
    def unconstrained() -> Precision: ...

class Interval:
    lower: Any
    upper: Any
    def __init__(self, lower: Any, upper: Any) -> None: ...

class Analytical: ...

class Feedback:
    def __init__(
        self,
        search_seconds: float,
        scope: Any = None,
        seed: int = 0,
        experiment_memory_bytes: int = 268435456,
        reference_work_limit: int = 1000000,
    ) -> None: ...

class Function:
    @property
    def signature(self) -> Signature: ...
    @property
    def elements(self) -> tuple[str, ...]: ...
    @property
    def dimensions(self) -> tuple[str, ...]: ...
    @property
    def numerical_subjects(self) -> tuple[str, ...]: ...
    def scope(
        self,
        *,
        dimensions: Mapping[str, Any] | None = None,
        scalars: Mapping[str, Any] | None = None,
        range_starts: Mapping[str, Any] | None = None,
        range_ends: Mapping[str, Any] | None = None,
    ) -> Any: ...
    def prepare(
        self,
        *,
        device: str | Device,
        elements: Mapping[str, Element | str] | None = None,
        precision: Precision | None = None,
        evaluation: Analytical | Feedback | None = None,
    ) -> Kernel: ...
    def prepare_native(
        self,
        *,
        device: str | Device,
        elements: Mapping[str, Element | str] | None = None,
    ) -> Kernel: ...
    def start_feedback(
        self,
        *,
        device: str | Device,
        options: Feedback,
        elements: Mapping[str, Element | str] | None = None,
        precision: Precision | None = None,
    ) -> FeedbackSession: ...

class Kernel:
    function: Function
    device: Device
    precision: Precision
    native: bool
    @property
    def signature(self) -> Signature: ...
    @property
    def preparation_report(self) -> Mapping[str, Any]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

class FeedbackSession:
    kernel: Kernel
    @property
    def report(self) -> Mapping[str, Any]: ...
    def continue_for(self, seconds: float) -> Kernel: ...
    def close(self) -> None: ...
    def __enter__(self) -> FeedbackSession: ...  # noqa: PYI034 (Python 3.10)
    def __exit__(self, *exc: object) -> None: ...

class Pending:
    def __getitem__(self, key: slice) -> Pending: ...

class Workflow:
    def __init__(self, *, device: str | Device) -> None: ...
    def enqueue(self, kernel: Kernel, /, *args: Any, **kwargs: Any) -> Any: ...
    def run(self, outputs: Any = None) -> Any: ...

class Module:
    @property
    def identity(self) -> str: ...
    @property
    def functions(self) -> Mapping[str, Function]: ...
    def __getitem__(self, name: str) -> Function: ...
    def __getattr__(self, name: str) -> Function: ...
    def save(self, path: str | PathLike[str]) -> None: ...

def load(
    path_or_paths: str | PathLike[str] | Sequence[str | PathLike[str]],
    *,
    std: bool = True,
) -> Module: ...
def load_source(
    text: str,
    *,
    name: str = "<string>",
    std: bool = True,
    base_dir: str | PathLike[str] | None = None,
) -> Module: ...
def benchmark(
    kernel: Kernel,
    *,
    args: tuple[Any, ...] = (),
    kwargs: Mapping[str, Any] | None = None,
    setup: Callable[[], tuple[tuple[Any, ...], Mapping[str, Any]]] | None = None,
    warmup: int = 3,
    repeat: int = 20,
) -> BenchmarkResult: ...
