"""Owned ctypes boundary. Importing this module does not load a native library."""

from __future__ import annotations

import ctypes
import hashlib
import json
import sys
import threading
import weakref
from time import perf_counter_ns
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, JsonValue

from templates.events import (
    TERMINAL_CAUSES,
    ContentDelta,
    Event,
    Finish,
    ReasoningDelta,
    TerminalCause,
    ToolArguments,
    ToolComplete,
    ToolStart,
)

UPSTREAM_REVISION = "930e2fa5995789efbf249a8bf61325bb626e417b"
ABI_VERSION = 1


class BuildInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    version: int
    abi: int
    extraction: int
    upstream: str
    build: str


class GrammarTrigger(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    type: int
    value: str
    token: int


class PreparedDescription(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    version: Literal[1]
    prompt: str
    generation_prefix: str
    parser: str
    format: str
    grammar: str
    grammar_dialect: Literal["gbnf"]
    grammar_initial_prefix: str
    grammar_lazy: bool
    grammar_triggers: tuple[GrammarTrigger, ...]
    preserved_tokens: tuple[str, ...]
    additional_stops: tuple[str, ...]
    supports_thinking: bool
    thinking_start: str
    thinking_ends: tuple[str, ...]
    diagnostics: tuple[str, ...]


class NativeError(RuntimeError):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(message)


class _Buffer(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("size", ctypes.c_uint64),
        ("owner", ctypes.c_uint64),
    ]


class _Event(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_uint32),
        ("index", ctypes.c_uint32),
        ("text", ctypes.POINTER(ctypes.c_uint8)),
        ("text_size", ctypes.c_uint64),
        ("id", ctypes.POINTER(ctypes.c_uint8)),
        ("id_size", ctypes.c_uint64),
    ]


class _Events(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(_Event)), ("size", ctypes.c_uint64)]


class _Library:
    def __init__(self):
        directory = Path(__file__).resolve().parent / "_native"
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
        path = directory / f"libtemplates{suffix}"
        if not path.is_file() or not (directory / "build.json").is_file():
            raise NativeError(
                5, "Native templates library is missing; run native/templates/tools/build.py"
            )
        try:
            # CDLL releases the GIL during calls; no callbacks cross this boundary.
            self.lib = ctypes.CDLL(str(path))
            self._bind()
            if self.lib.templates_abi_version() != ABI_VERSION:
                raise NativeError(5, "Native templates ABI is incompatible")
            actual = BuildInfo.model_validate_json(self._output(self.lib.templates_build_info))
            expected = BuildInfo.model_validate_json((directory / "build.json").read_bytes())
            if (
                actual != expected
                or actual.upstream != UPSTREAM_REVISION
                or actual.abi != ABI_VERSION
            ):
                raise NativeError(5, "Native templates build identity is incompatible")
            self.identity = actual
        except (OSError, AttributeError) as error:
            raise NativeError(
                5, f"Cannot load bundled native templates library: {error}"
            ) from error

    def _bind(self):
        handle = ctypes.c_uint64
        buffer = ctypes.POINTER(_Buffer)
        events = ctypes.POINTER(_Events)
        signatures = {
            "abi_version": (ctypes.c_uint32, []),
            "build_info": (ctypes.c_int32, [buffer, buffer]),
            "buffer_release": (ctypes.c_int32, [handle]),
            "template_create": (
                ctypes.c_int32,
                [ctypes.c_char_p, handle, ctypes.POINTER(handle), buffer],
            ),
            "template_release": (ctypes.c_int32, [handle, buffer]),
            "template_inspect": (ctypes.c_int32, [handle, buffer, buffer]),
            "template_render": (ctypes.c_int32, [handle, ctypes.c_char_p, handle, buffer, buffer]),
            "request_create": (
                ctypes.c_int32,
                [handle, ctypes.c_char_p, handle, ctypes.POINTER(handle), buffer],
            ),
            "request_describe": (ctypes.c_int32, [handle, buffer, buffer]),
            "request_release": (ctypes.c_int32, [handle, buffer]),
            "stream_create": (ctypes.c_int32, [handle, handle, ctypes.POINTER(handle), buffer]),
            "stream_feed": (ctypes.c_int32, [handle, ctypes.c_char_p, handle, events, buffer]),
            "stream_finish": (ctypes.c_int32, [handle, ctypes.c_uint32, events, buffer]),
            "stream_release": (ctypes.c_int32, [handle, buffer]),
        }
        for name, (result, arguments) in signatures.items():
            function = getattr(self.lib, f"templates_{name}")
            function.restype = result
            function.argtypes = arguments

    def _take(self, buffer: _Buffer) -> bytes:
        try:
            return ctypes.string_at(buffer.data, buffer.size) if buffer.size else b""
        finally:
            if buffer.owner:
                status = self.lib.templates_buffer_release(buffer.owner)
                if status:
                    raise NativeError(status, "Native templates buffer ownership violation")

    def _check(self, status: int, error: _Buffer):
        message = self._take(error).decode("utf-8", errors="replace")
        if status:
            raise NativeError(status, message)

    def _output(self, function, *args) -> bytes:
        output, error = _Buffer(), _Buffer()
        status = function(*args, ctypes.byref(output), ctypes.byref(error))
        try:
            self._check(status, error)
        except BaseException:
            self._take(output)
            raise
        return self._take(output)

    def release(self, handle: int):
        error = _Buffer()
        self._check(self.lib.templates_template_release(handle, ctypes.byref(error)), error)

    def release_request(self, handle: int):
        error = _Buffer()
        self._check(self.lib.templates_request_release(handle, ctypes.byref(error)), error)

    def release_stream(self, handle: int):
        error = _Buffer()
        self._check(self.lib.templates_stream_release(handle, ctypes.byref(error)), error)


_library: _Library | None = None
_library_lock = threading.Lock()


def _get_library() -> _Library:
    global _library
    with _library_lock:
        if _library is None:
            _library = _Library()
        return _library


def _encode(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(
        "utf-8"
    )


class Template:
    """Compiled selected source. Calls and close are serialized per owner."""

    def __init__(self, source: str, *, special_tokens: dict[str, str] | None = None):
        self._library = _get_library()
        self._lock = threading.RLock()
        payload = _encode({"version": 1, "source": source, "special_tokens": special_tokens or {}})
        handle, error = ctypes.c_uint64(), _Buffer()
        status = self._library.lib.templates_template_create(
            payload, len(payload), ctypes.byref(handle), ctypes.byref(error)
        )
        self._library._check(status, error)
        self._handle = handle.value
        self.identity = hashlib.sha256(
            _encode(
                {
                    "source": source,
                    "special_tokens": dict(sorted((special_tokens or {}).items())),
                    "native": self._library.identity.model_dump(),
                }
            )
        ).hexdigest()
        self._finalizer = weakref.finalize(self, self._library.release, self._handle)

    def _require_open(self):
        if not self._finalizer.alive:
            raise NativeError(2, "Template has been closed")

    @property
    def build_info(self) -> BuildInfo:
        return self._library.identity

    def capabilities(self) -> dict[str, bool]:
        with self._lock:
            self._require_open()
            result = self._library._output(
                self._library.lib.templates_template_inspect, self._handle
            )
            return json.loads(result)["capabilities"]

    def render(self, context: dict[str, JsonValue], *, now: int) -> str:
        """Render authored context with explicit time and no injected controls."""
        payload = _encode({"version": 1, "context": context, "now": now})
        with self._lock:
            self._require_open()
            result = self._library._output(
                self._library.lib.templates_template_render, self._handle, payload, len(payload)
            )
        return result.decode("utf-8")

    def close(self):
        with self._lock:
            self._finalizer()

    def prepare(
        self,
        messages: list[dict[str, JsonValue]],
        *,
        now: int,
        tools: list[dict[str, JsonValue]] | None = None,
        tool_choice: Literal["auto", "none", "required"] = "auto",
        parallel_tool_calls: bool = True,
        template_arguments: dict[str, JsonValue] | None = None,
        json_schema: dict[str, JsonValue] | None = None,
    ) -> PreparedRequest:
        payload = {
            "version": 1,
            "messages": messages,
            "now": now,
            "tools": tools or [],
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
            "template_arguments": template_arguments or {},
        }
        if json_schema is not None:
            payload["json_schema"] = json_schema
        encoded = _encode(payload)
        with self._lock:
            self._require_open()
            handle, error = ctypes.c_uint64(), _Buffer()
            status = self._library.lib.templates_request_create(
                self._handle, encoded, len(encoded), ctypes.byref(handle), ctypes.byref(error)
            )
            self._library._check(status, error)
            return PreparedRequest(self._library, handle.value)

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, *_):
        self.close()


class PreparedRequest:
    """Immutable prepared data; valid independently of its source template owner."""

    def __init__(self, library: _Library, handle: int):
        self._library = library
        self._handle = handle
        self._lock = threading.RLock()
        self._finalizer = weakref.finalize(self, library.release_request, handle)
        try:
            self.description = PreparedDescription.model_validate_json(
                library._output(library.lib.templates_request_describe, handle)
            )
        except BaseException:
            self.close()
            raise

    def close(self):
        with self._lock:
            self._finalizer()

    def stream(self, *, max_output_bytes: int = 16 * 1024 * 1024) -> OutputStream:
        if not 1 <= max_output_bytes <= 64 * 1024 * 1024:
            raise ValueError("max_output_bytes must be between 1 and 64 MiB")
        with self._lock:
            if not self._finalizer.alive:
                raise NativeError(2, "Prepared request has been closed")
            handle, error = ctypes.c_uint64(), _Buffer()
            status = self._library.lib.templates_stream_create(
                self._handle, max_output_bytes, ctypes.byref(handle), ctypes.byref(error)
            )
            self._library._check(status, error)
            return OutputStream(self._library, handle.value)

    def parse(self, output: bytes, *, cause: TerminalCause = "natural") -> tuple[Event, ...]:
        with self.stream() as stream:
            return stream.feed(output) + stream.finish(cause)

    def __enter__(self) -> Self:
        if not self._finalizer.alive:
            raise NativeError(2, "Prepared request has been closed")
        return self

    def __exit__(self, *_):
        self.close()


class OutputStream:
    """One parser owner. Borrowed event spans are copied before another native call."""

    def __init__(self, library: _Library, handle: int):
        self._library = library
        self._handle = handle
        self._lock = threading.RLock()
        self._finalizer = weakref.finalize(self, library.release_stream, handle)
        self.elapsed_ns = self.input_bytes = self.calls = 0

    def _events(self, function, *args) -> tuple[Event, ...]:
        with self._lock:
            if not self._finalizer.alive:
                raise NativeError(2, "Output stream has been closed")
            batch, error = _Events(), _Buffer()
            status = function(self._handle, *args, ctypes.byref(batch), ctypes.byref(error))
            self._library._check(status, error)
            result: list[Event] = []
            for i in range(batch.size):
                event = batch.data[i]
                text = ctypes.string_at(event.text, event.text_size).decode("utf-8")
                match event.kind:
                    case 1:
                        result.append(ContentDelta(text=text))
                    case 2:
                        result.append(ReasoningDelta(text=text))
                    case 3:
                        call_id = ctypes.string_at(event.id, event.id_size).decode("utf-8")
                        result.append(ToolStart(index=event.index, name=text, id=call_id))
                    case 4:
                        result.append(ToolArguments(index=event.index, text=text))
                    case 5:
                        result.append(ToolComplete(index=event.index))
                    case 6:
                        result.append(Finish(cause=TERMINAL_CAUSES[event.index]))
                    case _:
                        raise NativeError(5, "Unknown native event tag")
            return tuple(result)

    def feed(self, data: bytes) -> tuple[Event, ...]:
        with self._lock:
            started = perf_counter_ns()
            try:
                return self._events(self._library.lib.templates_stream_feed, data, len(data))
            finally:
                self.elapsed_ns += perf_counter_ns() - started
                self.input_bytes += len(data)
                self.calls += 1

    def finish(self, cause: TerminalCause = "natural") -> tuple[Event, ...]:
        with self._lock:
            started = perf_counter_ns()
            try:
                return self._events(self._library.lib.templates_stream_finish, TERMINAL_CAUSES.index(cause))
            finally:
                self.elapsed_ns += perf_counter_ns() - started
                self.calls += 1

    def close(self):
        with self._lock:
            self._finalizer()

    def __enter__(self) -> Self:
        if not self._finalizer.alive:
            raise NativeError(2, "Output stream has been closed")
        return self

    def __exit__(self, *_):
        self.close()
