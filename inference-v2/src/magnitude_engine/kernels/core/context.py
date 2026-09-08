"""Trace-local state shared by declarations and compilation, never process-global."""

from contextvars import ContextVar

capturing: ContextVar[bool] = ContextVar("magnitude_kernel_capture", default=False)
markers: ContextVar[dict | None] = ContextVar("magnitude_kernel_markers", default=None)
