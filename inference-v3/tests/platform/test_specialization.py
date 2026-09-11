"""Specialization identity is binding-normalized without erasing parameter meaning."""

import pytest

from magnitude_engine.platform.execution import DType
from magnitude_engine.platform.specialization import Specialization


def test_specialization_normalizes_binding_without_erasing_parameter_meaning():
    def kernel(width, *, dtype=DType.F32, scale=0.0):
        raise AssertionError("identity lookup must not build IR")

    first = Specialization.bind(kernel, 1)
    assert first == Specialization.bind(kernel, width=1, dtype=DType.F32, scale=0.0)
    assert first != Specialization.bind(kernel, True)
    assert first != Specialization.bind(kernel, 1, scale=-0.0)
    assert first != Specialization.bind(kernel, 1, dtype="float32")
    with pytest.raises(TypeError, match="immutable"):
        Specialization.bind(kernel, [1])
