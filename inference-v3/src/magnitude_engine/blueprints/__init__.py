"""Explicit public construction catalog; importing it does not initialize devices."""

from magnitude_engine.blueprints import (
    execution,
    inputs,
    models,
    service,
    serving,
    weights,
)

__all__ = ["execution", "inputs", "models", "service", "serving", "weights"]
