"""Explicit public construction catalog; importing it does not initialize devices."""

from magnitude_engine.blueprints import (
    artifacts,
    execution,
    inputs,
    models,
    operations,
    service,
    serving,
)

__all__ = ["artifacts", "execution", "inputs", "models", "operations", "service", "serving"]
