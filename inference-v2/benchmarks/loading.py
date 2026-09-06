"""Python modules author experiments; workers import the same named definition."""

import hashlib
import importlib
import json

from benchmarks.contracts import Experiment


def load(selection: str) -> Experiment:
    module, separator, name = selection.partition(":")
    if not separator or not module or not name.isidentifier():
        raise ValueError("select a Python experiment as module:experiment_name")
    value = getattr(importlib.import_module(module), name)
    if not isinstance(value, Experiment):
        raise TypeError(f"{selection} is not an Experiment")
    return value


def digest(experiment: Experiment) -> str:
    data = json.dumps(experiment.record(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()
