"""Bind a numerical test formula to its explicitly exercised physical body."""

import ops

_registered = {}


def implement(function, body):
    name = f"test.{function.__module__}.{function.__qualname__}"
    formula = ops.formula(function, id=name)
    identity = body.__module__, body.__qualname__
    if name in _registered:
        if _registered[name] != identity:
            raise ValueError("one test formula cannot exercise competing operation definitions")
    else:
        ops.operation(formula)(body)
        _registered[name] = identity
    return formula
