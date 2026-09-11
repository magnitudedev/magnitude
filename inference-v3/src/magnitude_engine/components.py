"""Annotation required by the verbatim session-bench client.

The copied client imports this decorator. Preserve its source label as inert
external-tool metadata; it does not define the v3 component hierarchy, identify
Metrics, register implementations, or wrap execution. Engine components use
their typed contracts and actual construction/binding relationships.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ExternalComponent:
    source_label: str


def component[T](identity: str) -> Callable[[T], T]:
    def annotate(value: T) -> T:
        value.__external_component__ = ExternalComponent(identity)
        return value

    return annotate
