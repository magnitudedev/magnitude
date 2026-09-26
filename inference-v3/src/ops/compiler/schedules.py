"""Static schedule requests, resolved before authored functions are composed.

Ops describes candidates and their complete construction inputs. A resolver may
load a TileLang-qualified choice or collect requests for explicit calibration;
normal planning never measures or compiles candidates.
"""

import json
from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import product
from typing import Protocol

from .dependencies import code_identity
from .program import _identity


@dataclass(frozen=True, slots=True)
class ScheduleRequest:
    semantic_kernel: str
    family: str
    construction: str
    workload: str
    numerical_mode: str
    candidates: tuple


class ScheduleResolver(Protocol):
    def select(self, request: ScheduleRequest, default): ...


def schedule_boundary(context, template, workload):
    """Keep child choices qualified by their enclosing physical operation."""
    return replace(
        context, schedule_scope=(*context.schedule_scope, (code_identity(template), workload))
    )


def select_schedule(context, name, candidates, default, *, template, workload):
    candidates = tuple(candidates)
    if not candidates or default not in candidates:
        raise ValueError("a schedule family must include its default candidate")
    if context.schedules is None:
        return default
    request = ScheduleRequest(
        name,
        sha256(json.dumps(_identity(candidates), sort_keys=True).encode()).hexdigest(),
        code_identity(template),
        json.dumps(
            _identity(
                (
                    workload,
                    context.compiler_target,
                    context.compiler_identity,
                    context.device_identity,
                    context.mode,
                    context.schedule_scope,
                )
            ),
            sort_keys=True,
        ),
        context.precision,
        candidates,
    )
    selected = context.schedules.select(request, default)
    if selected not in candidates:
        raise ValueError(f"selected schedule is outside {name}'s declared family")
    return selected


@dataclass(frozen=True, slots=True)
class ScheduleChoice:
    kernel: str
    value: object


@dataclass(frozen=True, slots=True)
class ScheduleBundle:
    """Joint physical choices for one complete composed operation."""

    choices: tuple[ScheduleChoice, ...]


class CollectSchedules:
    def __init__(self):
        self.requests = []
        self.defaults = []

    def select(self, request, default):
        self.requests.append(request)
        self.defaults.append(default)
        return default


class _ReplaySchedules:
    def __init__(self, requests, bundle):
        self.requests, self.bundle, self.position = requests, bundle, 0

    def select(self, request, default):
        if self.position >= len(self.requests):
            raise ValueError("composed schedule introduced an undeclared child choice")
        expected = self.requests[self.position]
        choice = self.bundle.choices[self.position]
        if request.semantic_kernel != expected.semantic_kernel or request.family != expected.family:
            raise ValueError("dependent schedule families require an explicit joint family")
        self.position += 1
        return choice.value


def select_composed_schedule(context, collected, *, name, template, workload, build):
    """Qualify a complete operation, including every child schedule combination.

    Child requests describe bounded portable families. The outer request contains
    their Cartesian product and complete construction identities; replay cannot
    silently introduce another choice or a conditionally different family.
    """
    requests = tuple(collected.requests)
    if not requests:
        return build(context)
    families = tuple(
        tuple(ScheduleChoice(r.semantic_kernel, candidate) for candidate in r.candidates)
        for r in requests
    )
    candidates = tuple(ScheduleBundle(choices) for choices in product(*families))
    default = ScheduleBundle(
        tuple(
            ScheduleChoice(r.semantic_kernel, value)
            for r, value in zip(requests, collected.defaults, strict=True)
        )
    )
    construction = tuple(
        (r.semantic_kernel, r.family, r.construction, r.workload, r.numerical_mode)
        for r in requests
    )
    selected = select_schedule(
        context, name, candidates, default, template=template, workload=(workload, construction)
    )
    replay = _ReplaySchedules(requests, selected)
    result = build(replace(context, schedules=replay))
    if replay.position != len(requests):
        raise ValueError("composed schedule omitted a declared child choice")
    return result


@dataclass(frozen=True, slots=True)
class QualifiedSchedules:
    """Read-only access to Magnitude's qualified schedule records, never a tuner.

    Validation identity is supplied by the calibration's independent reference
    suite. Strict mode requires coverage; misses cannot silently select a nearby
    workload, compiler, device, or numerical contract.
    """

    directory: str
    compiler: str
    target: str
    physical_device: str
    validation: str
    strict: bool = True

    @classmethod
    def for_device(cls, configuration, directory, validation, *, strict=True):
        from ..runtime.tilelang import describe_selection_configuration

        compiler, target, physical = describe_selection_configuration(configuration)
        return cls(str(directory), compiler, target, physical, validation, strict)

    def identity(self, request):
        from .selection import SelectionIdentity

        return SelectionIdentity(
            request.semantic_kernel,
            request.family,
            request.construction,
            request.workload,
            request.numerical_mode,
            self.validation,
            self.compiler,
            self.target,
            self.physical_device,
        )

    def select(self, request, default):
        from .selection import load_selection

        identity = self.identity(request)
        selected = load_selection(self.directory, identity)
        if selected is None:
            if self.strict:
                raise ValueError(
                    f"calibration required for {request.semantic_kernel}: {identity.key}"
                )
            return default
        # Configurations are stable candidate indices under the *entire* family
        # identity. Changing any member invalidates all indices.
        index = selected.config.get("candidate")
        if (
            set(selected.config) != {"candidate"}
            or type(index) is not int
            or not 0 <= index < len(request.candidates)
        ):
            raise ValueError("qualified record does not identify a declared candidate")
        return request.candidates[index]
