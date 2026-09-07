"""Policy and execution-boundary measurements using production operations."""

from dataclasses import dataclass
from typing import Any, cast

from performance.assembly import bind_operation
from performance.records import Observation, digest
from performance.runner import recording


def accounting(component=None, *, transactions=1024, pressure=False, **record):
    from magnitude_engine.resources.budget import MemoryBudget

    component = component or MemoryBudget(12288)
    bound = bind_operation(component, "MEMORY:ACCOUNTING:MAG:RESERVATIONS")
    budget = bound.instance
    with recording(
        bound,
        benchmark="control.accounting",
        workload={"transactions": transactions, "pressure": pressure},
        boundary="reservation-release",
        **record,
    ) as run:
        if transactions < 1 or budget.snapshot().reserved:
            raise ValueError("positive work and an idle budget are required")

        def execute():
            rejected = 0
            for _ in range(transactions):
                with budget.reserve("retained", 8192), budget.reserve("in-flight", 4096):
                    if pressure:
                        try:
                            extra = budget.reserve("over-budget", 1)
                        except MemoryError:
                            rejected += 1
                        else:
                            extra.close()
            return rejected

        def validate(rejected):
            usage = budget.snapshot()
            if usage.reserved or usage.owners or rejected != transactions * pressure:
                raise ValueError("reservation ledger violates capacity/release oracle")
            return Observation(
                digest(rejected),
                {
                    "transactions": transactions,
                    "rejected": rejected,
                    "peak_reserved_bytes": usage.peak,
                },
            )

        run.measure(execute, validate=validate, deterministic=True)
    return run


@dataclass
class MetadataCheckpoint:
    length: int
    closed: bool = False
    reclaimable: bool = True

    def close(self):
        self.closed = True

    def retained_storage(self):
        return ()


def prefix_lookup(component=None, *, context_tokens=4096, entries=32, queries=64, **record):
    from magnitude_engine.engine.prefixes.index import PrefixIdentity
    from magnitude_engine.engine.prefixes.radix import Radix
    from magnitude_engine.engine.prefixes.retention import LeastRecentlyUsed

    owned = component is None
    component = component or Radix(retention=LeastRecentlyUsed(entries, None))
    bound = bind_operation(component, "CACHE:PREFIX:MAG:CHECKPOINTS")
    store = bound.instance
    workload = {
        "context_tokens": context_tokens,
        "entries": entries,
        "queries": queries,
        "eligible_prefix_tokens": context_tokens * queries,
    }
    with recording(
        bound,
        benchmark="control.prefix_lookup",
        workload=workload,
        boundary="metadata-prefix-lookup-lease-release",
        **record,
    ) as run:
        try:
            if min(context_tokens, entries, queries) < 1:
                raise ValueError("positive prefix geometry required")
            identities = [
                PrefixIdentity(
                    b"performance",
                    tuple(
                        [(t, b"") for t in range(context_tokens - 1)] + [(context_tokens + i, b"")]
                    ),
                )
                for i in range(entries)
            ]
            for identity in identities:
                store.retain(identity, MetadataCheckpoint(context_tokens))
            requests = [
                PrefixIdentity(i.namespace, i.tokens + ((-1, b""),))
                for i in (identities[j % entries] for j in range(queries))
            ]

            def execute():
                matches = []
                for query in requests:
                    lease = store.match(query)
                    if lease is None:
                        raise ValueError("known compatible prefix did not match")
                    try:
                        matches.append(lease.checkpoint.length)
                    finally:
                        lease.close()
                return matches

            def validate(matches):
                if matches != [context_tokens] * queries:
                    raise ValueError("prefix index violated longest-prefix oracle")
                return Observation(
                    digest(matches), {"queries": queries}, metrics={"REUSE": sum(matches)}
                )

            run.measure(execute, validate=validate, deterministic=True)
        finally:
            if owned:
                store.close()
    return run


def sampling(component=None, *, vocabulary=248320, positions=4, policy="greedy", **record):
    import mlx.core as mx
    import numpy as np

    from magnitude_engine.generation.sampling import SequenceSampler
    from magnitude_engine.generation.sampling_policy import SamplingPolicy

    if policy not in ("greedy", "filtered", "categorical"):
        raise ValueError("unknown sampling policy")
    component = component or SequenceSampler(
        SamplingPolicy(
            temperature=0 if policy == "greedy" else 0.7,
            seed=73,
            top_k=40 if policy == "filtered" else 0,
            top_p=0.9 if policy == "filtered" else 1,
        )
    )
    bound = bind_operation(component, "GENERATION:SAMPLING:MAG:POSITION_KEYED")
    workload = {
        "vocabulary": vocabulary,
        "positions": positions,
        "policy": policy,
        "element_bytes": 2,
    }
    with recording(bound, benchmark="control.sampling", workload=workload, **record) as run:
        if min(vocabulary, positions) < 1:
            raise ValueError("positive sampling geometry required")
        raw = mx.random.normal((positions, vocabulary), key=mx.random.key(918)).astype(mx.bfloat16)
        mx.eval(raw)
        expected = np.asarray(raw.astype(mx.float32)).argmax(axis=1).tolist()

        def validate(outputs):
            values = [cast(int, token.item()) for token in outputs]
            if policy == "greedy" and values != expected:
                raise ValueError("greedy sampler disagrees with NumPy argmax")
            if any(not 0 <= token < vocabulary for token in values):
                raise ValueError("sample outside vocabulary")
            return Observation(
                digest(values),
                {"positions": positions},
                {
                    "tokens": values,
                    "validation": "argmax" if policy == "greedy" else "seeded token bounds",
                },
            )

        run.measure(
            lambda: [bound.instance.sample(raw[i], i) for i in range(positions)],
            complete=lambda out: mx.eval(*out),
            validate=validate,
            deterministic=True,
        )
    return run


def acceptance(component=None, *, width=4, rounds=64, **record):
    import mlx.core as mx

    from magnitude_engine.generation.acceptance import accept_prefix

    bound = bind_operation(component or accept_prefix, "GENERATION:ACCEPTANCE:MAG:PREFIX")
    with recording(
        bound, benchmark="control.acceptance", workload={"width": width, "rounds": rounds}, **record
    ) as run:
        if width < 0 or rounds < 1:
            raise ValueError("invalid acceptance geometry")
        inputs, expected = [], []
        for index in range(rounds):
            proposed = list(range(1, width + 1))
            target = proposed + [width + 2]
            mismatch = index % (width + 1)
            if mismatch < width:
                target[mismatch] = width + 3
            stops = (1 + ((index // (width + 1)) % max(1, width)),) if index % 3 == 0 else ()
            count = 0
            while (
                count < width and proposed[count] == target[count] and proposed[count] not in stops
            ):
                count += 1
            expected.append((count, target[count]))
            inputs.append((mx.array(proposed, dtype=mx.int32), mx.array(target), stops))
        mx.eval(*[v for proposed, target, _ in inputs for v in (proposed, target)])

        def validate(outputs):
            actual = [(cast(int, o.count.item()), cast(int, o.bonus.item())) for o in outputs]
            if actual != expected:
                raise ValueError("acceptance disagrees with scalar prefix scan")
            return Observation(digest(actual), {"rounds": rounds}, {"accepted_bonus": actual})

        run.measure(
            lambda: [bound.instance(*args) for args in inputs],
            complete=lambda out: mx.eval(*[v for o in out for v in (o.count, o.bonus)]),
            validate=validate,
            deterministic=True,
        )
    return run


def device(component=None, *, elements=4096, operations=8, execution="scoped", **record):
    import mlx.core as mx
    import numpy as np

    from magnitude_engine.models.execution import ExecutionOwner

    owned = component is None
    bound = bind_operation(component or ExecutionOwner(), "EXECUTION:DEVICE:MAG:ASYNC")
    owner = bound.instance
    with recording(
        bound,
        benchmark="control.device",
        workload={"elements": elements, "operations": operations, "execution": execution},
        **record,
    ) as run:
        try:
            if min(elements, operations) < 1 or execution not in ("scoped", "direct"):
                raise ValueError("invalid execution workload")
            source = mx.linspace(-1, 1, elements)
            mx.eval(source)

            def execute():
                value = source
                for _ in range(operations):
                    value = value * 0.5 + 0.25
                if execution == "scoped":
                    with owner.scope() as scope:
                        pending = scope.seal(value)
                        pending.submit()
                    return value, pending
                mx.async_eval(value)
                return value, None

            def complete(result):
                value, pending = result
                pending.complete() if pending else mx.eval(value)

            def validate(result):
                actual = np.asarray(result[0])
                expected = 0.5 + (np.asarray(source) - 0.5) * (0.5**operations)
                if not np.allclose(actual, expected, atol=1e-7, rtol=1e-7):
                    raise ValueError("graph disagrees with closed-form affine composition")
                return Observation(digest(actual.tolist()), {"input_bytes": source.nbytes})

            run.measure(
                execute,
                prepare=owner.backend.drain,
                complete=complete,
                validate=validate,
                deterministic=True,
            )
        finally:
            if owned:
                owner.close()
    return run


def scheduling(component=None, *, rounds=128, **record):
    from magnitude_engine.engine.scheduler.contracts import CompletedService, Runnable
    from magnitude_engine.engine.scheduler.time_shared import TimeShared

    bound = bind_operation(
        component or TimeShared(prefill_tokens=512, decode_share=0.5),
        "SCHEDULING:SERVICE:MAG:TIME_SHARING",
    )
    scheduler = bound.instance
    with recording(
        bound,
        benchmark="control.scheduling",
        workload={"rounds": rounds},
        boundary="policy-bookkeeping-supplied-service",
        **record,
    ) as run:
        if rounds < 1:
            raise ValueError("positive scheduling rounds required")
        rows = (Runnable("prompt", 65536, 64), Runnable("decode", 0, 64))

        def execute():
            phases = []
            for _ in range(rounds):
                selected = scheduler.select(rows)
                if selected is None:
                    raise ValueError("ready workload received no service plan")
                phases.append(selected.phase)
                scheduler.observe(
                    CompletedService(
                        selected.phase,
                        40000000 if selected.phase == "prefill" else 10000000,
                        512 if selected.phase == "prefill" else 0,
                    )
                )
            return phases

        def validate(phases):
            prompt = decode = 0
            for phase in phases:
                if phase == "prefill":
                    prompt += 40000000
                else:
                    decode += 10000000
                if not -10000000 <= prompt - decode <= 40000000:
                    raise ValueError("service share exceeds one-round overshoot")
            return Observation(
                digest(phases),
                {"rounds": rounds, "supplied_prefill_ns": prompt, "supplied_decode_ns": decode},
            )

        run.measure(
            execute, prepare=scheduler.reset, validate=validate, deterministic=True, dimension=None
        )
    return run


def policy_suite(**record):
    return [accounting(pressure=pressure, **record) for pressure in (False, True)] + [
        scheduling(**record),
        prefix_lookup(**record),
    ]


@dataclass(eq=False)
class DescriptorState:
    runtime: Any
    identity: int
    pending: None = None
    failed: bool = False

    def complete_committed(self):
        return False


class DescriptorRuntime:
    """Controlled compatibility and zero neural work; engine grouping stays real."""

    def __init__(self, capacity: int, records: list):
        self.capacity, self.records = capacity, records

    def can_batch(self, rows):
        return len(rows) <= self.capacity

    def forward(self, row, inputs, request):
        self.records.append((tuple([row.identity]), inputs.count, request.committed_inputs))
        return row.identity

    def forward_batch(self, rows, inputs, request):
        self.records.append(
            (tuple(row.identity for row in rows), inputs[0].count, request.committed_inputs)
        )
        return tuple(row.identity for row in rows)


def ready_assembly(component=None, *, rows=32, capacity=4, **record):
    from magnitude_engine.generation.execution import execute
    from magnitude_engine.models.inputs import ModelInputs
    from magnitude_engine.models.operations import Forward
    from magnitude_engine.models.runtime import ForwardRequest

    bound = bind_operation(component or execute, "BATCHING:ASSEMBLY:MAG:READY_COMPATIBLE")
    with recording(
        bound,
        benchmark="control.ready_assembly",
        workload={"rows": rows, "capacity": capacity},
        boundary="ready-round-zero-work-dispatch",
        **record,
    ) as run:
        if min(rows, capacity) < 1:
            raise ValueError("positive assembly geometry required")
        records = []
        runtimes = tuple(DescriptorRuntime(capacity, records) for _ in range(2))
        states = tuple(DescriptorState(runtimes[i % 2], i) for i in range(rows))
        widths = tuple(1 if (i // 2) % 2 else 4 for i in range(rows))
        inputs = tuple(ModelInputs.from_tokens((1,) * width) for width in widths)
        compatibility = tuple((i % 2, width) for i, width in enumerate(widths))
        counts = {key: compatibility.count(key) for key in set(compatibility)}
        minimum = sum((count + capacity - 1) // capacity for count in counts.values())

        def continuation(state, inputs):
            return (yield Forward(cast(Any, state), inputs, ForwardRequest()))

        def invoke():
            return bound.instance(
                tuple(
                    continuation(state, value) for state, value in zip(states, inputs, strict=True)
                )
            )

        def validate(outputs):
            seen = []
            for indices, width, committed in records:
                if not indices or len(indices) > capacity or committed != 0:
                    raise ValueError("assembly violated group capacity/commitment")
                if any(not 0 <= index < rows for index in indices):
                    raise ValueError("assembly produced unknown continuation")
                keys = {compatibility[i] for i in indices}
                if len(keys) != 1 or next(iter(keys))[1] != width:
                    raise ValueError("assembly mixed incompatible operations")
                seen.extend(indices)
            if sorted(seen) != list(range(rows)) or tuple(o.result for o in outputs) != tuple(
                range(rows)
            ):
                raise ValueError("assembly lost or duplicated work")
            return Observation(
                digest(records),
                {
                    "ready_rows": rows,
                    "dispatch_groups": len(records),
                    "minimum_legal_groups": minimum,
                },
                {"groups": records},
            )

        run.measure(invoke, prepare=records.clear, validate=validate, deterministic=True)
    return run
