"""KV writes/branches and actual model-state restoration, with shared lifecycle helpers."""

from contextlib import ExitStack

from magnitude_engine import components as c
from performance.assembly import Binding, inspect_engine, inspect_state
from performance.benchmarks.fixtures import record_inputs, tokens
from performance.benchmarks.model import prefix
from performance.records import Observation, digest
from performance.runner import recording


def append(component, *, prefix_tokens, append_tokens=1, granularity="runs", **record):
    import mlx.core as mx

    if isinstance(component, Binding):
        bound = component
    else:
        assembly = inspect_state(component)
        bound = assembly.at(assembly.graph.root)
    actual = bound.instance
    arena = actual if hasattr(actual, "allocator") else actual.arena
    from magnitude_engine.models.state.pages import PageStore

    store = actual if isinstance(actual, PageStore) else PageStore(arena)
    workload = {
        "prefix_tokens": prefix_tokens,
        "append_tokens": append_tokens,
        "retained_positions": prefix_tokens + append_tokens,
        "granularity": granularity,
    }
    with (
        recording(bound, benchmark="state.append", workload=workload, **record) as run,
        ExitStack() as life,
    ):
        if prefix_tokens < 1 or append_tokens < 1 or granularity not in ("runs", "pages"):
            raise ValueError("invalid KV append workload")
        row = store.create()
        life.callback(row.close)
        life.callback(arena.complete)
        row.reserve(prefix_tokens + append_tokens)
        payloads = []
        for i, geometry in enumerate(arena.layers):
            row.write(
                i,
                0,
                mx.full((geometry.heads, prefix_tokens, geometry.key_width), 0.25, arena.dtype),
                mx.full((geometry.heads, prefix_tokens, geometry.value_width), -0.25, arena.dtype),
            )
            payloads.append(
                (
                    mx.full((geometry.heads, append_tokens, geometry.key_width), 2, arena.dtype),
                    mx.full((geometry.heads, append_tokens, geometry.value_width), -2, arena.dtype),
                )
            )
        row.commit(prefix_tokens)
        mx.eval(payloads)
        arena.complete()

        def reset():
            arena.complete()
            row.trim(prefix_tokens)
            row.reserve(prefix_tokens + append_tokens)

        def execute():
            for layer, (keys, values) in enumerate(payloads):
                offset = 0
                while offset < append_tokens:
                    count = append_tokens - offset
                    if granularity == "pages":
                        count = min(
                            count, arena.page_size - (prefix_tokens + offset) % arena.page_size
                        )
                    row.write(
                        layer,
                        prefix_tokens + offset,
                        keys[:, offset : offset + count],
                        values[:, offset : offset + count],
                    )
                    offset += count
            row.commit(prefix_tokens + append_tokens)

        def validate(_):
            sums = []
            for i in range(len(arena.layers)):
                keys, values = row.read(i)
                expected = mx.concatenate(
                    [
                        mx.full((prefix_tokens,), 0.25, arena.dtype),
                        mx.full((append_tokens,), 2, arena.dtype),
                    ]
                )[None, :, None]
                if not mx.all(keys == expected).item() or not mx.all(values == -expected).item():
                    raise ValueError("append changed prefix or appended KV values")
                sums.append(keys.sum().item())
            physical = sum(a.nbytes for a in (*arena.keys, *arena.values))
            return Observation(
                digest(sums),
                {"physical_kv_bytes": physical},
                metrics={"MEM": physical} if bound.node.component == c.KV_STORE else {},
            )

        run.measure(
            execute,
            prepare=reset,
            complete=lambda _: arena.complete(),
            validate=validate,
            deterministic=True,
            dimension="EXEC" if bound.node.component == "KV:APPEND" else None,
        )
    return run


def branches(component, *, prefix_tokens, branch_tokens=1, branches=2, **record):
    import mlx.core as mx

    bound = (
        component if isinstance(component, Binding) else inspect_state(component).at("state.branch")
    )
    store = bound.instance
    arena = store.arena
    workload = {
        "prefix_tokens": prefix_tokens,
        "branch_tokens": branch_tokens,
        "branches": branches,
    }
    with (
        recording(bound, benchmark="state.branch", workload=workload, **record) as run,
        ExitStack() as life,
    ):
        if min(prefix_tokens, branches) < 1 or branch_tokens < 0:
            raise ValueError("invalid branch geometry")
        source = store.create()
        life.callback(source.close)

        def write(row, count, value):
            row.reserve(row.length + count)
            for i, g in enumerate(arena.layers):
                row.write(
                    i,
                    row.length,
                    mx.full((g.heads, count, g.key_width), value, arena.dtype),
                    mx.full((g.heads, count, g.value_width), -value, arena.dtype),
                )
            row.commit(row.length + count)

        write(source, prefix_tokens, 1)
        arena.complete()
        checkpoint = source.checkpoint()
        life.callback(checkpoint.close)
        source.close()
        rows = []

        def reset():
            arena.complete()
            for row in rows:
                row.close()
            rows.clear()

        life.callback(reset)

        def execute():
            for i in range(branches):
                row = store.create(checkpoint)
                rows.append(row)
                if branch_tokens:
                    write(row, branch_tokens, i + 2)

        def validate(_):
            store.validate()
            outputs = []
            for index, row in enumerate(rows):
                for layer in range(len(arena.layers)):
                    keys, values = row.read(layer)
                    expected = mx.array(
                        [1] * prefix_tokens + [index + 2] * branch_tokens, dtype=arena.dtype
                    )[None, :, None]
                    if (
                        not mx.all(keys == expected).item()
                        or not mx.all(values == -expected).item()
                    ):
                        raise ValueError("branch changed shared prefix or branch payload")
                outputs.append(row.length)
            return Observation(digest(outputs), dict(arena.counters))

        run.measure(
            execute,
            prepare=reset,
            complete=lambda _: arena.complete(),
            validate=validate,
            deterministic=True,
        )
    return run


def restore(
    engine,
    *,
    context_tokens,
    advance_tokens=0,
    accepted_tokens=0,
    fixture="prose.moby-dick",
    prompt=None,
    continuation=None,
    **record,
):
    import mlx.core as mx

    from magnitude_engine.models.runtime import ForwardRequest
    from performance.benchmarks.numerics import compare

    assembly = inspect_engine(engine)
    bound = assembly.at(assembly.graph.nodes["target"].dependencies["state"])
    model = engine.engine.generation.model
    mode = "saved_boundary" if not advance_tokens else "accepted_prefix"
    workload = {
        "context_tokens": context_tokens,
        "advanced_tokens": advance_tokens,
        "accepted_tokens": accepted_tokens,
        "restore_mode": mode,
        "fixture": fixture,
        "retained_positions": context_tokens + accepted_tokens,
        "retained_rows": 1 + int(accepted_tokens > 0),
        "memory_boundary": "current state and restore checkpoint; shared backing once",
    }
    with recording(bound, benchmark="state.restore", workload=workload, **record) as run:
        if not 0 <= accepted_tokens <= advance_tokens:
            raise ValueError("acceptance outside advance")
        provenance = {}
        if prompt is None:
            prepared = tokens(
                engine.properties["target_path"],
                fixture=fixture,
                context_tokens=context_tokens,
                continuation_tokens=max(1, advance_tokens),
            )
            prompt, continuation, provenance = (
                prepared.prompt,
                prepared.continuation,
                prepared.provenance,
            )
        continuation = tuple(continuation or ())
        if len(continuation) < advance_tokens:
            raise ValueError("insufficient continuation for restoration")

        def logical(state):
            pages = getattr(state, "pages", None)
            if pages is not None:
                return tuple(
                    a for i in range(len(pages.store.arena.layers)) for a in pages.read(i)
                ) + tuple(a for slot in state.slots for a in slot.values)
            if hasattr(state, "read"):
                return tuple(a for i in range(len(state.store.arena.layers)) for a in state.read(i))
            return tuple(model.states.arrays(state))

        record_inputs(run, prompt=tuple(prompt), continuation=continuation[:advance_tokens])
        with prefix(model, prompt) as checkpoint:
            oracle = model.create(checkpoint)
            try:
                if accepted_tokens:
                    step = model.forward(
                        oracle,
                        continuation[:accepted_tokens],
                        ForwardRequest(logits=False, committed_inputs=accepted_tokens),
                    )
                    step.accept(accepted_tokens)
                    step.complete()
                expected = tuple(mx.array(a) for a in logical(oracle.state))
                mx.eval(expected)
            finally:
                oracle.close()
            row = advance = None

            def reset():
                nonlocal row, advance
                if row is not None:
                    row.close()
                row = None
                if advance_tokens:
                    row = model.create(checkpoint)
                    advance = model.forward(
                        row, continuation[:advance_tokens], ForwardRequest(logits=False)
                    )
                    advance.complete()

            def execute():
                nonlocal row
                if advance_tokens:
                    assert advance is not None
                    advance.accept(accepted_tokens)
                else:
                    row = model.create(checkpoint)
                assert row is not None
                row.complete_committed()

            def validate(_):
                assert row is not None
                actual = logical(row.state)
                observed = compare(actual, expected)
                metrics = {}
                if hasattr(row.state, "pages") and hasattr(row.state, "recurrent"):
                    arena = row.state.pages.store.arena
                    images = {
                        id(r.image): r.image for r in (row.state.recurrent, checkpoint.recurrent)
                    }
                    physical = sum(a.nbytes for a in (*arena.keys, *arena.values))
                    physical += sum(
                        a.nbytes
                        for image in images.values()
                        for i in range(len(image.layouts))
                        for a in image.read(i)
                    )
                    metrics["MEM"] = physical
                return Observation(
                    observed.output_digest,
                    {"retained_arrays": len(actual)},
                    {"fixture": provenance, **observed.evidence},
                    metrics=metrics,
                )

            try:
                run.measure(
                    execute,
                    prepare=reset,
                    complete=lambda _: model.owner.backend.drain(),
                    validate=validate,
                    deterministic=True,
                    dimension=None if bound.node.component == c.KV_STORE else "RESTORE",
                )
            finally:
                if row is not None:
                    row.close()
    return run


def page_cases(store, *, contexts=(4096, 16384, 65536), **record):
    graph = inspect_state(store)
    return [
        append(graph.at(graph.graph.root), prefix_tokens=context, **record) for context in contexts
    ]


def recurrent_image(image, *, live_rows=1, checkpoints=4, **record):
    """Retained physical image versus required logical rows; saved-reference acquisition."""
    import mlx.core as mx

    from performance.assembly import bind_operation

    bound = bind_operation(
        image,
        c.Implementation(c.RECURRENT_STATE, c.Source.MAG, "CHECKPOINTED"),
        parameters=c.RecurrentStorage(
            layouts=tuple(
                tuple(
                    c.TensorFacts(
                        identity=f"recurrent.{i}.{j}",
                        shape=t.shape,
                        bytes=t.nbytes,
                        dtype=str(t.dtype),
                    )
                    for j, t in enumerate(layer.tensors)
                )
                for i, layer in enumerate(image.layouts)
            )
        ),
    )
    workload = {
        "retained_rows": live_rows,
        "checkpoints": checkpoints,
        "restore_mode": "saved_boundary",
    }
    with (
        recording(bound, benchmark="state.recurrent_image", workload=workload, **record) as run,
        ExitStack() as life,
    ):
        if not 1 <= live_rows <= image.width or checkpoints < 1:
            raise ValueError("invalid recurrent row retention")
        rows = [image.acquire(i) for i in range(image.width)]
        for row in rows:
            life.callback(row.close)
        for i, layout in enumerate(image.layouts):
            image.write(
                i,
                tuple(
                    mx.full((image.width, *t.shape[1:]), i + 0.25, t.dtype) for t in layout.tensors
                ),
            )
        mx.eval(*[a for i in range(len(image.layouts)) for a in image.read(i)])
        for departed in rows[live_rows:]:
            departed.close()
        retained = []

        def reset():
            for row in retained:
                row.close()
            retained.clear()

        life.callback(reset)

        def execute():
            retained.extend(row.acquire() for row in rows[:live_rows] for _ in range(checkpoints))

        def validate(_):
            for row in retained:
                for i in range(len(image.layouts)):
                    if any(not mx.all(a == i + 0.25).item() for a in row.values(i)):
                        raise ValueError("restored recurrent row changed")
            size = sum(a.nbytes for i in range(len(image.layouts)) for a in image.read(i))
            return Observation(
                digest((live_rows, len(retained))),
                {"physical_state_bytes": size},
                metrics={"MEM": size},
            )

        run.measure(
            execute, prepare=reset, validate=validate, deterministic=True, dimension="RESTORE"
        )
    return run


def checkpoint(component, saved, *, retained_shapes, **record):
    """Restore an actual native checkpoint; score its sole retained backing.

    The caller supplies the logically required representation, which may be
    smaller than the cache's allocated capacity. Preparation owns the checkpoint.
    """
    import mlx.core as mx

    from performance.benchmarks.numerics import compare

    if isinstance(component, Binding):
        bound = component
    else:
        assembly = inspect_state(component)
        bound = assembly.at(assembly.graph.root)
    store = bound.instance
    workload = {
        "retained_shapes": retained_shapes,
        "restore_mode": "saved_boundary",
        "memory_boundary": "sole checkpoint backing before restoration",
    }
    with recording(bound, benchmark="state.checkpoint", workload=workload, **record) as run:
        owners = {id(s.owner): s.nbytes for s in saved.retained_storage()}
        backing = sum(owners.values())
        oracle = store.create(saved)
        try:
            expected = tuple(mx.array(a) for a in store.arrays(oracle))
            mx.eval(expected)
        finally:
            store.release(oracle)
        row = None

        def reset():
            nonlocal row
            if row is not None:
                store.release(row)
            row = None

        def execute():
            nonlocal row
            row = store.create(saved)
            return store.arrays(row)

        def validate(values):
            observed = compare(values, expected)
            return Observation(observed.output_digest, observed.counters, metrics={"MEM": backing})

        try:
            run.measure(
                execute,
                prepare=reset,
                complete=mx.eval,
                validate=validate,
                deterministic=True,
                dimension="RESTORE",
            )
        finally:
            reset()
    return run
