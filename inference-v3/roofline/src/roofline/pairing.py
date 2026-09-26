"""Fresh paired execution of safely replaceable authored operation revisions."""

from contextlib import ExitStack
from pathlib import Path

from .contracts import Measurement, identity
from .store import atomic_write


def replaceable(path):
    return (
        path.startswith("src/ops/kernels/")
        and path.endswith(".py")
        or path == "src/ops/compiler/streaming.py"
    )


def changed_paths(first, second):
    left, right = ({f.path: f.blob for f in s.files} for s in (first, second))
    return {p for p in left.keys() | right.keys() if left.get(p) != right.get(p)}


def install(source, store, root, changed):
    files = {f.path: f for f in source.files}
    for path in changed:
        if path in files:
            atomic_write(root / path, store.blob(files[path].blob))
        else:
            (root / path).unlink(missing_ok=True)
    from ops.lab.refresh import OperationSources

    OperationSources().refresh()


def paired(request, attempt, store, *, owner=None):
    from ops.lab.comparison import compare_prepared
    from ops.lab.evidence import ExecutionContext, Model, Workload

    from .integrations.magnitude import Magnitude, measurement_protocol

    if request.experiment.engine != "magnitude":
        raise NotImplementedError("source pairing applies to the Magnitude integration")
    if request.experiment.protocol.samples < 2:
        raise ValueError("paired measurements require at least two sample blocks")
    candidate_source = store.source(request.experiment.source)
    baseline_source = store.source(request.experiment.against_source)
    changed = changed_paths(baseline_source, candidate_source)
    if any(not replaceable(p) for p in changed):
        raise NotImplementedError(
            "pairing is unavailable across runtime, formula or compiler changes"
        )
    import ops.kernels

    root = Path(ops.kernels.__file__).resolve().parents[3]
    baseline_request = request.model_copy(
        update={
            "experiment": request.experiment.model_copy(
                update={"source": baseline_source.source_id, "against_source": None}
            )
        }
    )
    candidate_request = request.model_copy(
        update={"experiment": request.experiment.model_copy(update={"against_source": None})}
    )
    results = []
    try:
        install(baseline_source, store, root, changed)
        with ExitStack() as owned:
            baseline = owner or Magnitude(baseline_request, attempt, store)
            owned.callback(baseline.close)
            if owner is not None:
                baseline.request, baseline.attempt = baseline_request, attempt
                baseline.experiment = baseline_request.experiment
                baseline.refresh()

            def finish():
                if any(m.status != "complete" for m in results):
                    return results, None
                baseline.request = candidate_request
                baseline.experiment = candidate_request.experiment
                baseline.refresh()
                owned.pop_all()
                return results, baseline

            if "/" not in request.experiment.scope:
                a = baseline.measure()[0]
                install(candidate_source, store, root, changed)
                candidate = Magnitude(candidate_request, attempt, store, shared=baseline)
                owned.callback(candidate.close)
                b = candidate.measure()[0]
                samples = {"baseline": [], "candidate": []}
                owners = {"baseline": baseline, "candidate": candidate}
                pairs = tuple(identity() for _ in range(request.experiment.protocol.samples))
                for i in range(len(pairs)):
                    for name in (
                        ("baseline", "candidate") if i % 2 == 0 else ("candidate", "baseline")
                    ):
                        samples[name].append(owners[name].workload_sample())
                for name, m in (("baseline", a), ("candidate", b)):
                    results.append(
                        m.model_copy(
                            update={"samples_seconds": tuple(samples[name]), "pair_ids": pairs}
                        )
                    )
                candidate.close()
                return finish()
            # The prepared handles and immutable weights share one runtime, while
            # each source produces its own logical component inputs before pairing.
            position = request.experiment.step
            if position is None:
                positions = (
                    range(request.experiment.steps)
                    if request.experiment.scope.startswith("decode/")
                    else range(0, request.experiment.context, baseline.prefill_rows)
                )
            else:
                positions = (position,)
            for position in positions:
                install(baseline_source, store, root, changed)
                baseline.model.program.close()
                from engine.models.qwen35.tensor_program import TensorProgram

                baseline.model.program = TensorProgram(
                    baseline.description, baseline.device, baseline.weights
                )
                first, scopes = baseline.prepare_component(position)
                with ExitStack() as prepared:
                    prepared.callback(first.close)
                    install(candidate_source, store, root, changed)
                    candidate = Magnitude(candidate_request, attempt, store, shared=baseline)
                    prepared.callback(candidate.close)
                    second, candidate_scopes = candidate.prepare_component(position)
                    prepared.callback(second.close)
                    if first.fixture.identity != second.fixture.identity:
                        raise ValueError(
                            "producer changes yield different boundary inputs; "
                            "same-input pairing is unavailable"
                        )
                    context = ExecutionContext(
                        model=Model(
                            identity=request.experiment.model, label=request.experiment.model
                        ),
                        workload=Workload(
                            kind="benchmark",
                            recipe={"position": position},
                            realization=first.fixture.identity,
                        ),
                        engine="magnitude",
                        artifact=str(baseline.format.identity),
                        numerical_contract="production",
                        hardware=baseline.device.evidence_identity,
                        implementation="resolved-per-candidate",
                    )
                    previous_runs = {
                        run.identity for run in baseline.native_store.runs(request.experiment.model)
                    }
                    try:
                        runs = compare_prepared(
                            {"baseline": first, "candidate": second},
                            context=context,
                            store=baseline.native_store,
                            protocol=measurement_protocol(request.experiment),
                            blocks=request.experiment.protocol.samples,
                        )
                    except Exception:
                        # Ops publishes partial/failing evidence before propagating
                        # a numerical or execution error. Preserve those observations.
                        published = {
                            run.attachments["candidate"]: run
                            for run in baseline.native_store.runs(request.experiment.model)
                            if run.identity not in previous_runs
                        }
                        if set(published) != {"baseline", "candidate"}:
                            raise
                        runs = (published["baseline"], published["candidate"])
                    for owner, run, hierarchy in zip(
                        (baseline, candidate), runs, (scopes, candidate_scopes), strict=True
                    ):
                        m = baseline.native_store.measurement(run.measurements[0])
                        artifacts = {
                            name: store.put_blob(baseline.native_store.artifact(blob))
                            for name, blob in m.artifacts.items()
                        }
                        artifacts["ops-measurement"] = store.put_blob(m.model_dump_json().encode())
                        if owner.characterization_artifact:
                            artifacts["resource-characterization"] = owner.characterization_artifact
                        results.append(
                            Measurement(
                                **owner.base(scope=request.experiment.scope, position=position),
                                status="complete" if m.outcome.value == "complete" else "failed",
                                correctness=run.correctness,
                                samples_seconds=tuple(s.elapsed_ns / 1e9 for s in m.samples),
                                pair_ids=tuple(run.protocol["pair_ids"][: len(m.samples)]),
                                scopes=hierarchy,
                                artifacts=artifacts,
                                details={
                                    "boundary": "isolated-complete-operation",
                                    "comparison": m.series.model_dump(
                                        mode="json", exclude={"protocol", "device"}
                                    ),
                                },
                                error=m.error,
                            )
                        )
            return finish()
    finally:
        install(candidate_source, store, root, changed)
