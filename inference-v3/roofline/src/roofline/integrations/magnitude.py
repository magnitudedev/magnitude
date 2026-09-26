"""Production Qwen workloads and typed Ops formula measurements."""

from __future__ import annotations

import asyncio
import hashlib
import platform
import time
from collections import OrderedDict
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Any

from roofline.contracts import Measurement, Scope, digest, encoded, identity, now


def numerical_protocol(experiment):
    """Existing qualification gates, fixed before execution and persisted with evidence."""
    # The connected hybrid-model gate uses 0.08/0.08. Artifact-backed isolated
    # FFN qualification uses 0.003/0.0078125. Neither adapts to observed errors.
    if "/" not in experiment.scope:
        return {
            "policy": "qwen35-production-operations-v2",
            "absolute_tolerance": 0.08,
            "relative_tolerance": 0.08,
        }
    return {
        "policy": "qwen35-component-v1",
        "absolute_tolerance": 0.003,
        "relative_tolerance": 0.0078125,
    }


def measurement_protocol(experiment):
    from ops.lab.records import MeasurementProtocol

    numerical = numerical_protocol(experiment)
    return MeasurementProtocol(
        samples=experiment.protocol.samples,
        warmups=experiment.protocol.warmups,
        inputs="production",
        kernel_limit=None,
        absolute_tolerance=numerical["absolute_tolerance"],
        relative_tolerance=numerical["relative_tolerance"],
    )


def tree_scopes(tree, phase):
    paths, counts, result = {}, {}, []
    for target in tree:
        parent = paths.get(target.call.parent, phase)
        key = (parent, target.definition.id)
        index = counts.get(key, 0)
        counts[key] = index + 1
        selector = f"{parent}/{target.definition.id}[{index}]"
        paths[target.call.occurrence] = selector
        result.append(
            Scope(
                selector=selector,
                parent=parent,
                contract=target.definition.id,
                semantics=target.semantic_identity if target.call.complete else None,
                occurrence=target.call.occurrence,
                complete=target.call.complete,
                reason=None if target.call.complete else "incomplete production boundary",
            )
        )
    return tuple(result)


def artifact_path(model, target_name, model_id):
    if model_id.split(":")[1] != "gguf":
        raise NotImplementedError("integration requires one supported GGUF artifact")
    return Path(model.locations[target_name])


def verify_artifact(model, target_name, model_id):
    path = artifact_path(model, target_name, model_id)
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != model.sha256:
        raise ValueError(f"artifact checksum mismatch: {path}")
    return path


class Magnitude:
    def __init__(self, request, attempt, store, *, shared=None):
        import ops
        from engine import DevicePlan
        from engine.models.qwen35.formats.gguf import describe
        from engine.models.qwen35.runtime import DenseRuntime
        from engine.weights.formats.gguf import GGUFFormat
        from engine.weights.tensor_residency import TensorWeights
        from ops.lab.refresh import OperationSources
        from ops.lab.store import ObservationStore

        self._closed = False
        self.operation_sources = OperationSources()
        self.operation_sources.refresh()
        self.production_dependencies = set()
        self.request, self.attempt, self.store = request, attempt, store
        self.experiment = request.experiment
        assert request.model is not None
        self.prefill_rows = min(512, max(2, self.experiment.context))
        self.owns_runtime = shared is None
        if shared is None:
            self.path = artifact_path(request.model, attempt.target_name, request.experiment.model)
            self.format = GGUFFormat(str(self.path))
            if str(self.format.identity) != request.model.sha256:
                self.format.close()
                raise ValueError(f"artifact checksum mismatch: {self.path}")
            self.description = describe(self.format)
            self.device = ops.DeviceRuntime.open(
                DevicePlan.discover(
                    backend=attempt.target.device.backend,
                    maximum_bytes=attempt.target.device.maximum_bytes or (12 << 30),
                    ordinal=attempt.target.device.index,
                )
            )
            self.weights = TensorWeights(self.format, self.device)
        else:
            self.path, self.format = shared.path, shared.format
            self.description, self.device, self.weights = (
                shared.description,
                shared.device,
                shared.weights,
            )
        self.device.load_capacity_evidence(attempt.target.capacities)
        capacity = self.experiment.context + self.experiment.steps
        self.model = DenseRuntime(
            self.description,
            self.device,
            self.weights,
            max_sequences=1,
            prefill_rows=self.prefill_rows,
            context_capacity=capacity,
        )
        self.native_store = (
            ObservationStore(store.root / "ops.sqlite") if shared is None else shared.native_store
        )
        from ops.lab.characterization import Characterization

        self.characterization_artifact = None
        for recorded in reversed(store.records("characterization")):
            profile = Characterization.model_validate(recorded["profile"])
            if (
                profile.device == self.device.evidence_identity
                and profile.compiler == self.device.compiler_identity
                and profile.compiler_target == self.device.compiler_target.identity
            ):
                self.device.load_characterization(profile)
                self.characterization_artifact = recorded["artifact_id"]
                break
        self.runners = OrderedDict()
        self.prepared_tokens = None
        self.hardware = {
            "host": platform.node(),
            "system": platform.platform(),
            "device": self.device.evidence_identity,
            "compiler": self.device.compiler_identity,
        }

    def close(self):
        if self._closed:
            return
        for runner in self.runners.values():
            runner.close()
        self.model.close()
        if self.owns_runtime:
            self.weights.close()
            self.device.close()
            self.format.close()
            self.native_store.close()
        self._closed = True

    def tokens(self):
        if self.prepared_tokens is None:
            from benchmark_fixtures.preparation import Fixture, Tokenization, prepare

            self.prepared_tokens = asyncio.run(
                prepare(
                    Fixture(
                        identity="prose.moby-dick",
                        context_tokens=self.experiment.context,
                        continuation_tokens=self.experiment.steps,
                    ),
                    Tokenization(self.path),
                )
            )
        return self.prepared_tokens

    def decode_weight(self, value):
        import gguf
        import numpy as np
        from engine.models.qwen35.tensor_program import weight_roles
        from engine.weights.descriptor import WeightTransform
        from engine.weights.formats.gguf import Encoding
        from ops.tensor.primitive import round_reference

        roles = {role.name: role for role, _ in weight_roles(self.description)}
        entry = self.format.directory.tensor(value.name)
        content = self.format.source.read(
            self.format.directory.data_offset + entry.offset, entry.nbytes
        )
        if entry.encoding in (Encoding.F32, Encoding.F16):
            result = (
                np.frombuffer(
                    content, dtype=np.float32 if entry.encoding == Encoding.F32 else np.float16
                )
                .astype(np.float32)
                .reshape(entry.shape)
            )
            if roles[value.name].transform == WeightTransform.NEGATIVE_EXP:
                result = -np.exp(result)
        else:
            result = gguf.dequantize(
                np.frombuffer(content, dtype=np.uint8),
                gguf.GGMLQuantizationType(int(entry.encoding)),
            ).reshape(entry.shape)
            # Packed weights denote their decoded coefficient values. The port's
            # activation dtype does not add a dense BF16 materialization that the
            # production packed contraction never performs.
            if value.spec.representation is not None:
                return result
        return round_reference(result, value.spec.dtype)

    def forward(self, sequence, tokens, *, commit, completed=None):
        from engine.data import TokenId
        from engine.models.sequence import LogitsSelection, ModelRequest

        batch = self.model.prepare(
            (
                ModelRequest(
                    sequence, tuple(map(TokenId, tokens)), LogitsSelection.LAST, (0, 0, 0, 0, 0, 0)
                ),
            )
        )
        try:
            batch.completion.wait()
            if completed is not None:
                completed(batch)
            if commit:
                batch.advances[0].commit()
        finally:
            batch.close()

    def sequence(self):
        from engine.data import TokenId
        from engine.models.qwen35.inputs import InputPlan

        return self.model.create(InputPlan.text(tuple(map(TokenId, self.tokens().prompt))))

    def prefill(self, sequence):
        tokens = self.tokens().prompt
        for start in range(0, len(tokens), self.prefill_rows):
            self.forward(sequence, tokens[start : start + self.prefill_rows], commit=True)

    def runner(self, sequence, tokens, position):
        import ops
        from engine.models.qwen35.inspection import inspect_forwards
        from ops.lab.runner import MeasurementRunner

        # Each coordinate has an immutable logical starting state. It is reused
        # across repetitions, while the runner resets writable resources per sample.
        key = (self.experiment.scope, position)
        if key not in self.runners:
            captures = []

            def capture(invocation):
                self.production_dependencies.update(
                    (d.module, d.symbol) for d in invocation.compiled.code_dependencies
                )
                captures.append(invocation.fixture(self.decode_weight))

            with inspect_forwards(self.model, captured=capture):
                self.forward(sequence, tokens, commit=False)
            if len(captures) != 1:
                raise RuntimeError("expected one production invocation for this scope")
            fixture = captures[0]
            self.runners[key] = MeasurementRunner(
                fixture,
                self.device,
                self.native_store,
                ops.CompileOptions(mode=self.experiment.scope.split("/")[0], precision="model"),
                protocol=measurement_protocol(self.experiment),
                # At most two runners share a bounded host-reference allowance.
                # MoE boundary weights exceed the old fixed 512 MiB cache.
                reference_bytes=min(
                    4 << 30, (self.attempt.target.device.maximum_bytes or (12 << 30)) // 8
                ),
            )
            if "component" in self.request.inputs:
                from ops.formula import FormulaTree

                tree = FormulaTree(fixture.root)
                scopes = tree_scopes(tree, self.experiment.scope.split("/")[0])
                selected = next(s for s in scopes if s.selector == self.experiment.scope)
                target = next(t for t in tree if t.call.occurrence == selected.occurrence)
                path = self.store.blob_path(self.request.inputs["component"])
                provenance = self.shared_boundary()["provenance"]
                expected = {
                    "source_id": self.experiment.input_source,
                    "target": self.experiment.input_target,
                    "model_artifact": self.request.model.sha256,
                    "scope": self.experiment.scope,
                    "step": self.experiment.step,
                    "tokens": digest(
                        {"prompt": self.tokens().prompt, "continuation": self.tokens().continuation}
                    ),
                }
                if any(provenance.get(k) != v for k, v in expected.items()):
                    raise ValueError(
                        "shared boundary provenance differs from the requested producer or workload"
                    )
                self.runners[key].restore(target, path)
            while len(self.runners) > 2:
                _, retired = self.runners.popitem(last=False)
                retired.close()
        return self.runners[key]

    def shared_boundary(self):
        import json
        import zipfile

        with zipfile.ZipFile(self.store.blob_path(self.request.inputs["component"])) as archive:
            return json.loads(archive.read("manifest.json"))

    def base(self, *, scope, position=None) -> dict[str, Any]:
        tokens = self.tokens()
        return dict(
            measurement_id=identity(),
            request_id=self.request.request_id,
            attempt_id=self.attempt.attempt_id,
            created=now(),
            model=self.request.experiment.model,
            artifact=self.request.model.sha256,
            source_id=self.request.experiment.source,
            target=self.attempt.target_name,
            engine="magnitude",
            workload={
                "workload": "prose",
                "context": self.experiment.context,
                "steps": self.experiment.steps,
                "step": position,
                "tokens": digest({"prompt": tokens.prompt, "continuation": tokens.continuation}),
                "provenance": tokens.provenance,
                **(
                    {
                        "input_policy": {
                            "kind": "frozen-shared-boundary",
                            "source_id": self.experiment.input_source,
                            "target": self.experiment.input_target,
                            "boundary": self.shared_boundary()["boundary"],
                        }
                    }
                    if "component" in self.request.inputs
                    else {}
                ),
            },
            scope=scope,
            hardware=self.hardware,
            protocol={
                **self.experiment.protocol.model_dump(exclude={"deadline_seconds"}),
                "numerical": numerical_protocol(self.experiment),
            },
        )

    def component(self, sequence, tokens, position):
        from ops.formula import FormulaTree

        runner = self.runner(sequence, tokens, position)
        base = self.base(scope=self.experiment.scope, position=position)
        scopes = tree_scopes(FormulaTree(runner.fixture.root), self.experiment.scope.split("/")[0])
        selected = next((s for s in scopes if s.selector == self.experiment.scope), None)
        if selected is None or not selected.complete:
            raise ValueError("scope is absent or unsupported in this production trace; use scopes")
        target = next(
            t for t in FormulaTree(runner.fixture.root) if t.call.occurrence == selected.occurrence
        )
        measured = runner.measure(target).measurement
        artifacts = {
            name: self.store.put_blob(self.native_store.artifact(artifact))
            for name, artifact in measured.artifacts.items()
        }
        artifacts["ops-measurement"] = self.store.put_blob(measured.model_dump_json().encode())
        if self.characterization_artifact:
            artifacts["resource-characterization"] = self.characterization_artifact
        unavailable = [u.reason for u in measured.unavailable]
        diagnostic = diagnostic_program = None
        try:
            diagnostic, diagnostic_program = runner.diagnose(target, kernel_limit=2048)
            artifacts["isolated-native-observation"] = self.store.put_blob(
                encoded(asdict(diagnostic))
            )
        except Exception as error:
            unavailable.append(f"Separate native observation unavailable: {error}")
        analysis = measured.model_dump(mode="json", include={"roofline", "ceilings", "quantities"})
        from formula_performance.records import Publication
        from formula_performance.records import identity as fingerprint
        from ops.performance.publication import hardware as hardware_binding

        publication = Publication.model_validate_json(
            self.store.blob(artifacts["formula-performance"])
        )
        system = hardware_binding(
            self.device, capacities=self.attempt.target.capacities
        ).model_copy(update={"label": self.attempt.target_name})
        publication = publication.model_copy(
            update={
                "hardware": (system,),
                "observations": tuple(
                    o.model_copy(
                        update={
                            "hardware": fingerprint(system),
                            "implementation": self.experiment.source,
                            "evidence": (base["measurement_id"], measured.identity),
                            "coordinates": {
                                "artifact": self.request.model.sha256,
                                "context": self.experiment.context,
                                "position": position,
                                "phase": self.experiment.scope.split("/")[0],
                                "inputs": measured.preparation["inputs"],
                                "input_boundary": measured.preparation["boundary"],
                            },
                        }
                    )
                    for o in publication.observations
                ),
            }
        )
        if diagnostic is not None:
            from ops.performance.publication import capture

            original = publication.observations[0]
            captured = capture(
                publication.manifests[0],
                diagnostic,
                capture_id=base["measurement_id"] + ":native",
                execution_graph=diagnostic_program.graph.fingerprint,
                component=original.component,
            )
            if captured is not None:
                instrumented = original.model_copy(
                    update={
                        "identity": base["measurement_id"] + ":diagnostic",
                        "boundary": "instrumented-isolated-operation",
                        "samples": (diagnostic.elapsed_ns / 1e9,),
                        "captures": (captured.identity,),
                    }
                )
                publication = publication.model_copy(
                    update={
                        "captures": (*publication.captures, captured),
                        "observations": (*publication.observations, instrumented),
                    }
                )
        # Validate the complete closure before making the artifact queryable.
        publication = Publication.model_validate(publication.model_dump())
        artifacts["formula-performance"] = self.store.put_blob(
            publication.model_dump_json().encode()
        )
        return Measurement(
            **base,
            status="complete" if measured.outcome.value == "complete" else "failed",
            correctness=(
                "passed"
                if measured.checked
                else "failed"
                if (measured.error or "").startswith("NumericalMismatch:")
                else "unchecked"
            ),
            samples_seconds=tuple(s.elapsed_ns / 1e9 for s in measured.samples),
            scopes=scopes,
            artifacts=artifacts,
            error=measured.error,
            costs={str(p.phase): p.elapsed_ns / 1e9 for p in measured.phases},
            details={
                "semantics": selected.semantics,
                "comparison": measured.series.model_dump(
                    mode="json", exclude={"protocol", "device"}
                ),
                "boundary": "isolated-complete-operation",
                "preparation": {
                    **measured.preparation,
                    **(
                        {"inputs": "frozen-shared-boundary"}
                        if "component" in self.request.inputs
                        else {}
                    ),
                },
                "analysis": analysis,
            },
            unavailable=tuple(unavailable),
        )

    def refresh(self):
        # Runner refresh tracks dependencies of captured production prefixes.
        # Cross-token state is conservatively rebuilt after authored changes.
        from engine.models.qwen35.tensor_program import TensorProgram

        revision = self.operation_sources.refresh()
        if not revision.changed:
            return
        changed_production = (
            not self.production_dependencies
            or not self.production_dependencies.isdisjoint(revision.changed)
        )
        for key, runner in list(self.runners.items()):
            runner.refresh()
            if changed_production and (self.experiment.scope.startswith("decode") or key[1] != 0):
                runner.close()
                del self.runners[key]
        if not changed_production:
            return
        self.model.program.close()
        self.model.program = TensorProgram(self.description, self.device, self.weights)

    def workload_sample(self, observed=None, captured=None):
        from contextlib import nullcontext

        from engine.models.qwen35.inspection import inspect_forwards

        phase = self.experiment.scope.split("/")[0]
        sequence = self.sequence()
        try:
            if phase == "decode":
                self.prefill(sequence)
            observation = (
                inspect_forwards(
                    self.model, observed=observed, captured=captured, kernel_limit=2048
                )
                if observed or captured
                else nullcontext()
            )
            with observation:
                started = time.perf_counter()
                if phase == "prefill":
                    self.prefill(sequence)
                else:
                    for token in self.tokens().continuation:
                        self.forward(sequence, (token,), commit=True)
                elapsed = time.perf_counter() - started
            return elapsed
        finally:
            sequence.close()

    def prepare_component(self, position):
        from ops.formula import FormulaTree
        from ops.lab.preparation import PreparedFormula

        sequence = self.sequence()
        try:
            if self.experiment.scope.startswith("decode/"):
                self.prefill(sequence)
                for token in self.tokens().continuation[:position]:
                    self.forward(sequence, (token,), commit=True)
                tokens = (self.tokens().continuation[position],)
            else:
                prompt = self.tokens().prompt
                for start in range(0, position, self.prefill_rows):
                    self.forward(sequence, prompt[start : start + self.prefill_rows], commit=True)
                tokens = prompt[position : position + self.prefill_rows]
            runner = self.runner(sequence, tokens, position)
            tree = FormulaTree(runner.fixture.root)
            scopes = tree_scopes(tree, self.experiment.scope.split("/")[0])
            selected = next(s for s in scopes if s.selector == self.experiment.scope)
            target = next(t for t in tree if t.call.occurrence == selected.occurrence)
            boundary, _, _ = runner.fixture.capture_boundary(target, self.device, runner.options)
            return PreparedFormula(boundary, self.device, runner.options), scopes
        finally:
            sequence.close()

    def capture_inputs(self):
        """Prepare and publish the requested logical boundary for another worker."""
        import tempfile

        from ops.lab.retention import save_boundary

        prepared, _ = self.prepare_component(self.experiment.step)
        try:
            with tempfile.TemporaryDirectory(dir=self.store.root) as directory:
                path = Path(directory) / "boundary.zip"
                save_boundary(
                    prepared.fixture,
                    path,
                    provenance={
                        "source_id": self.experiment.source,
                        "target": self.attempt.target_name,
                        "hardware": self.hardware,
                        "model_artifact": self.request.model.sha256,
                        "scope": self.experiment.scope,
                        "step": self.experiment.step,
                        "tokens": self.base(scope=self.experiment.scope)["workload"]["tokens"],
                    },
                )
                return self.store.put_blob(path.read_bytes())
        finally:
            prepared.close()

    def measure(self):
        """Ordinary workload repetitions and separate production instrumentation."""
        phase = self.experiment.scope.split("/")[0]
        if "/" in self.experiment.scope:
            positions = (
                [self.experiment.step]
                if self.experiment.step is not None
                else list(range(self.experiment.steps))
                if phase == "decode"
                else list(range(0, self.experiment.context, self.prefill_rows))
            )
            if all((self.experiment.scope, p) in self.runners for p in positions):
                return [self.component(None, None, p) for p in positions]
            sequence = self.sequence()
            results = []
            try:
                if phase == "decode":
                    self.prefill(sequence)
                    for step, token in enumerate(self.tokens().continuation):
                        if self.experiment.step is None or step == self.experiment.step:
                            results.append(self.component(sequence, (token,), step))
                        self.forward(sequence, (token,), commit=True)
                        if self.experiment.step is not None and step >= self.experiment.step:
                            break
                else:
                    # Each prefill chunk is identified separately, never extrapolated.
                    prompt = self.tokens().prompt
                    for start in range(0, len(prompt), self.prefill_rows):
                        chunk = prompt[start : start + self.prefill_rows]
                        results.append(self.component(sequence, chunk, start))
                        self.forward(sequence, chunk, commit=True)
            finally:
                sequence.close()
            return results

        samples, observations, structures = [], [], {}
        analytical = []
        invocations = []
        costs = {}
        phase_started = time.perf_counter()
        unavailable = ["Reference-engine comparison requires separately measured evidence"]
        if self.device.characterization is None:
            unavailable.append("No compatible resource characterization loaded")

        def observed(invocation, observation):
            from ops.performance.publication import capture, manifest
            from ops.runtime.observation import KernelObservation

            graph = manifest(invocation.compiled.formulas.graph, compiled=invocation.compiled)
            captured = capture(
                graph,
                observation,
                capture_id=identity(),
                execution_graph=invocation.compiled.graph.fingerprint,
            )
            analytical.append(
                (
                    graph,
                    captured,
                    observation.elapsed_ns / 1e9,
                    {
                        "artifact": self.request.model.sha256,
                        "phase": phase,
                        "positions": list(invocation.positions),
                        "lengths": list(invocation.lengths),
                        "physical_rows": invocation.physical_rows,
                    },
                )
            )
            hierarchy = tree_scopes(invocation.compiled.formulas, phase)
            observations.append(
                {
                    "positions": invocation.positions,
                    "lengths": invocation.lengths,
                    "observation": asdict(observation),
                    "scopes": [scope.model_dump() for scope in hierarchy],
                }
            )
            paths = {scope.occurrence: scope.selector for scope in hierarchy}
            for scope in hierarchy:
                previous = structures.get(scope.selector)
                if previous is not None and previous.semantics != scope.semantics:
                    structures[scope.selector] = scope.model_copy(
                        update={
                            "complete": False,
                            "reason": "multiple concrete traces; see per-position observations",
                        }
                    )
                    continue
                kernels = observation.kernels
                if kernels is not None and kernels.attribution == "compiled-order-and-symbols":
                    activities = tuple(
                        event
                        for event in kernels.activities
                        if paths.get(event.owner) == scope.selector
                        or paths.get(event.owner, "").startswith(scope.selector + "/")
                    )
                    busy = KernelObservation(kernels.clock, activities).busy_ns
                    if activities and busy is not None:
                        seconds = busy / 1e9 + (
                            previous.contribution_seconds or 0 if previous else 0
                        )
                        scope = scope.model_copy(update={"contribution_seconds": seconds})
                structures[scope.selector] = scope

        def execute(*, diagnostic=False, captured=None):
            return self.workload_sample(observed if diagnostic else None, captured=captured)

        # Keep only graph/coordinate metadata during preparation; ordinary timing
        # samples do not run callbacks, tensor reads, or native instrumentation.
        execute(captured=invocations.append)
        costs["preparation_seconds"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        for _ in range(self.experiment.protocol.warmups):
            execute()
        costs["warmup_seconds"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        base = self.base(scope=phase)
        for _ in range(self.experiment.protocol.samples):
            samples.append(execute())
        costs["sampling_seconds"] = time.perf_counter() - phase_started
        checkpoint = Measurement(
            **base,
            status="incomplete",
            correctness="unchecked",
            samples_seconds=tuple(samples),
            costs=costs,
            unavailable=("Diagnostics and independent checks not completed",),
        )
        self.store.put("measurement-checkpoint", base["measurement_id"], checkpoint)
        phase_started = time.perf_counter()
        # Instrumentation and independent checking never contaminate ordinary samples.
        try:
            execute(diagnostic=True)
        except Exception as failure:
            unavailable.append(f"Separate production observation failed: {failure}")
        costs["observation_seconds"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        checks, error = [], None
        status, correctness = "complete", "passed"
        try:
            checks = self.check_workload()
        except (ArithmeticError, AssertionError) as failure:
            status, correctness, error = "failed", "failed", str(failure)
        except Exception as failure:
            status, correctness, error = "incomplete", "unchecked", str(failure)
            unavailable.append(f"Independent workload check could not complete: {failure}")
        costs["checking_seconds"] = time.perf_counter() - phase_started
        artifact = self.store.put_blob(
            encoded({"observations": observations, "checks": checks, "check_error": error})
        )
        from formula_performance.records import Observation, Publication
        from formula_performance.records import identity as fingerprint
        from ops.performance.publication import hardware as hardware_binding

        system = hardware_binding(
            self.device, capacities=self.attempt.target.capacities
        ).model_copy(update={"label": self.attempt.target_name})
        from formula_performance.composition import sequential
        from ops.performance.publication import manifest

        concrete = {}
        for invocation in invocations:
            key = invocation.compiled.formulas.graph.fingerprint
            if key not in concrete:
                concrete[key] = manifest(
                    invocation.compiled.formulas.graph, compiled=invocation.compiled
                )
        ordinary_graph = sequential(
            concrete[i.compiled.formulas.graph.fingerprint] for i in invocations
        )
        ordinary = Observation(
            identity=base["measurement_id"] + ":ordinary",
            manifest=fingerprint(ordinary_graph),
            component="",
            hardware=fingerprint(system),
            implementation=self.experiment.source,
            created=base["created"],
            coordinates={
                "artifact": self.request.model.sha256,
                "phase": phase,
                "context": self.experiment.context,
                "steps": self.experiment.steps,
                "tokens": base["workload"]["tokens"],
                "forward_count": len(invocations),
            },
            samples=tuple(samples),
            boundary="ordinary-sequential-workload",
            correctness=correctness,
            status=status,
            evidence=(base["measurement_id"],),
        )
        publication = Publication(
            manifests=tuple(
                {
                    fingerprint(g): g
                    for g in (*concrete.values(), ordinary_graph, *(g for g, _, _, _ in analytical))
                }.values()
            ),
            hardware=(system,),
            captures=tuple(c for _, c, _, _ in analytical if c is not None),
            observations=(
                ordinary,
                *tuple(
                    Observation(
                        identity=base["measurement_id"] + ":forward:" + str(i),
                        manifest=fingerprint(g),
                        component="",
                        hardware=fingerprint(system),
                        implementation=self.experiment.source,
                        created=base["created"],
                        coordinates=coordinates,
                        samples=(seconds,),
                        boundary="instrumented-forward-through-retirement",
                        correctness=correctness,
                        status=status,
                        captures=(c.identity,) if c else (),
                        evidence=(base["measurement_id"],),
                    )
                    for i, (g, c, seconds, coordinates) in enumerate(analytical)
                ),
            ),
        )
        analytical_artifact = self.store.put_blob(publication.model_dump_json().encode())
        return [
            Measurement(
                **base,
                costs=costs,
                status=status,
                correctness=correctness,
                samples_seconds=tuple(samples),
                scopes=tuple(structures.values()),
                artifacts={
                    "production-observations": artifact,
                    "formula-performance": analytical_artifact,
                },
                error=error,
                details={
                    "boundary": "production-workload-through-completion",
                    "checks": len(checks),
                    "contribution_basis": (
                        "separate instrumented native interval unions; not ordinary wall time"
                    ),
                },
                unavailable=tuple(unavailable),
            )
        ]

    def check_workload(self):
        """Check production operations and encoded state at their actual inputs."""
        from contextlib import ExitStack

        from engine.models.qwen35.inspection import inspect_forwards
        from ops.lab.checking import OperationChecker

        phase = self.experiment.scope.split("/")[0]
        protocol = measurement_protocol(self.experiment)
        checks, prepared = [], {}
        sequence = self.sequence()
        with ExitStack() as owned:
            owned.callback(sequence.close)
            if phase == "decode":
                self.prefill(sequence)
                positions = [(i, (token,)) for i, token in enumerate(self.tokens().continuation)]
            else:
                prompt = self.tokens().prompt
                positions = [
                    (i, prompt[i : i + self.prefill_rows])
                    for i in range(0, len(prompt), self.prefill_rows)
                ]
            for position, tokens in positions:
                expected = {}

                def before(invocation, expected=expected):
                    compiled = invocation.compiled
                    if compiled not in prepared:
                        checker = OperationChecker(compiled, self.decode_weight, protocol)
                        owned.callback(checker.close)
                        prepared[compiled] = checker
                    expected["invocation"] = invocation
                    expected["checked"] = prepared[compiled].check(
                        invocation.inputs, invocation.resources
                    )

                def after(batch, expected=expected):
                    expected["checked"].compare(
                        self.device,
                        batch.execution.execution.outputs,
                        expected["invocation"].resources,
                        protocol,
                    )

                with inspect_forwards(self.model, captured=before):
                    self.forward(sequence, tokens, commit=True, completed=after)
                checked = expected["checked"]
                checks.append(
                    {
                        "position": position,
                        "contract": "qwen35.model",
                        "passed": True,
                        "reference": "independent-primitives-at-production-operation-inputs",
                        "operations": checked.operations,
                        "encoded_writes": checked.encoded_writes,
                    }
                )
        return checks


def discover(model, target, experiment):
    """Trace the production definition from GGUF metadata without loading weights."""
    import ops
    from engine.models.qwen35.description import MixerKind
    from engine.models.qwen35.formats.gguf import inspect_dense, inspect_moe
    from engine.models.qwen35.tensor_program import InvocationSpecs, define, weight_roles
    from engine.platform.storage import FileSource
    from engine.weights.formats.gguf import read_directory
    from engine.weights.identity import ArtifactIdentity
    from ops.formula import FormulaTree

    if experiment.model.split(":")[1] != "gguf":
        raise NotImplementedError("scope discovery requires a supported GGUF artifact")
    path = Path(model.locations[experiment.targets[0]])
    with closing(FileSource(path)) as source:
        directory = read_directory(source)
        architecture = directory.value("general.architecture")
        inspectors = {"qwen35": inspect_dense, "qwen35moe": inspect_moe}
        if architecture not in inspectors:
            raise ValueError(f"unsupported Qwen GGUF architecture {architecture!r}")
        # Metadata discovery carries the declared identity without claiming verification.
        # Execution checks all bytes before creating a measurement.
        description = inspectors[architecture](directory, ArtifactIdentity(model.sha256))
        g = description.geometry
        phase = experiment.scope.split("/")[0]
        rows = min(512, max(2, experiment.context)) if phase == "prefill" else 1
        attention = sum(layer == MixerKind.ATTENTION for layer in g.layers)
        recurrent = len(g.layers) - attention
        spec = ops.TensorSpec
        i32, u32 = ops.DType.I32, ops.DType.U32
        representation = ops.default_kv_representation(g.attention_width, g.attention_width)
        specs = InvocationSpecs(
            batch=1,
            tokens=spec((rows,), i32),
            coordinates=spec((rows, 3), i32),
            recurrent_offsets=spec((2,), i32) if recurrent else None,
            output_rows=spec((1,), i32),
            draws=spec((1, 6), u32),
            destinations=(spec((rows,), i32),) * attention,
            visible=(spec((rows, 4), i32),) * attention,
            attention_state=(
                ops.kv_state_spec(
                    experiment.context + experiment.steps,
                    g.kv_heads,
                    g.activation_dtype,
                    representation,
                ),
            )
            * attention,
            convolution_state=(
                spec((1, g.recurrent_channels, g.convolution_width - 1), g.activation_dtype),
            )
            * recurrent,
            delta_state=(
                spec(
                    (1, g.recurrent_value_heads, g.recurrent_width, g.recurrent_width),
                    ops.DType.F32,
                ),
            )
            * recurrent,
            packed_controls=phase == "decode",
            recurrent_sequence_length=rows if recurrent and phase == "prefill" else None,
        )
        definition = define(
            description,
            {r.name: spec(r.shape, dtype) for r, dtype in weight_roles(description)},
            phase,
            specs,
        )
        graph = ops.trace(definition.function, definition.signature)
        from ops.performance.publication import manifest

        return {
            "performance_manifest": manifest(graph).model_dump(mode="json"),
            "graph": graph.fingerprint,
            "scopes": [s.model_dump() for s in tree_scopes(FormulaTree(graph), phase)],
            "unresolved": [],
            "artifact_verified": False,
            "note": "Metadata discovery; execution verifies complete artifact checksums.",
        }
