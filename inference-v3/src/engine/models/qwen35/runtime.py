"""Packed Qwen execution through one Ops whole-step program."""

from __future__ import annotations

import struct
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass

import ops
from engine.data import TokenId
from engine.models.qwen35.description import DenseDescription, MixerKind
from engine.models.qwen35.inputs import Feature, InputPlan, Inputs, InputState
from engine.models.qwen35.sequence import (
    Checkpoint,
    Sequence,
    SequenceAdvance,
    SequenceBatch,
)
from engine.models.qwen35.state import (
    QwenAdvance,
    QwenState,
    QwenStateStore,
    RecurrentState,
)
from engine.models.qwen35.tensor_program import InvocationSpecs, TensorProgram
from engine.models.sequence import (
    LogitsSelection,
    ModelExecutor,
    ModelRequest,
    ModelSequence,
    TokenMask,
)
from engine.weights.residency import WeightResidency


@dataclass(frozen=True)
class ForwardRequest:
    state: QwenState
    inputs: Inputs
    selection: LogitsSelection = LogitsSelection.LAST
    draw_words: tuple[int, int, int, int, int, int] | None = None
    allowed_tokens: bytes | TokenMask | None = None


@dataclass(frozen=True)
class _FixedMask:
    content: bytes

    def mask(self) -> bytes:
        return self.content


class ForwardOutput:
    def __init__(
        self,
        runtime: DenseRuntime,
        advance: QwenAdvance,
        logits: ops.Resource | None,
        sample: ops.Resource | None,
    ):
        self.runtime, self.advance = runtime, advance
        self.logits, self.sample = logits, sample
        self.completion: ops.Completion | None = None
        self.closed = False

    def commit(self) -> None:
        if self.closed:
            raise RuntimeError("forward output is closed")
        self.advance.commit()

    def read_logits(self) -> bytes:
        if self.closed or self.completion is None or self.logits is None:
            raise RuntimeError("forward has no submitted logit output")
        return self.runtime.device.read(self.logits, after=self.completion)

    def read_sample(self) -> tuple[int, int] | None:
        if self.sample is None:
            return None
        if self.closed or self.completion is None:
            raise RuntimeError("forward has no submitted sample output")
        return struct.unpack("=ii", self.runtime.device.read(self.sample, after=self.completion))

    def close(self) -> None:
        if not self.closed:
            self.advance.abort()
            if self.logits is not None:
                self.logits.close()
            if self.sample is not None:
                self.sample.close()
            self.closed = True


class Forward:
    def __init__(self, runtime: DenseRuntime, outputs, execution: ops.Execution, owned: ExitStack):
        self.runtime, self.outputs, self.execution = runtime, tuple(outputs), execution
        self.logits = (
            execution.outputs[0] if any(value.logits is not None for value in outputs) else None
        )
        self._owned = owned
        self._observation: Callable[[], object] | None = None
        self.completion = execution.completion
        self.closed = False
        for output in self.outputs:
            output.completion = self.completion
        runtime._forwards.add(self)

    def close(self) -> None:
        if not self.closed:
            for output in self.outputs:
                output.close()
            for resource in self.execution.outputs:
                resource.close()
            self._owned.close()
            self.closed = True
            self.runtime._forwards.discard(self)
            if self._observation is not None:
                self._observation()


class DenseRuntime(ModelExecutor):
    def __init__(
        self,
        description: DenseDescription,
        device: ops.DeviceRuntime,
        weights: WeightResidency,
        *,
        max_sequences: int = 8,
        prefill_rows: int | None = None,
        context_capacity: int | None = None,
    ):
        if description.artifact_identity != weights.identity:
            raise ValueError("model description and residency refer to different artifacts")
        self.description, self.geometry = description, description.geometry
        self.device = self.context = device
        self.artifact_identity = weights.identity
        self.weights = weights
        if prefill_rows is not None and prefill_rows <= 1:
            raise ValueError("prefill row capacity must exceed one")
        self.prefill_rows = prefill_rows
        self.program = TensorProgram(description, device, weights)
        self._samplers: dict[tuple[int, int], ops.CompiledFunction] = {}
        self.states = QwenStateStore(
            device,
            self.geometry,
            max_sequences,
            context_capacity=context_capacity,
        )
        self.context_capacity = self.states.context_capacity
        self._forwards: set[Forward] = set()
        self._sequences: set[Sequence] = set()
        self._checkpoints: set[Checkpoint] = set()
        self._inspection: tuple[
            Callable[..., object] | None, Callable[..., object] | None, int | None
        ] | None = None
        self.closed = False

    def prime(self, rows: int, horizon: int) -> None:
        self.device.check()
        if not 0 < rows <= horizon <= self.context_capacity:
            raise ValueError("prime geometry must fit the bound model context")
        sequence = self.create(InputPlan.text(tuple(TokenId(0) for _ in range(horizon))))
        try:
            capacities = {rows}
            if self.prefill_rows is not None:
                capacities.update(1 << power for power in range(1, rows.bit_length()) if 1 << power <= rows)
            counts = sorted(capacities)
            if MixerKind.RECURRENT in self.geometry.layers:
                # Full and padded recurrence domains have distinct cache entries.
                counts = sorted(set(counts) | {count - 1 for count in capacities if count > 2})
            unrestricted_mask = _FixedMask(b"\xff" * (((self.geometry.vocabulary + 31) // 32) * 4))
            invocations = [
                (count, selection, draws, mask)
                for count in counts
                for selection, draws, mask in (
                    (LogitsSelection.NONE, None, None),
                    (LogitsSelection.LAST, (0, 0, 0, 0, 0, 0), None),
                    (LogitsSelection.LAST, (0, 0, 0, 0, 0, 0), unrestricted_mask),
                )
            ]
            invocations.extend(
                (1, LogitsSelection.LAST, (0, 0, 0, 0, 0, 0), mask)
                for mask in (None, unrestricted_mask)
            )
            for count, selection, draws, mask in invocations:
                batch = self.prepare(
                    (
                        ModelRequest(
                            sequence,
                            tuple(TokenId(0) for _ in range(count)),
                            selection,
                            draws,
                            mask,
                        ),
                    )
                )
                batch.completion.wait()
                batch.close()
        finally:
            sequence.close()

    def reclaim(self) -> int:
        return (self.states.release_idle() + self.program.reclaim()
                + sum(sampler.release_output_storage() for sampler in self._samplers.values()))

    def reclaimable(self, sequences: tuple[ModelSequence, ...]) -> int:
        states = []
        for sequence in sequences:
            if not isinstance(sequence, Sequence) or sequence.runtime is not self:
                raise ValueError("reclamation query belongs to another model")
            states.append(sequence.state)
        return self.states.reclaimable(tuple(states))

    def input(self, plan: InputPlan, features: tuple[Feature, ...] = ()):
        from engine.models.qwen35.sequence import Source

        return Source(self, plan, features)

    def create(self, plan: InputPlan, features: tuple[Feature, ...] = ()) -> Sequence:
        self.device.check()
        if self.closed:
            raise RuntimeError("model runtime is closed")
        if len(plan.tokens) > self.context_capacity:
            raise ValueError("input plan exceeds the model context")
        if any(feature.values.device is not self.device for feature in features):
            raise ValueError("input features belong to another device")
        inputs = InputState(plan, 0, features, self.geometry.hidden)
        try:
            return Sequence(self, self.states.create(), inputs)
        except BaseException:
            inputs.close()
            raise

    def prepare(self, requests: tuple[ModelRequest, ...]) -> SequenceBatch:
        if not requests or len(requests) > self.states.slots:
            raise ValueError("model preparation requires a nonempty batch within its slot limit")
        if len({id(request.sequence) for request in requests}) != len(requests):
            raise ValueError("model preparation requires distinct sequences")
        sequences = []
        numerical = []
        following = []
        with ExitStack() as cleanup:
            for request in requests:
                if not request.tokens or any(
                    type(token) is not int or not 0 <= token < self.geometry.vocabulary
                    for token in request.tokens
                ):
                    raise ValueError("model input tokens must belong to the bound vocabulary")
                if (request.draw_words is None) != (request.selection == LogitsSelection.NONE):
                    raise ValueError("sampling draws must exactly accompany requested logits")
                if request.allowed_tokens is not None and (
                    request.selection != LogitsSelection.LAST
                    or (
                        len(request.allowed_tokens) != ((self.geometry.vocabulary + 31) // 32) * 4
                        if isinstance(request.allowed_tokens, bytes)
                        else not callable(getattr(request.allowed_tokens, "mask", None))
                    )
                ):
                    raise ValueError("selection mask requires one last-logit row and exact vocabulary words")
                sequence = request.sequence
                if not isinstance(sequence, Sequence) or sequence.runtime is not self:
                    raise ValueError("sequence belongs to another model")
                sequence.check()
                if sequence.pending is not None:
                    raise RuntimeError("sequence already has an unresolved advance")
                inputs = sequence.inputs.assemble(request.tokens)
                next_inputs = sequence.inputs.after(sequence.position + len(request.tokens))
                cleanup.callback(next_inputs.close)
                sequences.append(sequence)
                following.append(next_inputs)
                numerical.append(
                    ForwardRequest(sequence.state, inputs, request.selection, request.draw_words,
                                   request.allowed_tokens)
                )
            forward = self._prepare_numerical(tuple(numerical))
            cleanup.callback(forward.close)
            advances = []
            for sequence, output, inputs in zip(sequences, forward.outputs, following, strict=True):
                advance = SequenceAdvance(sequence, output, inputs)
                cleanup.callback(advance.close)
                sequence.pending = advance
                advances.append(advance)
            result = SequenceBatch(forward, tuple(advances))
            cleanup.pop_all()
            return result

    def _prepare_numerical(self, requests: tuple[ForwardRequest, ...]) -> Forward:
        if not requests or len({id(item.state) for item in requests}) != len(requests):
            raise ValueError("a forward requires distinct sequence states")
        with ExitStack() as owned:
            deferred = any(
                request.allowed_tokens is not None and not isinstance(request.allowed_tokens, bytes)
                for request in requests
            )
            capture = None
            if self._inspection is not None and self._inspection[1] is not None:
                # Enter before packing; exit is attached after transient cleanup.
                capture = owned.enter_context(self.device.observe(kernel_limit=self._inspection[2]))
            mode = "decode" if all(len(item.inputs.tokens) == 1 for item in requests) else "prefill"
            actual_rows = sum(len(item.inputs.tokens) for item in requests)
            if mode != "decode" and self.prefill_rows is not None and actual_rows > self.prefill_rows:
                raise ValueError("packed prefill exceeds the configured physical row capacity")
            physical_rows = (
                actual_rows
                if mode == "decode" or self.prefill_rows is None
                else min(self.prefill_rows, 1 << (actual_rows - 1).bit_length())
            )
            tokens = []
            coordinates = []
            destinations = []
            visible = []
            output_rows = []
            draw_words = []
            mask_payload = bytearray()
            mask_rows = []
            mask_width = (self.geometry.vocabulary + 31) // 32
            selections = []
            advances = []
            feature_slices = []
            for request in requests:
                state, inputs = request.state, request.inputs
                advance = state.begin(len(inputs.tokens))
                owned.callback(advance.abort)
                advances.append(advance)
                start = len(tokens)
                tokens.extend(inputs.tokens)
                coordinates.extend(value for triple in inputs.coordinates for value in triple)
                base = state.slot * self.context_capacity
                for offset in range(len(inputs.tokens)):
                    destinations.append(base + state.position + offset)
                    visible.extend((base, state.position, start, offset + 1))
                selected = (
                    range(start, start + len(inputs.tokens))
                    if request.selection == LogitsSelection.ALL
                    else (start + len(inputs.tokens) - 1,)
                    if request.selection == LogitsSelection.LAST
                    else ()
                )
                first = len(output_rows)
                output_rows.extend(selected)
                mask_row = -1
                if request.allowed_tokens is not None and not deferred:
                    assert isinstance(request.allowed_tokens, bytes)
                    mask_row = len(mask_payload) // (mask_width * 4)
                    mask_payload.extend(request.allowed_tokens)
                for _ in selected:
                    mask_rows.append(mask_row)
                    draw_words.extend(request.draw_words or (0, 0, 0, 0, 0, 0))
                selections.append((first, len(tuple(selected))))
                for feature in inputs.features:
                    feature_slices.append((start, feature))

            padding = physical_rows - actual_rows
            tokens.extend(TokenId(0) for _ in range(padding))
            coordinates.extend(0 for _ in range(padding * 3))
            destinations.extend(-1 for _ in range(padding))
            visible.extend(0 for _ in range(padding * 4))

            packed_controls = mode == "decode" and not feature_slices
            token_spec = ops.TensorSpec((len(tokens),), ops.DType.I32)
            coordinate_spec = ops.TensorSpec((len(tokens), 3), ops.DType.I32)
            destination_spec = ops.TensorSpec((len(tokens),), ops.DType.I32)
            visible_spec = ops.TensorSpec((len(tokens), 4), ops.DType.I32)
            def field(name, code, values, spec) -> tuple[str, bytes, ops.TensorSpec]:
                return name, struct.pack(f"={len(values)}{code}", *values), spec

            fields = [field("tokens", "i", tokens, token_spec),
                      field("coordinates", "i", coordinates, coordinate_spec)]
            recurrent_offsets = [0]
            for request in requests:
                recurrent_offsets.append(recurrent_offsets[-1] + len(request.inputs.tokens))
            recurrent_offset_spec = None
            if any(kind == MixerKind.RECURRENT for kind in self.geometry.layers):
                recurrent_offset_spec = ops.TensorSpec((len(recurrent_offsets),), ops.DType.I32)
                fields.append(field("recurrent_offsets", "i", recurrent_offsets, recurrent_offset_spec))
            output_spec = draw_spec = None
            if output_rows:
                output_spec = ops.TensorSpec((len(output_rows),), ops.DType.I32)
                fields.append(field("output_rows", "i", output_rows, output_spec))
                if not deferred:
                    draw_spec = ops.TensorSpec((len(output_rows), 6), ops.DType.U32)
                    fields.append(field("draws", "I", draw_words, draw_spec))
            masks_spec = mask_rows_spec = None
            if mask_payload:
                masks_spec = ops.TensorSpec((len(mask_payload) // (mask_width * 4), mask_width), ops.DType.U32)
                mask_rows_spec = ops.TensorSpec((len(output_rows),), ops.DType.I32)
                fields.extend((("masks", bytes(mask_payload), masks_spec),
                               field("mask_rows", "i", mask_rows, mask_rows_spec)))
            argument_fields = len(fields)
            if self.states.attention:
                fields.extend((field("destinations", "i", destinations, destination_spec),
                               field("visible", "i", visible, visible_spec)))
            control_resources = {}
            if packed_controls:
                payload = b"".join(content for _, content, _ in fields)
                control = self.device.upload(ops.TensorSpec((len(payload) // 4,), ops.DType.U32), payload)
                owned.callback(control.close)
                dynamic = [control]
            else:
                for name, content, spec in fields:
                    control_resources[name] = self.device.upload(spec, content)
                    owned.callback(control_resources[name].close)
                dynamic = [control_resources[name] for name, _, _ in fields[:argument_fields]]
            feature_values = []
            feature_rows = []
            for batch_start, feature in feature_slices:
                spec = ops.TensorSpec((feature.count, self.geometry.hidden), ops.DType.F32)
                value = feature.values.view(
                    spec, feature.source * self.geometry.hidden * ops.DType.F32.itemsize
                )
                owned.callback(value.close)
                rows = tuple(
                    range(
                        batch_start + feature.destination,
                        batch_start + feature.destination + feature.count,
                    )
                )
                row_resource = self._upload("i", rows, (feature.count,), ops.DType.I32, owned)
                feature_values.append(value)
                feature_rows.append(row_resource)
            dynamic.extend(feature_rows)
            recurrent = [state for request in requests for state in request.state.recurrent]
            recurrent_layers = len(recurrent) // len(requests)
            recurrent = [
                requests[sequence].state.recurrent[layer]
                for layer in range(recurrent_layers)
                for sequence in range(len(requests))
            ]
            specs = InvocationSpecs(
                batch=len(requests),
                tokens=token_spec,
                coordinates=coordinate_spec,
                recurrent_offsets=recurrent_offset_spec,
                output_rows=output_spec,
                draws=draw_spec,
                masks=masks_spec,
                mask_rows=mask_rows_spec,
                destinations=tuple(destination_spec for _ in self.states.attention),
                visible=tuple(visible_spec for _ in self.states.attention),
                attention_state=tuple(cache.spec for cache in self.states.attention),
                convolution_state=tuple(value.convolution.spec for value in recurrent),
                delta_state=tuple(value.delta.spec for value in recurrent),
                features=tuple(value.spec for value in feature_values),
                feature_rows=tuple(value.spec for value in feature_rows),
                packed_controls=packed_controls,
                recurrent_sequence_length=(
                    actual_rows
                    if mode == "prefill" and len(requests) == 1 and padding == 0
                    and recurrent_offset_spec is not None
                    else None
                ),
            )
            resources = {}
            for index, cache in enumerate(self.states.attention):
                if not packed_controls:
                    resources[f"attention.{index}.destinations"] = control_resources["destinations"]
                    resources[f"attention.{index}.visible"] = control_resources["visible"]
                resources[f"attention.{index}.state"] = cache
            for index, value in enumerate(recurrent):
                layer, sequence = divmod(index, len(requests))
                resources[f"recurrent.{layer}.{sequence}.convolution"] = value.convolution
                resources[f"recurrent.{layer}.{sequence}.delta"] = value.delta
            for index, value in enumerate(feature_values):
                resources[f"feature.{index}.values"] = value
            compiled = self.program.specialize(
                mode,
                specs,
                static_resources={
                    f"attention.{index}.state": cache
                    for index, cache in enumerate(self.states.attention)
                },
            )
            if self._inspection is not None:
                from .inspection import Invocation
                invocation = Invocation(compiled, tuple(dynamic), resources, self.program.weights,
                                        mode, tuple(r.state.position for r in requests),
                                        tuple(len(r.inputs.tokens) for r in requests), physical_rows)
                captured, observed, _ = self._inspection
                if captured is not None:
                    captured(invocation)
            sampler = (
                self._sampler(len(output_rows), sum(r.allowed_tokens is not None for r in requests))
                if deferred else None
            )
            execution = compiled.submit(*dynamic, resources=resources)
            if sampler is not None:
                execution = self._sample_deferred(execution, sampler, requests, selections, draw_words)
            attention_count = len(self.states.attention)
            cursor = 2 if output_rows else 0
            logits = execution.outputs[0] if output_rows else None
            samples = execution.outputs[1] if output_rows else None
            cursor += attention_count
            convolution = execution.outputs[cursor : cursor + recurrent_layers]
            cursor += recurrent_layers
            delta = execution.outputs[cursor : cursor + recurrent_layers]
            outputs = []
            for sequence, (first, count), advance in zip(
                range(len(requests)), selections, advances, strict=True
            ):
                logit = sample = None
                if count:
                    assert logits is not None and samples is not None
                    logit = logits.view(
                        ops.TensorSpec((count, self.geometry.vocabulary), ops.DType.F32),
                        first * self.geometry.vocabulary * 4,
                    )
                    sample = samples.view(ops.TensorSpec((count, 2), ops.DType.I32), first * 8)
                following_states = []
                for layer in range(recurrent_layers):
                    conv_spec = recurrent[layer * len(requests) + sequence].convolution.spec
                    delta_spec = recurrent[layer * len(requests) + sequence].delta.spec
                    following_states.append(
                        RecurrentState(
                            convolution[layer].view(conv_spec, sequence * conv_spec.storage_nbytes),
                            delta[layer].view(delta_spec, sequence * delta_spec.storage_nbytes),
                        )
                    )
                advance.submitted(execution.completion, tuple(following_states))
                outputs.append(ForwardOutput(self, advance, logit, sample))
            forward = Forward(self, outputs, execution, owned.pop_all())
            if capture is not None:
                assert observed is not None
                forward._observation = lambda: observed(invocation, capture.result)
            return forward

    def _sampler(self, rows: int, masked: int) -> ops.CompiledFunction:
        key = rows, masked
        if key not in self._samplers:
            width = self.geometry.vocabulary
            specs = (
                ops.TensorSpec((rows, 6), ops.DType.U32),
                ops.TensorSpec((masked, (width + 31) // 32), ops.DType.U32),
                ops.TensorSpec((rows,), ops.DType.I32),
            )

            def select(logits, controls):
                draws, masks, mask_rows = ops.unpack_words(controls, specs)
                return ops.sample_constrained(logits, draws, masks, mask_rows)

            sampler = ops.compile(
                select,
                signature=ops.Signature((
                    ops.Argument(ops.TensorSpec((rows, width), ops.DType.F32), "logits"),
                    ops.Argument(
                        ops.TensorSpec((sum(spec.elements for spec in specs),), ops.DType.U32),
                        "controls",
                    ),
                )),
                device=self.device, constants={}, options=ops.CompileOptions(mode="decode"),
            )
            sampler.reuse_output_storage(max_frames=2)
            self._samplers[key] = sampler
        return self._samplers[key]

    def _sample_deferred(
        self, forward: ops.Execution, sampler: ops.CompiledFunction,
        requests: tuple[ForwardRequest, ...], selections: list[tuple[int, int]],
        draw_words: list[int],
    ) -> ops.Execution:
        transfer = selected = None
        try:
            masks, rows = [], []
            width = ((self.geometry.vocabulary + 31) // 32) * 4
            for request, (_, count) in zip(requests, selections, strict=True):
                value = request.allowed_tokens
                if value is None:
                    rows.extend([-1] * count)
                else:
                    mask = value if isinstance(value, bytes) else value.mask()
                    if type(mask) is not bytes or len(mask) != width:
                        raise ValueError("selection mask requires exact vocabulary words")
                    rows.extend([len(masks)] * count)
                    masks.append(mask)
            payload = (struct.pack(f"={len(draw_words)}I", *draw_words) + b"".join(masks)
                       + struct.pack(f"={len(rows)}i", *rows))
            transfer = self.device.upload_async(
                ops.TensorSpec((len(payload) // 4,), ops.DType.U32), payload
            )
            selected = sampler.submit(forward.outputs[0], transfer.outputs[0])
            completion = ops.Completion.join((
                forward.completion, transfer.completion, selected.completion,
            ))
            outputs = (forward.outputs[0], selected.outputs[0], *forward.outputs[1:])
            return ops.Execution(outputs, completion)
        except BaseException:
            executions = [value for value in (forward, transfer, selected) if value is not None]
            try:
                ops.Completion.join(tuple(value.completion for value in executions)).wait()
            finally:
                for execution in executions:
                    for output in execution.outputs:
                        output.close()
            raise
        finally:
            if transfer is not None:
                transfer.outputs[0].close()

    def _upload(self, code, values, shape, dtype, owned):
        resource = self.device.upload(
            ops.TensorSpec(shape, dtype), struct.pack(f"={len(values)}{code}", *values)
        )
        owned.callback(resource.close)
        return resource

    def close(self) -> None:
        if not self.closed:
            for sequence in tuple(self._sequences):
                sequence.close()
            for checkpoint in tuple(self._checkpoints):
                checkpoint.close()
            for forward in tuple(self._forwards):
                forward.close()
            self.program.close()
            for sampler in self._samplers.values():
                sampler.close()
            self.states.close()
            self.closed = True
