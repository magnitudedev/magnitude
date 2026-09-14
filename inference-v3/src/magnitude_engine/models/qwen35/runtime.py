"""Packed Qwen execution through one Magnitensor whole-step program."""

from __future__ import annotations

import struct
from contextlib import ExitStack
from dataclasses import dataclass

import magnitensor as mt
from magnitude_engine.data import TokenId
from magnitude_engine.models.qwen35.description import DenseDescription, MixerKind
from magnitude_engine.models.qwen35.inputs import Feature, InputPlan, Inputs, InputState
from magnitude_engine.models.qwen35.sequence import (
    Checkpoint,
    Sequence,
    SequenceAdvance,
    SequenceBatch,
)
from magnitude_engine.models.qwen35.state import (
    QwenAdvance,
    QwenState,
    QwenStateStore,
    RecurrentState,
)
from magnitude_engine.models.qwen35.tensor_program import InvocationSpecs, TensorProgram
from magnitude_engine.models.sequence import (
    LogitsSelection,
    ModelExecutor,
    ModelRequest,
    ModelSequence,
)
from magnitude_engine.weights.tensor_residency import TensorWeights


@dataclass(frozen=True)
class ForwardRequest:
    state: QwenState
    inputs: Inputs
    selection: LogitsSelection = LogitsSelection.LAST
    draw_words: tuple[int, int, int, int, int, int] | None = None


class ForwardOutput:
    def __init__(
        self,
        runtime: DenseRuntime,
        advance: QwenAdvance,
        logits: mt.Resource | None,
        sample: mt.Resource | None,
    ):
        self.runtime, self.advance = runtime, advance
        self.logits, self.sample = logits, sample
        self.completion: mt.Completion | None = None
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
    def __init__(self, runtime: DenseRuntime, outputs, execution: mt.Execution, owned: ExitStack):
        self.runtime, self.outputs, self.execution = runtime, tuple(outputs), execution
        self.logits = (
            execution.outputs[0] if any(value.logits is not None for value in outputs) else None
        )
        self._owned = owned
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


class DenseRuntime(ModelExecutor):
    def __init__(
        self,
        description: DenseDescription,
        device: mt.Device,
        weights: TensorWeights,
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
        self.closed = False

    def prime(self, rows: int, horizon: int) -> None:
        self.device.check()
        if not 0 < rows <= horizon <= self.context_capacity:
            raise ValueError("prime geometry must fit the bound model context")
        sequence = self.create(InputPlan.text(tuple(TokenId(0) for _ in range(horizon))))
        try:
            for count, selection, draws in (
                (rows, LogitsSelection.NONE, None),
                (rows, LogitsSelection.LAST, (0, 0, 0, 0, 0, 0)),
                (1, LogitsSelection.LAST, (0, 0, 0, 0, 0, 0)),
            ):
                batch = self.prepare(
                    (
                        ModelRequest(
                            sequence,
                            tuple(TokenId(0) for _ in range(count)),
                            selection,
                            draws,
                        ),
                    )
                )
                batch.completion.wait()
                batch.close()
        finally:
            sequence.close()

    def reclaim(self) -> int:
        return self.states.release_idle()

    def reclaimable(self, sequences: tuple[ModelSequence, ...]) -> int:
        states = []
        for sequence in sequences:
            if not isinstance(sequence, Sequence) or sequence.runtime is not self:
                raise ValueError("reclamation query belongs to another model")
            states.append(sequence.state)
        return self.states.reclaimable(tuple(states))

    def input(self, plan: InputPlan, features: tuple[Feature, ...] = ()):
        from magnitude_engine.models.qwen35.sequence import Source

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
                    ForwardRequest(sequence.state, inputs, request.selection, request.draw_words)
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
            mode = "decode" if all(len(item.inputs.tokens) == 1 for item in requests) else "prefill"
            actual_rows = sum(len(item.inputs.tokens) for item in requests)
            physical_rows = (
                actual_rows if mode == "decode" or self.prefill_rows is None else self.prefill_rows
            )
            if actual_rows > physical_rows:
                raise ValueError("packed prefill exceeds the configured physical row capacity")
            tokens = []
            coordinates = []
            destinations = []
            visible = []
            output_rows = []
            draw_words = []
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
                    visible.extend((base, state.position + offset + 1))
                selected = (
                    range(start, start + len(inputs.tokens))
                    if request.selection == LogitsSelection.ALL
                    else (start + len(inputs.tokens) - 1,)
                    if request.selection == LogitsSelection.LAST
                    else ()
                )
                first = len(output_rows)
                output_rows.extend(selected)
                for _ in selected:
                    draw_words.extend(request.draw_words or (0, 0, 0, 0, 0, 0))
                selections.append((first, len(tuple(selected))))
                for feature in inputs.features:
                    feature_slices.append((start, feature))

            padding = physical_rows - actual_rows
            tokens.extend(TokenId(0) for _ in range(padding))
            coordinates.extend(0 for _ in range(padding * 3))
            destinations.extend(-1 for _ in range(padding))
            visible.extend(0 for _ in range(padding * 2))

            token_resource = self._upload("i", tokens, (len(tokens),), mt.DType.I32, owned)
            coordinate_resource = self._upload(
                "i", coordinates, (len(tokens), 3), mt.DType.I32, owned
            )
            dynamic = [token_resource, coordinate_resource]
            recurrent_offsets = [0]
            for request in requests:
                recurrent_offsets.append(recurrent_offsets[-1] + len(request.inputs.tokens))
            recurrent_offset_resource = None
            if any(kind == MixerKind.RECURRENT for kind in self.geometry.layers):
                recurrent_offset_resource = self._upload(
                    "i",
                    recurrent_offsets,
                    (len(recurrent_offsets),),
                    mt.DType.I32,
                    owned,
                )
                dynamic.append(recurrent_offset_resource)
            output_spec = draw_spec = None
            if output_rows:
                output_resource = self._upload(
                    "i", output_rows, (len(output_rows),), mt.DType.I32, owned
                )
                draw_resource = self._upload(
                    "I", draw_words, (len(output_rows), 6), mt.DType.U32, owned
                )
                dynamic.extend((output_resource, draw_resource))
                output_spec, draw_spec = output_resource.spec, draw_resource.spec
            feature_values = []
            feature_rows = []
            for batch_start, feature in feature_slices:
                spec = mt.TensorSpec((feature.count, self.geometry.hidden), mt.DType.F32)
                value = feature.values.view(
                    spec, feature.source * self.geometry.hidden * mt.DType.F32.itemsize
                )
                owned.callback(value.close)
                rows = tuple(
                    range(
                        batch_start + feature.destination,
                        batch_start + feature.destination + feature.count,
                    )
                )
                row_resource = self._upload("i", rows, (feature.count,), mt.DType.I32, owned)
                feature_values.append(value)
                feature_rows.append(row_resource)
            dynamic.extend(feature_rows)
            destination_resource = self._upload(
                "i", destinations, (len(tokens),), mt.DType.I32, owned
            )
            visible_resource = self._upload("i", visible, (len(tokens), 2), mt.DType.I32, owned)
            recurrent = [state for request in requests for state in request.state.recurrent]
            recurrent_layers = len(recurrent) // len(requests)
            recurrent = [
                requests[sequence].state.recurrent[layer]
                for layer in range(recurrent_layers)
                for sequence in range(len(requests))
            ]
            specs = InvocationSpecs(
                batch=len(requests),
                tokens=token_resource.spec,
                coordinates=coordinate_resource.spec,
                recurrent_offsets=(
                    recurrent_offset_resource.spec
                    if recurrent_offset_resource is not None
                    else None
                ),
                output_rows=output_spec,
                draws=draw_spec,
                destinations=tuple(destination_resource.spec for _ in self.states.attention),
                visible=tuple(visible_resource.spec for _ in self.states.attention),
                attention_state=tuple(cache.spec for cache in self.states.attention),
                convolution_state=tuple(value.convolution.spec for value in recurrent),
                delta_state=tuple(value.delta.spec for value in recurrent),
                features=tuple(value.spec for value in feature_values),
                feature_rows=tuple(value.spec for value in feature_rows),
            )
            resources = {}
            for index, cache in enumerate(self.states.attention):
                resources[f"attention.{index}.destinations"] = destination_resource
                resources[f"attention.{index}.visible"] = visible_resource
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
            execution = compiled.submit(*dynamic, resources=resources)
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
                        mt.TensorSpec((count, self.geometry.vocabulary), mt.DType.F32),
                        first * self.geometry.vocabulary * 4,
                    )
                    sample = samples.view(mt.TensorSpec((count, 2), mt.DType.I32), first * 8)
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
            return Forward(self, outputs, execution, owned.pop_all())

    def _upload(self, code, values, shape, dtype, owned):
        resource = self.device.upload(
            mt.TensorSpec(shape, dtype), struct.pack(f"={len(values)}{code}", *values)
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
            self.states.close()
            self.closed = True
