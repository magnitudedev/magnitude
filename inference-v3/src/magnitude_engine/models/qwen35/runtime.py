"""Packed model requests with independent complete continuation acceptance.

Input meaning is assembled once per sequence, then numerical execution shares
row-independent operations. The numerical batch owns commands; each sequence
owns its accepted input and numerical state. Generation owns publication.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass

from magnitude_engine.models.qwen35.artifact import DenseArtifact
from magnitude_engine.models.qwen35.inputs import Feature, InputPlan, Inputs, InputState
from magnitude_engine.models.qwen35.program import DenseProgram
from magnitude_engine.models.qwen35.sequence import (
    Checkpoint,
    Sequence,
    SequenceAdvance,
    SequenceBatch,
)
from magnitude_engine.models.qwen35.state import QwenAdvance, QwenState, QwenStateStore
from magnitude_engine.models.sequence import (
    LogitsSelection,
    ModelExecutor,
    ModelRequest,
    ModelSequence,
)
from magnitude_engine.numerics.policy import NumericalFamily
from magnitude_engine.operations.copy import Copy
from magnitude_engine.operations.factory import WeightOperations
from magnitude_engine.operations.preparation import Preparation
from magnitude_engine.platform.execution import DType, Prepared, Tensor, TensorSpec, Ticket


@dataclass(frozen=True)
class ForwardRequest:
    state: QwenState
    inputs: Inputs
    selection: LogitsSelection = LogitsSelection.LAST


class ForwardOutput:
    """One candidate's acceptance and output, independent of batch ownership."""

    def __init__(self, runtime: DenseRuntime, advance: QwenAdvance, logits: Tensor | None):
        self.runtime, self.advance, self.logits = runtime, advance, logits
        self.completion: Ticket | None = None
        self.closed = False

    def commit(self) -> None:
        if self.closed:
            raise RuntimeError("forward output is closed")
        self.advance.commit()

    def read_logits(self) -> bytes:
        if self.closed or self.completion is None or self.logits is None:
            raise RuntimeError("forward has no submitted logit output")
        return self.runtime.context.read(self.logits, after=self.completion)

    def close(self) -> None:
        self.runtime.context.check_thread()
        if not self.closed:
            self.advance.abort()
            if self.logits is not None:
                self.logits.close()
            self.closed = True


class Forward:
    """One physical execution owns commands; its outputs accept independently."""

    def __init__(
        self,
        runtime: DenseRuntime,
        outputs: tuple[ForwardOutput, ...],
        logits: Tensor | None,
        commands: tuple[Prepared, ...],
        ownership: ExitStack,
    ):
        self.runtime, self.outputs, self.logits = runtime, outputs, logits
        self._commands, self._ownership = commands, ownership
        self.completion: Ticket | None = None
        self.closed = False

    @property
    def commands(self) -> tuple[Prepared, ...]:
        if self.closed or self.completion is not None:
            raise RuntimeError("forward is not awaiting submission")
        return self._commands

    def submitted(self, completion: Ticket) -> None:
        if self.closed or self.completion is not None:
            raise RuntimeError("forward is not awaiting submission")
        if any(command.submission is not completion for command in self._commands):
            raise ValueError("completion does not cover every command in this forward")
        for output in self.outputs:
            if output.advance.closed:
                output.close()
            if not output.closed:
                output.advance.submitted(completion)
                output.completion = completion
        self.completion = completion

    def close(self) -> None:
        self.runtime.context.check_thread()
        if not self.closed:
            for command in reversed(self._commands):
                command.close()
            self._commands = ()
            self._ownership.close()
            self.closed = True
            self.runtime._forwards.discard(self)


class DenseRuntime(ModelExecutor):
    def __init__(
        self,
        description: DenseArtifact,
        operations: WeightOperations,
        numerics: NumericalFamily = NumericalFamily.NATIVE_BF16,
    ):
        self.context, self.geometry, self.numerics = (
            operations.context,
            description.geometry,
            numerics,
        )
        self.artifact_identity = operations.artifact_identity
        self._forwards: set[Forward] = set()
        self._sequences: set[Sequence] = set()
        self._checkpoints: set[Checkpoint] = set()
        self.closed = False
        with ExitStack() as cleanup:
            self.copy = Copy(self.context)
            cleanup.callback(self.copy.close)
            self.program = DenseProgram(description, operations, numerics)
            cleanup.callback(self.program.close)
            self.states = QwenStateStore(self.context, self.geometry, numerics)
            cleanup.callback(self.states.close)
            self._cleanup = cleanup.pop_all()

    def reclaim(self) -> int:
        self.context.check()
        before = self.context.allocated_bytes
        self.program.release_binding()
        self.program.workspace.close()
        self.states.release_idle()
        return before - self.context.allocated_bytes

    def reclaimable(self, sequences: tuple[ModelSequence, ...]) -> int:
        states = []
        for sequence in sequences:
            if not isinstance(sequence, Sequence) or sequence.runtime is not self:
                raise ValueError("reclamation query belongs to another model")
            sequence.check()
            if sequence.pending is not None:
                raise ValueError("reclamation requires reconciled sequences")
            states.append(sequence.state)
        return self.states.reclaimable(tuple(states))

    def input(self, plan: InputPlan, features: tuple[Feature, ...] = ()):
        from magnitude_engine.models.qwen35.sequence import Source

        return Source(self, plan, features)

    def create(self, plan: InputPlan, features: tuple[Feature, ...] = ()) -> Sequence:
        self.context.check()
        if self.closed:
            raise RuntimeError("model runtime is closed")
        if len(plan.tokens) > self.geometry.context_limit or any(
            token >= self.geometry.vocabulary for token in plan.tokens
        ):
            raise ValueError("input plan exceeds the bound model's vocabulary or context")
        if any(feature.values.context is not self.context for feature in features):
            raise ValueError("input features belong to another execution owner")
        with ExitStack() as cleanup:
            inputs = InputState(plan, 0, features, self.geometry.hidden)
            cleanup.callback(inputs.close)
            state = self.states.create()
            cleanup.callback(state.close)
            result = Sequence(self, state, inputs)
            cleanup.pop_all()
            return result

    def prepare(self, requests: tuple[ModelRequest, ...]) -> SequenceBatch:
        """Prepare one packed execution and independent complete continuations."""
        self.context.check()
        if self.closed:
            raise RuntimeError("model runtime is closed")
        if not requests or len({id(request.sequence) for request in requests}) != len(requests):
            raise ValueError("a model batch requires distinct sequence requests")
        sequences: list[Sequence] = []
        for request in requests:
            sequence = request.sequence
            if not isinstance(sequence, Sequence) or sequence.runtime is not self:
                raise ValueError("sequence belongs to another bound model")
            sequence.check()
            if sequence.pending is not None:
                raise RuntimeError("sequence already has an unresolved advance")
            sequences.append(sequence)
        with ExitStack() as cleanup:
            following: list[InputState] = []
            numerical: list[ForwardRequest] = []
            for request, sequence in zip(requests, sequences, strict=True):
                inputs = sequence.inputs.assemble(request.tokens)
                next_inputs = sequence.inputs.after(sequence.position + len(request.tokens))
                cleanup.callback(next_inputs.close)
                following.append(next_inputs)
                numerical.append(ForwardRequest(sequence.state, inputs, request.selection))
            execution = self._prepare_numerical(tuple(numerical))
            cleanup.callback(execution.close)
            advances: list[SequenceAdvance] = []
            for sequence, output, inputs in zip(
                sequences, execution.outputs, following, strict=True
            ):
                advance = SequenceAdvance(sequence, output, inputs)
                cleanup.callback(advance.close)
                sequence.pending = advance
                advances.append(advance)
            result = SequenceBatch(execution, tuple(advances))
            cleanup.pop_all()
            return result

    def _validate(self, request: ForwardRequest) -> None:
        state, inputs, selection = request.state, request.inputs, request.selection
        if state.store is not self.states:
            raise ValueError("continuation belongs to another bound model")
        if not isinstance(selection, LogitsSelection):
            raise ValueError("invalid logit selection")
        tokens = inputs.tokens
        if not tokens or any(
            type(t) is not int or not 0 <= t < self.geometry.vocabulary for t in tokens
        ):
            raise ValueError("model input contains invalid token IDs")
        if len(inputs.coordinates) != len(tokens) or any(
            len(coordinate) != 3
            or any(type(value) is not int or not 0 <= value <= 0x7FFFFFFF for value in coordinate)
            for coordinate in inputs.coordinates
        ):
            raise ValueError("model inputs require three int32 rotary coordinates per token")
        end = 0
        for feature in inputs.features:
            if (
                len(feature.values.spec.shape) != 2
                or feature.values.spec.dtype != DType.F32
                or feature.values.spec.shape[1] != self.geometry.hidden
                or any(
                    type(value) is not int
                    for value in (feature.source, feature.destination, feature.count)
                )
                or feature.source < 0
                or feature.destination < end
                or feature.count <= 0
                or feature.source + feature.count > feature.values.spec.shape[0]
                or feature.destination + feature.count > len(tokens)
            ):
                raise ValueError("conditioning slices must be valid, ordered and disjoint")
            end = feature.destination + feature.count

    def _prepare_numerical(self, requests: tuple[ForwardRequest, ...]) -> Forward:
        self.context.check()
        if self.closed:
            raise RuntimeError("model runtime is closed")
        if not requests or len({id(request.state) for request in requests}) != len(requests):
            raise ValueError("a forward requires distinct nonempty sequence requests")
        for request in requests:
            self._validate(request)
        with ExitStack() as ownership, Preparation(self.context) as p:
            advances: list[QwenAdvance] = []
            tokens: list[int] = []
            positions: list[int] = []
            coordinates: list[int] = []
            output_rows: list[int] = []
            selections: list[tuple[int, int]] = []
            for request in requests:
                state, inputs = request.state, request.inputs
                count, start = len(inputs.tokens), len(tokens)
                selected = (
                    tuple(range(start, start + count))
                    if request.selection == LogitsSelection.ALL
                    else (start + count - 1,)
                    if request.selection == LogitsSelection.LAST
                    else ()
                )
                selections.append((len(output_rows), len(selected)))
                output_rows.extend(selected)
                tokens.extend(inputs.tokens)
                positions.extend(range(state.position, state.position + count))
                coordinates.extend(
                    value for coordinate in inputs.coordinates for value in coordinate
                )
            embeddings = self.program.input_buffer(len(tokens))
            ownership.callback(embeddings.close)
            logits = None
            if output_rows:
                logits = self.context.allocate(
                    TensorSpec((len(output_rows), self.geometry.vocabulary), DType.F32)
                )
                ownership.callback(logits.close)
            # Workspace/readout are mandatory for this selected service unit.
            # Allocate them before state growth may choose a larger KV slab.
            for request in requests:
                advance = request.state.begin(len(request.inputs.tokens))
                ownership.callback(advance.abort)
                advances.append(advance)
            p.add(*self.program.embedding.prepare(p.indices(tuple(tokens)), embeddings))
            start = 0
            for request in requests:
                for feature in request.inputs.features:
                    spec = TensorSpec((feature.count, self.geometry.hidden), DType.F32)
                    stride = self.geometry.hidden * 4
                    source = p.view(feature.values, spec, feature.source * stride)
                    destination = p.view(
                        embeddings,
                        TensorSpec(spec.shape, self.numerics.activation),
                        (start + feature.destination)
                        * self.geometry.hidden
                        * self.numerics.activation.itemsize,
                    )
                    p.add(*self.copy.prepare(source, destination))
                start += len(request.inputs.tokens)
            p.add(
                *self.program.prepare(
                    embeddings,
                    p.indices(tuple(coordinates), (len(tokens), 3)),
                    p.indices(tuple(positions)),
                    tuple(advances),
                    logits,
                    output_rows=tuple(output_rows),
                )
            )
            outputs: list[ForwardOutput] = []
            for advance, (start, count) in zip(advances, selections, strict=True):
                selected_logits = None
                if count:
                    assert logits is not None
                    selected_logits = logits.view(
                        TensorSpec((count, self.geometry.vocabulary), DType.F32),
                        start * self.geometry.vocabulary * 4,
                    )
                output = ForwardOutput(self, advance, selected_logits)
                ownership.callback(output.close)
                outputs.append(output)
            result = Forward(self, tuple(outputs), logits, p.finish(), ownership.pop_all())
            self._forwards.add(result)
            return result

    def close(self) -> None:
        self.context.check_thread()
        if not self.closed:
            for sequence in tuple(self._sequences):
                sequence.close()
            for checkpoint in tuple(self._checkpoints):
                checkpoint.close()
            for forward in tuple(self._forwards):
                forward.close()
            self._cleanup.close()
            self.closed = True
