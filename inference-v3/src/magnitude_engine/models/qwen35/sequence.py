"""Complete Qwen input/numerical continuations at the model boundary."""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING

from magnitude_engine.models.qwen35.inputs import Feature, InputPlan, InputState
from magnitude_engine.models.qwen35.state import QwenCheckpoint, QwenState
from magnitude_engine.platform.execution import Ticket

if TYPE_CHECKING:
    from magnitude_engine.models.qwen35.runtime import DenseRuntime, Forward, ForwardOutput


class SequenceAdvance:
    def __init__(self, sequence: Sequence, forward: ForwardOutput, inputs: InputState):
        self.sequence, self.forward, self._inputs = sequence, forward, inputs
        self.closed, self.committed = False, False

    @property
    def logits(self):
        return self.forward.logits

    def commit(self) -> None:
        if self.closed or self.committed or self.sequence.pending is not self:
            raise RuntimeError("sequence advance is no longer current")
        self.forward.commit()
        previous = self.sequence.inputs
        self.sequence.inputs = self._inputs
        self.sequence.pending = None
        self.committed = True
        previous.close()

    def close(self) -> None:
        if not self.closed:
            self.forward.close()
            if not self.committed:
                self._inputs.close()
            if self.sequence.pending is self:
                self.sequence.pending = None
            self.closed = True


class SequenceBatch:
    def __init__(self, execution: Forward, advances: tuple[SequenceAdvance, ...]):
        self.execution, self.advances = execution, advances
        self.closed = False

    @property
    def logits(self):
        if self.closed:
            raise RuntimeError("model batch is closed")
        return self.execution.logits

    @property
    def commands(self):
        if self.closed:
            raise RuntimeError("model batch is closed")
        return self.execution.commands

    def submitted(self, completion: Ticket) -> None:
        if self.closed:
            raise RuntimeError("model batch is closed")
        self.execution.submitted(completion)

    def close(self) -> None:
        if not self.closed:
            for advance in self.advances:
                advance.close()
            self.execution.close()
            self.closed = True


class Sequence:
    def __init__(self, runtime: DenseRuntime, state: QwenState, inputs: InputState):
        if state.position != inputs.position or state.store is not runtime.states:
            raise ValueError("numerical and semantic continuation boundaries differ")
        self.runtime, self.state, self.inputs = runtime, state, inputs
        state.anticipate(len(inputs.plan.tokens))
        self.pending: SequenceAdvance | None = None
        self.closed = False
        runtime._sequences.add(self)

    @property
    def context(self):
        return self.runtime.context

    @property
    def position(self):
        return self.state.position

    @property
    def layout(self):
        return self.inputs.plan.layout

    @property
    def context_limit(self):
        return self.runtime.geometry.context_limit

    def check(self) -> None:
        self.state.check()
        self.inputs.check()
        if self.closed or self.runtime.closed:
            raise RuntimeError("model sequence is closed")

    @property
    def model(self):
        return self.runtime

    def checkpoint(self) -> Checkpoint:
        self.check()
        if self.pending is not None:
            raise RuntimeError("checkpoint requires reconciled model and input state")
        with ExitStack() as cleanup:
            inputs = self.inputs.after(self.position)
            cleanup.callback(inputs.close)
            numerical = self.state.checkpoint()
            cleanup.callback(numerical.close)
            result = Checkpoint(self.runtime, numerical, inputs)
            cleanup.pop_all()
            return result

    def close(self) -> None:
        self.context.check_thread()
        if not self.closed:
            if self.pending is not None:
                self.pending.close()
            self.state.close()
            self.inputs.close()
            self.closed = True
            self.runtime._sequences.discard(self)


class Checkpoint:
    def __init__(self, runtime: DenseRuntime, numerical: QwenCheckpoint, inputs: InputState):
        self.runtime, self.numerical, self.inputs = runtime, numerical, inputs
        self.position, self.closed = numerical.position, False
        runtime._checkpoints.add(self)

    def fork(self) -> Sequence:
        self.runtime.context.check()
        if self.closed or self.runtime.closed:
            raise RuntimeError("model checkpoint is closed")
        with ExitStack() as cleanup:
            inputs = self.inputs.after(self.position)
            cleanup.callback(inputs.close)
            numerical = self.runtime.states.create(self.numerical)
            cleanup.callback(numerical.close)
            result = Sequence(self.runtime, numerical, inputs)
            cleanup.pop_all()
            return result

    def close(self) -> None:
        self.runtime.context.check_thread()
        if not self.closed:
            self.inputs.close()
            self.numerical.close()
            self.closed = True
            self.runtime._checkpoints.discard(self)


class Source:
    """Retain original conditioning independently of resident decoder state."""

    def __init__(self, model: DenseRuntime, plan: InputPlan, features: tuple[Feature, ...] = ()):
        model.context.check()
        if model.closed:
            raise RuntimeError("model runtime is closed")
        if (
            not plan.tokens
            or len(plan.tokens) > model.geometry.context_limit
            or any(token >= model.geometry.vocabulary for token in plan.tokens)
        ):
            raise ValueError("input plan exceeds the bound model's vocabulary or context")
        if any(feature.values.context is not model.context for feature in features):
            raise ValueError("input features belong to another execution owner")
        self.model = model
        self.inputs = InputState(plan, 0, features, model.geometry.hidden)

    @property
    def prompt(self):
        return self.inputs.plan.tokens

    def open(self) -> Sequence:
        self.inputs.check()
        return self.model.create(self.inputs.plan, self.inputs.features)

    def close(self) -> None:
        self.inputs.close()
