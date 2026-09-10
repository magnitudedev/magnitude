"""Plain generation through one prepared-work and completion lifecycle.

The execution owner submits work. This continuation owns logical history,
pending input and output credit; it never inspects a model's state storage.
"""

from __future__ import annotations

from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated

from pydantic import Field

from magnitude_engine.data import Record, TokenId
from magnitude_engine.models.sequence import (
    LogitsSelection,
    ModelAdvance,
    ModelBatch,
    ModelCheckpoint,
    ModelRequest,
    ModelSequence,
)
from magnitude_engine.operations.preparation import Preparation
from magnitude_engine.operations.sampling import (
    Draw,
    SamplePosition,
    SampleSelector,
    SamplingSeed,
    SelectionKind,
    UnselectableDistribution,
)
from magnitude_engine.platform.execution import DType, Prepared, Tensor, TensorSpec, Ticket


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    CONTEXT = "context"
    CANCELLED = "cancelled"
    FAILED = "failed"


class WaitReason(StrEnum):
    COMPLETION = "completion"
    OUTPUT = "output"
    FINISHED = "finished"
    RESIDENCY = "residency"


class WorkKind(StrEnum):
    PREFILL = "prefill"
    DECODE = "decode"
    REPLAY = "replay"


class Options(Record):
    max_tokens: int = Field(ge=0, le=0x7FFFFFFF)
    stop_tokens: frozenset[Annotated[TokenId, Field(ge=0, le=0x7FFFFFFF)]] = frozenset()
    selection: SelectionKind = SelectionKind.GREEDY
    seed: SamplingSeed = Field(default=SamplingSeed(0), ge=0, lt=2**64)
    output_capacity: int = Field(default=16, gt=0)


class OutputToken(Record):
    index: int = Field(ge=0)
    token: TokenId = Field(ge=0, le=0x7FFFFFFF)


class Continuation(Record):
    """Published cursor and undelivered output survive preemption together."""

    sampled: tuple[TokenId, ...] = ()
    output: tuple[OutputToken, ...] = ()
    published: int = Field(default=0, ge=0)
    finish: FinishReason | None = None


class Recovery(Record):
    """Accepted logical history needed to reconstruct discarded numerical state."""

    processed: int = Field(ge=0)
    continuation: Continuation


@dataclass(frozen=True)
class Ready:
    """A logical proposal; no execution or capacity is reserved until batching."""

    generation: Generation
    kind: WorkKind
    tokens: tuple[TokenId, ...]
    selection: LogitsSelection
    processed: int
    sample_position: SamplePosition


class GenerationWork:
    def __init__(
        self,
        ready: Ready,
        advance: ModelAdvance,
        sample: Tensor | None,
    ):
        self.generation, self.kind, self.count = ready.generation, ready.kind, len(ready.tokens)
        self.advance, self.sample = advance, sample
        self.completion: Ticket | None = None
        self.closed = False

    def finish(self) -> None:
        generation = self.generation
        if self.closed or self.completion is None or generation.pending is not self:
            raise RuntimeError("generation work is not awaiting completion")
        if not self.completion.done:
            raise RuntimeError("generation requires proven execution completion")
        try:
            self.completion.wait()
            selected = None
            if self.sample is not None:
                selected = generation.selector.read(self.sample, after=self.completion)[0]
                if isinstance(selected, UnselectableDistribution):
                    raise ValueError(f"distribution cannot be selected: {selected.reason.name}")
            self.advance.commit()
            if selected is not None:
                generation._accept(selected.token)
        except BaseException:
            generation.finish_reason = FinishReason.FAILED
            raise
        finally:
            self.close()

    def close(self) -> None:
        if not self.closed:
            self.advance.close()
            if self.sample is not None:
                self.sample.close()
            if self.generation.pending is self:
                self.generation.pending = None
            self.closed = True


class GenerationBatch:
    def __init__(
        self,
        execution: ModelBatch,
        works: tuple[GenerationWork, ...],
        commands: tuple[Prepared, ...],
        ownership: ExitStack,
    ):
        self.execution, self.works = execution, works
        self._commands, self._ownership = commands, ownership
        self.completion: Ticket | None = None
        self.closed = False

    @classmethod
    def prepare(cls, ready: tuple[Ready, ...]) -> GenerationBatch:
        if not ready or len({id(item.generation) for item in ready}) != len(ready):
            raise ValueError("a generation batch requires distinct ready requests")
        first = ready[0].generation
        model, selector = first.sequence.model, first.selector
        for item in ready:
            generation = item.generation
            if generation.sequence.model is not model or generation.selector is not selector:
                raise ValueError("ready requests do not share model and selector bindings")
            if generation.ready(len(item.tokens)) != item:
                raise ValueError("generation proposal is no longer ready")
        with ExitStack() as ownership, Preparation(first.sequence.context) as p:
            execution = model.prepare(
                tuple(
                    ModelRequest(item.generation.sequence, item.tokens, item.selection)
                    for item in ready
                )
            )
            ownership.callback(execution.close)
            p.add(*execution.commands)
            draws = tuple(
                Draw(
                    kind=item.generation.options.selection,
                    seed=item.generation.options.seed,
                    position=item.sample_position,
                )
                for item in ready
                if item.selection != LogitsSelection.NONE
            )
            samples = None
            if draws:
                if execution.logits is None or execution.logits.spec.shape[0] != len(draws):
                    raise RuntimeError("model batch omitted requested logits")
                samples = p.allocate(TensorSpec((len(draws), 2), DType.I32))
                p.add(*selector.prepare(execution.logits, draws, samples))
            works: list[GenerationWork] = []
            selected = 0
            for item, advance in zip(ready, execution.advances, strict=True):
                sample = None
                if item.selection != LogitsSelection.NONE:
                    assert samples is not None
                    sample = samples.view(TensorSpec((1, 2), DType.I32), selected * 8)
                    selected += 1
                work = GenerationWork(item, advance, sample)
                ownership.callback(work.close)
                item.generation.pending = work
                works.append(work)
            return cls(execution, tuple(works), p.finish(), ownership.pop_all())

    @property
    def commands(self) -> tuple[Prepared, ...]:
        if self.closed or self.completion is not None:
            raise RuntimeError("generation batch is not awaiting submission")
        return self._commands

    def submitted(self, completion: Ticket) -> None:
        if self.closed or self.completion is not None:
            raise RuntimeError("generation batch is not awaiting submission")
        if any(command.submission is not completion for command in self._commands):
            raise ValueError("completion does not cover all generation work")
        self.execution.submitted(completion)
        for work in self.works:
            if not work.closed:
                work.completion = completion
        self.completion = completion

    def finish(self) -> None:
        if self.closed or self.completion is None or not self.completion.done:
            raise RuntimeError("generation batch requires proven execution completion")
        errors: list[Exception] = []
        try:
            for work in self.works:
                if not work.closed:
                    try:
                        work.finish()
                    except Exception as error:
                        errors.append(error)
            if len(errors) == 1:
                raise errors[0]
            if errors:
                raise ExceptionGroup("generation requests failed", errors)
        finally:
            self.close()

    def close(self) -> None:
        if not self.closed:
            for command in reversed(self._commands):
                command.close()
            self._commands = ()
            self._ownership.close()
            self.closed = True


class Generation:
    def __init__(
        self,
        sequence: ModelSequence,
        prompt: tuple[TokenId, ...],
        selector: SampleSelector,
        options: Options,
        *,
        continuation: Continuation | None = None,
        recovery: Recovery | None = None,
    ):
        if continuation is not None and recovery is not None:
            raise ValueError("supply either resident continuation or reconstruction history")
        if recovery is not None:
            continuation = recovery.continuation
        continuation = Continuation() if continuation is None else continuation
        accepted_position = sequence.position if recovery is None else recovery.processed
        if sequence.context is not selector.context:
            raise ValueError("model and selector must use the same execution owner")
        if (
            len(prompt) != sequence.layout.count
            or not prompt
            or len(prompt) > sequence.context_limit
        ):
            raise ValueError("generation requires a nonempty prompt inside the model context")
        if (
            not 0 <= sequence.position <= accepted_position <= sequence.context_limit
            or (
                continuation.sampled
                and accepted_position != len(prompt) + len(continuation.sampled) - 1
            )
            or (not continuation.sampled and accepted_position >= len(prompt))
            or len(continuation.output) > options.output_capacity
            or tuple(item.index for item in continuation.output)
            != tuple(
                range(continuation.published, continuation.published + len(continuation.output))
            )
        ):
            raise ValueError("generation continuation and processed/publication boundaries differ")
        self.sequence, self.prompt, self.selector, self.options = (
            sequence,
            prompt,
            selector,
            options,
        )
        self.sampled = list(continuation.sampled)
        self.output = deque(continuation.output)
        self.published = continuation.published
        self.finish_reason = continuation.finish or (
            FinishReason.LENGTH if options.max_tokens == 0 else None
        )
        self.pending: GenerationWork | None = None
        self._recovery_position = accepted_position
        self.resident = True
        self.closed = False

    @property
    def processed(self) -> int:
        return self.sequence.position

    @property
    def accepted_position(self) -> int:
        """Authoritative input boundary, including history awaiting reconstruction."""
        return max(self.processed, self._recovery_position)

    @property
    def pending_input(self) -> TokenId | None:
        return self.sampled[-1] if self.sampled and self.finish_reason is None else None

    def check(self) -> None:
        self.sequence.context.check()
        if self.closed:
            raise RuntimeError("generation is closed")

    def ready(self, allowance: int) -> Ready | WaitReason:
        self.check()
        if type(allowance) is not int or allowance <= 0:
            raise ValueError("service allowance must be positive")
        if self.pending is not None:
            return WaitReason.COMPLETION
        if self.finish_reason is not None:
            return WaitReason.FINISHED
        if len(self.output) >= self.options.output_capacity:
            return WaitReason.OUTPUT
        if not self.resident:
            return WaitReason.RESIDENCY
        position = self.processed
        if position < self._recovery_position:
            accepted = self.prompt + tuple(self.sampled)
            end = self.sequence.layout.chunk_end(position, self._recovery_position, allowance)
            return Ready(
                self,
                WorkKind.REPLAY,
                accepted[position:end],
                LogitsSelection.NONE,
                position,
                SamplePosition(len(self.sampled)),
            )
        if position < len(self.prompt):
            end = self.sequence.layout.chunk_end(position, len(self.prompt), allowance)
            tokens = self.prompt[position:end]
            selection = LogitsSelection.LAST if end == len(self.prompt) else LogitsSelection.NONE
            kind = WorkKind.PREFILL
        else:
            if self.pending_input is None or position != len(self.prompt) + len(self.sampled) - 1:
                raise RuntimeError("generation history and model continuation disagree")
            tokens, selection, kind = (self.pending_input,), LogitsSelection.LAST, WorkKind.DECODE
        return Ready(self, kind, tokens, selection, position, SamplePosition(len(self.sampled)))

    def _accept(self, token: TokenId) -> None:
        self.sampled.append(token)
        if token in self.options.stop_tokens:
            self.finish_reason = FinishReason.STOP
        else:
            self.output.append(OutputToken(index=self.published + len(self.output), token=token))
            if len(self.sampled) >= self.options.max_tokens:
                self.finish_reason = FinishReason.LENGTH
            elif self.processed >= self.sequence.context_limit:
                self.finish_reason = FinishReason.CONTEXT

    def take(self, count: int) -> tuple[OutputToken, ...]:
        self.sequence.context.check_thread()
        if self.closed:
            raise RuntimeError("generation is closed")
        if type(count) is not int or count <= 0:
            raise ValueError("output collection count must be positive")
        result = tuple(self.output.popleft() for _ in range(min(count, len(self.output))))
        self.published += len(result)
        return result

    def checkpoint(self) -> Checkpoint:
        self.check()
        if self.pending is not None:
            raise RuntimeError("generation checkpoint requires reconciled work")
        if not self.resident or self.rebuilding:
            raise RuntimeError("numerical checkpoint cannot represent unfinished reconstruction")
        state = Continuation(
            sampled=tuple(self.sampled),
            output=tuple(self.output),
            published=self.published,
            finish=self.finish_reason,
        )
        return Checkpoint(
            self.sequence.checkpoint(), self.prompt, self.selector, self.options, state
        )

    @property
    def rebuilding(self) -> bool:
        return self.processed < self._recovery_position

    def recovery(self) -> Recovery:
        self.check()
        if self.pending is not None:
            raise RuntimeError("reconstruction history requires reconciled work")
        return Recovery(
            processed=max(self.processed, self._recovery_position),
            continuation=Continuation(
                sampled=tuple(self.sampled),
                output=tuple(self.output),
                published=self.published,
                finish=self.finish_reason,
            ),
        )

    def evict(self) -> None:
        """Discard numerical ownership, preserving all accepted logical state."""
        self.check()
        if self.pending is not None:
            raise RuntimeError("eviction requires reconciled work")
        self._recovery_position = max(self.processed, self._recovery_position)
        self.sequence.close()
        self.resident = False

    def restore(self, sequence: ModelSequence) -> None:
        """Attach fresh numerical state; readiness schedules accepted-input replay."""
        self.check()
        if self.resident or self.pending is not None or self.finish_reason is not None:
            raise RuntimeError("only an evicted live generation may restore state")
        if (
            sequence.model is not self.sequence.model
            or sequence.context is not self.selector.context
            or sequence.layout != self.sequence.layout
            or sequence.position != 0
        ):
            raise ValueError("reconstruction requires the same input and fresh model state")
        self.sequence = sequence
        self.resident = True

    def retire(self) -> None:
        """Release terminal numerical state while retaining undelivered output."""
        self.check()
        if self.pending is not None or self.finish_reason is None:
            raise RuntimeError("only a reconciled terminal generation may retire state")
        self.sequence.close()
        self.resident = False

    def fail(self) -> None:
        """Stop failed work while retaining already accepted publication output."""
        self.sequence.context.check_thread()
        if self.pending is not None:
            self.pending.close()
        self._recovery_position = self.accepted_position
        self.finish_reason = FinishReason.FAILED
        self.sequence.close()
        self.resident = False

    def cancel(self) -> None:
        self.check()
        if self.pending is not None:
            self.pending.close()
        self.output.clear()
        self.finish_reason = FinishReason.CANCELLED
        self.sequence.close()
        self.resident = False

    def close(self) -> None:
        self.sequence.context.check_thread()
        if not self.closed:
            if self.pending is not None:
                self.pending.close()
            self.sequence.close()
            self.output.clear()
            self.closed = True


class Checkpoint:
    def __init__(
        self,
        model: ModelCheckpoint,
        prompt: tuple[TokenId, ...],
        selector: SampleSelector,
        options: Options,
        continuation: Continuation,
    ):
        self.model, self.prompt, self.selector = model, prompt, selector
        self.options, self.continuation, self.closed = options, continuation, False

    def fork(self) -> Generation:
        if self.closed:
            raise RuntimeError("generation checkpoint is closed")
        # Restoration installs the complete record; it does not replay published output.
        sequence = self.model.fork()
        try:
            return Generation(
                sequence,
                self.prompt,
                self.selector,
                self.options,
                continuation=self.continuation,
            )
        except BaseException:
            sequence.close()
            raise

    def close(self) -> None:
        if not self.closed:
            self.model.close()
            self.closed = True
