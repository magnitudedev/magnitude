"""Plain generation through one prepared-work and completion lifecycle.

The execution owner submits work. This continuation owns logical history,
pending input and output credit; it never inspects a model's state storage.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated

from pydantic import Field

import ops
from engine.data import Record, TokenId
from engine.generation.constraints import ConstraintState
from engine.models.sequence import (
    LogitsSelection,
    ModelAdvance,
    ModelBatch,
    ModelCheckpoint,
    ModelRequest,
    ModelSequence,
)
from engine.operations.sampling import (
    Draw,
    SamplePosition,
    SamplingSeed,
    SelectionFailure,
    SelectionKind,
)


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
    forced_quantum: int = Field(default=32, ge=0, le=256)


class OutputToken(Record):
    index: int = Field(ge=0)
    token: TokenId = Field(ge=0, le=0x7FFFFFFF)


class Continuation(Record):
    """Published cursor and undelivered output survive preemption together."""

    generated: tuple[TokenId, ...] = ()
    forced_tokens: int = Field(default=0, ge=0)
    forced_runs: tuple[tuple[int, int], ...] = Field(default=(), max_length=256)
    state_only_input_tokens: int = Field(default=0, ge=0)
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
    forced: tuple[TokenId, ...] = ()


class GenerationWork:
    def __init__(self, ready: Ready, advance: ModelAdvance):
        self.generation, self.kind, self.count = ready.generation, ready.kind, len(ready.tokens)
        self.advance = advance
        self.forced = ready.forced
        self.completion: ops.Completion | None = None
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
            sampled = self.advance.read_sample()
            if sampled is not None:
                token, status = sampled
                if status:
                    try:
                        reason = SelectionFailure(status)
                    except ValueError as error:
                        raise RuntimeError(f"sampling returned unknown status {status}") from error
                    raise ValueError(f"distribution cannot be selected: {reason.name}")
                selected = token
            if self.forced and selected is not None:
                raise RuntimeError("forced model advance unexpectedly returned a sample")
            accepted = self.forced or (() if selected is None else (TokenId(selected),))
            transition = (
                generation.constraint.stage(accepted)
                if accepted and generation.constraint is not None
                else None
            )
            self.advance.commit()
            if transition is not None:
                transition.commit()
            for index, token in enumerate(accepted):
                generation._accept(token, terminal=index == len(accepted) - 1)
            generation.forced_tokens += len(self.forced)
            if self.forced:
                length = len(self.forced)
                generation.forced_runs[length] = generation.forced_runs.get(length, 0) + 1
                generation.state_only_input_tokens += self.count
        except BaseException:
            generation.finish_reason = FinishReason.FAILED
            raise
        finally:
            self.close()

    def close(self) -> None:
        if not self.closed:
            self.advance.close()
            if self.generation.pending is self:
                self.generation.pending = None
            self.closed = True


class GenerationBatch:
    def __init__(self, execution: ModelBatch, works: tuple[GenerationWork, ...]):
        self.execution, self.works = execution, works
        self.completion = execution.completion
        self.closed = False
        for work in works:
            work.completion = self.completion

    @classmethod
    def prepare(cls, ready: tuple[Ready, ...]) -> GenerationBatch:
        if not ready or len({id(item.generation) for item in ready}) != len(ready):
            raise ValueError("a generation batch requires distinct ready requests")
        first = ready[0].generation
        model = first.sequence.model
        for item in ready:
            generation = item.generation
            if generation.sequence.model is not model:
                raise ValueError("ready requests do not share a model binding")
            if generation.ready(len(item.tokens)) != item:
                raise ValueError("generation proposal is no longer ready")
        execution = model.prepare(
            tuple(
                ModelRequest(
                    item.generation.sequence,
                    item.tokens,
                    item.selection,
                    None
                    if item.selection == LogitsSelection.NONE
                    else Draw(
                        kind=item.generation.options.selection,
                        seed=item.generation.options.seed,
                        position=item.sample_position,
                    ).words(),
                    None
                    if item.selection == LogitsSelection.NONE or item.generation.constraint is None
                    else item.generation.constraint,
                )
                for item in ready
            )
        )
        works = tuple(
            GenerationWork(item, advance)
            for item, advance in zip(ready, execution.advances, strict=True)
        )
        for item, work in zip(ready, works, strict=True):
            item.generation.pending = work
        return cls(execution, works)

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
            for work in self.works:
                work.close()
            self.execution.close()
            self.closed = True


class Generation:
    def __init__(
        self,
        sequence: ModelSequence,
        prompt: tuple[TokenId, ...],
        options: Options,
        *,
        continuation: Continuation | None = None,
        recovery: Recovery | None = None,
        constraint: ConstraintState | None = None,
    ):
        if continuation is not None and recovery is not None:
            raise ValueError("supply either resident continuation or reconstruction history")
        if recovery is not None:
            continuation = recovery.continuation
        continuation = Continuation() if continuation is None else continuation
        accepted_position = sequence.position if recovery is None else recovery.processed
        if (
            len(prompt) != sequence.layout.count
            or not prompt
            or len(prompt) > sequence.context_limit
        ):
            raise ValueError("generation requires a nonempty prompt inside the model context")
        if (
            not 0 <= sequence.position <= accepted_position <= sequence.context_limit
            or (
                continuation.generated
                and accepted_position != len(prompt) + len(continuation.generated) - 1
            )
            or (not continuation.generated and accepted_position >= len(prompt))
            or len(continuation.output) > options.output_capacity
            or continuation.forced_tokens > len(continuation.generated)
            or len(dict(continuation.forced_runs)) != len(continuation.forced_runs)
            or any(not 1 <= length <= 256 or count <= 0 for length, count in continuation.forced_runs)
            or sum(length * count for length, count in continuation.forced_runs) != continuation.forced_tokens
            or tuple(item.index for item in continuation.output)
            != tuple(
                range(continuation.published, continuation.published + len(continuation.output))
            )
        ):
            raise ValueError("generation continuation and processed/publication boundaries differ")
        self.constraint = None if constraint is None else constraint.fork()
        if self.constraint is not None:
            if self.constraint.vocabulary.tokenizer.stop_tokens != options.stop_tokens:
                raise ValueError("generation and constraint EOS identities differ")
            if self.constraint.position == 0 and continuation.generated:
                self.constraint.stage(continuation.generated).commit()
            if self.constraint.position != len(continuation.generated):
                raise ValueError("constraint progress differs from accepted generation history")
        self.sequence, self.prompt, self.options = sequence, prompt, options
        self.generated = list(continuation.generated)
        self.forced_tokens = continuation.forced_tokens
        self.forced_runs = dict(continuation.forced_runs)
        self.state_only_input_tokens = continuation.state_only_input_tokens
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
        return self.generated[-1] if self.generated and self.finish_reason is None else None

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
            accepted = self.prompt + tuple(self.generated)
            end = self.sequence.layout.chunk_end(position, self._recovery_position, allowance)
            return Ready(
                self,
                WorkKind.REPLAY,
                accepted[position:end],
                LogitsSelection.NONE,
                position,
                SamplePosition(len(self.generated)),
            )
        if position < len(self.prompt):
            end = self.sequence.layout.chunk_end(position, len(self.prompt), allowance)
            tokens = self.prompt[position:end]
            selection = LogitsSelection.LAST if end == len(self.prompt) else LogitsSelection.NONE
            kind = WorkKind.PREFILL
            forced = self._forced(1) if selection == LogitsSelection.LAST else ()
        else:
            if self.pending_input is None or position != len(self.prompt) + len(self.generated) - 1:
                raise RuntimeError("generation history and model continuation disagree")
            tokens, selection, kind = (self.pending_input,), LogitsSelection.LAST, WorkKind.DECODE
            forced = self._forced(min(allowance, self.sequence.context_limit - position))
            if forced:
                tokens = (self.pending_input, *forced[:-1])
        if forced:
            selection = LogitsSelection.NONE
        return Ready(
            self, kind, tokens, selection, position, SamplePosition(len(self.generated)), forced
        )

    def _forced(self, allowance: int) -> tuple[TokenId, ...]:
        limit = min(
            allowance,
            self.options.forced_quantum,
            self.options.max_tokens - len(self.generated),
            self.options.output_capacity - len(self.output),
        )
        return self.constraint.forced(limit) if self.constraint is not None and limit > 0 else ()

    def _accept(self, token: TokenId, *, terminal: bool = True) -> None:
        self.generated.append(token)
        if token in self.options.stop_tokens:
            self.finish_reason = FinishReason.STOP
        else:
            self.output.append(OutputToken(index=self.published + len(self.output), token=token))
            if not terminal:
                return
            if len(self.generated) >= self.options.max_tokens:
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
            generated=tuple(self.generated),
            forced_tokens=self.forced_tokens,
            forced_runs=tuple(sorted(self.forced_runs.items())),
            state_only_input_tokens=self.state_only_input_tokens,
            output=tuple(self.output),
            published=self.published,
            finish=self.finish_reason,
        )
        return Checkpoint(
            self.sequence.checkpoint(),
            self.prompt,
            self.options,
            state,
            None if self.constraint is None else self.constraint.fork(),
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
                generated=tuple(self.generated),
                forced_tokens=self.forced_tokens,
                forced_runs=tuple(sorted(self.forced_runs.items())),
                state_only_input_tokens=self.state_only_input_tokens,
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
        options: Options,
        continuation: Continuation,
        constraint: ConstraintState | None = None,
    ):
        self.constraint = constraint
        self.model, self.prompt = model, prompt
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
                self.options,
                continuation=self.continuation,
                constraint=self.constraint,
            )
        except BaseException:
            sequence.close()
            raise

    def close(self) -> None:
        if not self.closed:
            self.model.close()
            self.closed = True
