"""Linked model/method advancement and request-local sampling state."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Protocol, cast

import mlx.core as mx

from magnitude_engine.components import component
from magnitude_engine.models.inputs import ModelInputs
from magnitude_engine.models.operations import Task, accept, complete, forward, observe
from magnitude_engine.models.runtime import (
    ForwardRequest,
    ModelAdvance,
    ModelRuntime,
    ModelSequence,
)
from magnitude_engine.resources.retention import RetainedStorage

from .acceptance import accept_prefix
from .constraint_spec import ConstraintError, ConstraintSpec
from .constraints import ConstraintCompiler, TokenConstraint
from .execution import Continuation, execute, run, serve
from .methods.contracts import (
    CausalSession,
    GenerationMethod,
    MethodCheckpoint,
    MethodSession,
    Verification,
)
from .proposals import Proposal
from .sampling import SequenceSampler
from .sampling_policy import SamplingPolicy


class ModelCheckpoint(Protocol):
    @property
    def reclaimable(self) -> bool: ...

    length: int
    closed: bool

    def close(self) -> None: ...
    def retained_storage(self) -> tuple[RetainedStorage, ...]: ...


class GenerationCheckpoint[C: ModelCheckpoint]:
    def __init__(
        self,
        model: C,
        method: MethodCheckpoint,
        identity: str,
        tokens: tuple[int, ...],
        model_domain: object,
    ):
        self.model = model
        self.method = method
        self.identity = identity
        self.model_domain = model_domain
        self.tokens = tokens
        self.length = len(tokens)
        self.closed = False

    @property
    def reclaimable(self) -> bool:
        return self.model.reclaimable and self.method.reclaimable

    def retained_storage(self):
        return (*self.model.retained_storage(), *self.method.retained_storage())

    def close(self) -> None:
        if self.closed:
            return
        errors = []
        for checkpoint in (self.method, self.model):
            try:
                checkpoint.close()
            except BaseException as error:
                errors.append(error)
        self.closed = True
        if errors:
            raise BaseExceptionGroup("generation checkpoint release failed", errors)


@dataclass(frozen=True)
class GenerationResult:
    tokens: tuple[int, ...]
    proposed: int
    accepted: int
    finish_reason: str | None
    evaluated_inputs: int
    forced: int = 0


@dataclass(frozen=True)
class PreparedGeneration:
    inputs: ModelInputs
    proposal: Proposal
    forced: tuple[int, ...]
    request: ForwardRequest


@dataclass(frozen=True)
class GenerationService:
    outcome: GenerationResult | MemoryError | ConstraintError | None
    elapsed_ns: int
    batch_size: int


@dataclass(frozen=True)
class PrefillService:
    outcome: int | MemoryError | ConstraintError
    elapsed_ns: int
    batch_size: int


class GenerationSequence[S, C: ModelCheckpoint]:
    def __init__(
        self,
        runtime: GenerationRuntime[S, C],
        model: ModelSequence[S, C],
        method: MethodSession,
        prompt: tuple[int, ...],
        sampler: SequenceSampler,
        max_tokens: int,
        stop_tokens: tuple[int, ...],
        prefilled: int,
        constraint: TokenConstraint | None,
    ):
        self.runtime = runtime
        self.model = model
        self.method = method
        self.context = list(prompt)
        self.prompt_length = len(prompt)
        self.prefilled = prefilled
        self.target_position = prefilled
        self.sampler = sampler
        self.max_tokens = max_tokens
        self.stop_tokens = stop_tokens
        self.generated = 0
        self.finished = max_tokens == 0
        self.closed = False
        self.failed = False
        self.constraint = constraint
        self._round: Continuation[GenerationResult] | None = None
        self._round_allowance = 0

    @property
    def prefill_remaining(self) -> int:
        return self.prompt_length - 1 - self.prefilled

    def reserve_prompt(self) -> None:
        """Secure target state through the known prompt and its first decode input.

        This does not execute prompt tokens. Allocation failures leave admission
        free to close this prepared continuation and retry when peers release memory.
        """
        self.runtime.model.reserve(self.model, self.prefill_remaining + 1)

    def prefill(self, allowance: int) -> int:
        """Advance one bounded prompt chunk, then return control to service policy."""
        return run(self.prefill_task(allowance))

    def prefill_task(self, allowance: int) -> Task[int]:
        """Expose target work before publishing completed prompt features to the method."""
        self.model.check()
        if self.closed or self.failed:
            raise RuntimeError("generation sequence cannot prefill")
        if type(allowance) is not int or allowance < 1:
            raise ValueError("prefill allowance must be a positive integer")
        count = min(allowance, self.prefill_remaining)
        if count == 0:
            return 0
        tokens = tuple(self.context[self.prefilled : self.prefilled + count])
        try:
            inputs = ModelInputs.from_tokens(tokens)
            advance = yield from forward(
                self.model,
                inputs,
                ForwardRequest(False, self.method.prefill_features, committed_inputs=count),
            )
            yield from complete(advance)
            advance.accept(count)
            self.method.prefill(tokens, advance.output.features)
            self.prefilled += count
            self.target_position += count
            return count
        except BaseException:
            self.failed = True
            raise

    def step(self, token_allowance: int = 1) -> GenerationResult:
        self._check_step(token_allowance)
        try:
            if self._round is None:
                return run(self.round(token_allowance))
            serve((self._round,))
            outcome = self._round.result
            self._round = None
            if isinstance(outcome, (MemoryError, ConstraintError)):
                raise outcome
            if outcome is None:
                raise RuntimeError("generation round did not complete")
            return outcome
        except BaseException:
            self.failed = True
            raise

    def round(self, token_allowance: int) -> Task[GenerationResult]:
        self._check_step(token_allowance)
        if (
            self.constraint is None
            and not self.sampler.policy.uses_history
            and isinstance(self.method, CausalSession)
        ):
            result = yield from self.method.decode_causal(
                self.runtime.model,
                self.model,
                anchor=self.context[-1],
                position=len(self.context),
                sampler=self.sampler,
                allowance=min(token_allowance, self.max_tokens - self.generated),
                stop_tokens=self.stop_tokens,
            )
            self.target_position += result.evaluated_inputs
            reason = self._publish(result.tokens)
            return GenerationResult(result.tokens, 0, 0, reason, result.evaluated_inputs)
        self.runtime.model.reserve(
            self.model, min(token_allowance, self.max_tokens - self.generated)
        )
        work = yield from self._prepare_step(token_allowance)
        advance = yield from forward(self.model, work.inputs, work.request)
        return (yield from self._finish_step(work, advance))

    def _check_step(self, token_allowance: int) -> None:
        self.model.check()
        if self.closed or self.failed or self.finished:
            raise RuntimeError("generation sequence cannot advance")
        if self.prefill_remaining:
            raise RuntimeError("generation prompt prefill is incomplete")
        if type(token_allowance) is not int or token_allowance < 1:
            raise ValueError("token allowance must be a positive integer")
        if self._round is not None and token_allowance < self._round_allowance:
            raise ValueError("resuming a round must preserve its reserved output allowance")

    def _prepare_step(self, token_allowance: int) -> Task[PreparedGeneration]:
        self._check_step(token_allowance)
        remaining = self.max_tokens - self.generated
        limit = min(token_allowance, remaining) - 1
        try:
            forced = () if self.constraint is None else self.constraint.forced()[: limit + 1]
            stop = next((i for i, token in enumerate(forced) if token in self.stop_tokens), None)
            if stop is not None:
                forced = forced[: stop + 1]
            proposed = (
                Proposal.from_tokens(forced[:-1])
                if forced
                else (yield from self.method.propose(self.context, limit))
            )
            if proposed.count > limit:
                raise RuntimeError("generation method exceeded its proposal allowance")
            anchor = mx.array([[self.context[-1]]], dtype=mx.int32)
            inputs = ModelInputs(
                mx.concatenate([anchor, proposed.tokens[None]], axis=1)
                if proposed.count
                else anchor
            )
            return PreparedGeneration(
                inputs,
                proposed,
                forced,
                ForwardRequest(
                    logits=not bool(forced),
                    features=self.method.features,
                    committed_inputs=inputs.count if forced else 1,
                ),
            )
        except BaseException:
            self.failed = True
            raise

    def _finish_step(
        self, work: PreparedGeneration, advance: ModelAdvance[S, C]
    ) -> Task[GenerationResult]:
        inputs, proposed, forced = work.inputs, work.proposal, work.forced
        try:
            logits = advance.output.logits
            if forced:
                count, bonus = len(forced) - 1, forced[-1]
            else:
                if logits is None or logits.ndim != 3 or logits.shape[:2] != (1, inputs.count):
                    raise RuntimeError(
                        "generation requires one logit vector per verification input"
                    )
                samples = self._sample_verification(logits[0], proposed)
                if proposed.count:
                    accepted = accept_prefix(proposed.tokens, samples, self.stop_tokens)
                    yield from observe(accepted.count, accepted.bonus)
                    count = cast(int, accepted.count.item())
                    bonus = cast(int, accepted.bonus.item())
                else:
                    yield from observe(samples)
                    count, bonus = 0, cast(int, samples.item())
            host_proposal = proposed.host()
            emitted = (*host_proposal[:count], bonus)
            if self.constraint is not None:
                for token in emitted:
                    if not self.constraint.consume(token):
                        raise RuntimeError("constraint rejected a committed token")
            yield from accept(advance, count + 1)
            self.target_position += count + 1
            self.method.observe(
                Verification(
                    (self.context[-1], *host_proposal), count + 1, bonus, advance.output.features
                )
            )
            reason = self._publish(emitted)
            return GenerationResult(
                emitted,
                0 if forced else proposed.count,
                0 if forced else count,
                reason,
                inputs.count,
                len(forced),
            )
        except BaseException:
            self.failed = True
            raise

    def _publish(self, emitted: tuple[int, ...]) -> str | None:
        self.context.extend(emitted)
        self.sampler.observe(emitted)
        self.generated += len(emitted)
        reason = (
            "stop"
            if emitted[-1] in self.stop_tokens
            else ("length" if self.generated == self.max_tokens else None)
        )
        self.finished = reason is not None
        # An unfinished sequence leaves exactly one causal anchor to consume.
        # EOS lookahead may consume the final emitted token before stopping.
        if self.target_position != len(self.context) - 1 and not (
            reason == "stop" and self.target_position == len(self.context)
        ):
            raise RuntimeError("target consumed a prefix inconsistent with emitted context")
        return reason

    def _sample_verification(self, logits: mx.array, proposed: Proposal) -> mx.array:
        if not proposed.count:
            # With no candidates, sampling uses the committed history and matcher.
            # There is no hypothetical token prefix to fork or advance.
            raw = logits[0] if self.constraint is None else self.constraint.apply(logits[0])
            return self.sampler.sample(raw, len(self.context)).reshape(1)
        preview = None if self.constraint is None else self.constraint.fork()
        # Only constrained verification needs proposal values before sampling. The ordinary
        # path retains device proposals until prefix acceptance has completed.
        tokens = () if preview is None else proposed.host()
        samples = []
        try:
            for index in range(proposed.count + 1):
                raw = logits[index] if preview is None else preview.apply(logits[index])
                samples.append(
                    self.sampler.sample(
                        raw, len(self.context) + index, preview_tokens=proposed.tokens[:index]
                    )
                )
                if preview is not None and index < proposed.count:
                    if tokens[index] in self.stop_tokens or not preview.consume(tokens[index]):
                        # The first invalid candidate cannot be accepted under its mask. Later
                        # samples are unreachable, and the poisoned fork is discarded as a whole.
                        samples.extend([samples[-1]] * (proposed.count - index))
                        break
            return mx.stack(samples)
        finally:
            if preview is not None:
                preview.close()

    def checkpoint(self) -> GenerationCheckpoint[C]:
        self.model.check()
        if self.failed or self.closed:
            raise RuntimeError("cannot retain failed or closed generation state")
        if self._round is not None:
            raise RuntimeError("checkpoint requires a completed generation round")
        model = self.model.checkpoint()
        try:
            method = self.method.checkpoint()
        except BaseException:
            model.close()
            raise
        boundary = self.target_position
        if model.length != boundary:
            method.close()
            model.close()
            raise RuntimeError("target and generation boundary disagree")
        return GenerationCheckpoint(
            model,
            method,
            self.runtime.method.identity,
            tuple(self.context[:boundary]),
            self.runtime.model.checkpoint_domain,
        )

    def close(self) -> None:
        if self.closed:
            return
        errors = []
        if self._round is not None:
            self._round.close()
            self._round = None
        for resource in (self.constraint, self.method, self.model):
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as error:
                errors.append(error)
        self.closed = True
        if errors:
            raise BaseExceptionGroup("generation sequence release failed", errors)


class GenerationRuntime[S, C: ModelCheckpoint]:
    def __init__(
        self,
        model: ModelRuntime[S, C],
        method: GenerationMethod,
        constraints: ConstraintCompiler | None = None,
    ):
        self.model = model
        self.method = method
        self.constraints = constraints

    def prefill_groups(
        self,
        sequences: tuple[GenerationSequence[S, C], ...],
    ) -> tuple[tuple[GenerationSequence[S, C], ...], ...]:
        """Describe compatible prompt work before policy divides its token allowance."""
        groups: list[list[GenerationSequence[S, C]]] = []
        for sequence in sequences:
            if sequence.runtime is not self:
                raise ValueError("prompt grouping requires this runtime's sequences")
            for group in groups:
                if self.model.can_batch(tuple(row.model for row in (*group, sequence))):
                    group.append(sequence)
                    break
            else:
                groups.append([sequence])
        return tuple(tuple(group) for group in groups)

    @component("SCHEDULING:PREFILL:MAG:CHUNKED")
    def prefill_many(
        self,
        sequences: tuple[GenerationSequence[S, C], ...],
        token_allowances: tuple[int, ...],
        *,
        clock: Callable[[], int] = perf_counter_ns,
    ) -> tuple[PrefillService, ...]:
        """Complete bounded prompt work through ordinary compatible model operations."""
        if (
            len(sequences) != len(token_allowances)
            or len({id(s) for s in sequences}) != len(sequences)
            or any(s.runtime is not self for s in sequences)
        ):
            raise ValueError("prefill service requires distinct owned rows and aligned allowances")
        results = execute(
            tuple(
                sequence.prefill_task(limit)
                for sequence, limit in zip(sequences, token_allowances, strict=True)
            ),
            clock=clock,
        )
        services = []
        for sequence, result in zip(sequences, results, strict=True):
            outcome = result.result
            if isinstance(outcome, (MemoryError, ConstraintError)):
                sequence.failed = True
            if outcome is None:
                raise RuntimeError("prompt service did not complete")
            services.append(PrefillService(outcome, result.elapsed_ns, result.batch_size))
        return tuple(services)

    def step_many(
        self,
        sequences: tuple[GenerationSequence[S, C], ...],
        token_allowances: tuple[int, ...],
        *,
        clock: Callable[[], int] = perf_counter_ns,
        budget_ns: int | None = None,
    ) -> tuple[GenerationService, ...]:
        """Serve ready rounds, returning when results are publishable or the budget expires.

        Scheduling supplies members and allowances. Each bound method prepares its
        own proposal and observes its own accepted target features. Compatible model
        operations share physical calls; row-local sampling/constraints stay independent.
        An unfinished row returns no outcome and retains its round for the next service.
        Active service attributes shared neural duration to each participating row,
        plus that row's proposal and reconciliation duration, excluding peer-only work.
        """
        if len(sequences) != len(token_allowances) or len({id(s) for s in sequences}) != len(
            sequences
        ):
            raise ValueError("generation service requires distinct rows and aligned allowances")
        if any(s.runtime is not self for s in sequences):
            raise ValueError("generation service rows belong to another runtime")
        if len(sequences) == 1 and budget_ns is None:
            start = clock()
            try:
                outcome = sequences[0].step(token_allowances[0])
            except (MemoryError, ConstraintError) as error:
                outcome = error
            return (GenerationService(outcome, clock() - start, 1),)
        for sequence, limit in zip(sequences, token_allowances, strict=True):
            sequence._check_step(limit)
            if sequence._round is None:
                sequence._round_allowance = min(limit, sequence.max_tokens - sequence.generated)
                sequence._round = Continuation(sequence.round(limit))
        rounds = tuple(sequence._round for sequence in sequences if sequence._round is not None)
        previous = tuple(row.elapsed_ns for row in rounds)
        serve(rounds, clock=clock, budget_ns=budget_ns)
        services = []
        for sequence, result, before in zip(sequences, rounds, previous, strict=True):
            outcome = result.result
            if isinstance(outcome, (MemoryError, ConstraintError)):
                sequence.failed = True
            if result.done:
                sequence._round = None
            services.append(
                GenerationService(outcome, result.elapsed_ns - before, result.batch_size)
            )
        return tuple(services)

    def create(
        self,
        prompt: tuple[int, ...],
        sampling: SamplingPolicy,
        max_tokens: int,
        stop_tokens: tuple[int, ...] = (),
        chunk_size: int = 512,
        checkpoint: GenerationCheckpoint[C] | None = None,
        *,
        constraint: ConstraintSpec | None = None,
    ) -> GenerationSequence[S, C]:
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("prefill chunk size must be a positive integer")
        sequence = self.prepare(
            prompt, sampling, max_tokens, stop_tokens, checkpoint, constraint=constraint
        )
        try:
            while sequence.prefill_remaining:
                sequence.prefill(chunk_size)
            return sequence
        except BaseException:
            sequence.close()
            raise

    def prepare(
        self,
        prompt: tuple[int, ...],
        sampling: SamplingPolicy,
        max_tokens: int,
        stop_tokens: tuple[int, ...] = (),
        checkpoint: GenerationCheckpoint[C] | None = None,
        *,
        constraint: ConstraintSpec | None = None,
    ) -> GenerationSequence[S, C]:
        """Create or restore linked state without executing any prompt tokens."""
        if (
            not prompt
            or any(type(t) is not int or not 0 <= t < 2**31 for t in (*prompt, *stop_tokens))
            or type(max_tokens) is not int
            or max_tokens < 0
        ):
            raise ValueError("invalid generation input or allowance")
        start = 0
        if checkpoint is not None:
            if (
                checkpoint.closed
                or checkpoint.model_domain is not self.model.checkpoint_domain
                or checkpoint.identity != self.method.identity
                or checkpoint.length >= len(prompt)
                or prompt[: checkpoint.length] != checkpoint.tokens
            ):
                raise ValueError("generation checkpoint does not match this method and prompt")
            start = checkpoint.length
        model = self.model.create(None if checkpoint is None else checkpoint.model)
        method = None
        matcher = None
        try:
            if constraint is not None:
                if self.constraints is None:
                    raise ConstraintError("this runtime has no constraint compiler")
                matcher = self.constraints.create(constraint)
            method = self.method.create(
                None if checkpoint is None else checkpoint.method, target=self.model
            )
            if not (method.features | method.prefill_features) <= self.model.program.features:
                raise ValueError("target lacks generation method's required features")
            sampler = SequenceSampler(sampling)
            sampler.observe(prompt)
            return GenerationSequence(
                self, model, method, prompt, sampler, max_tokens, stop_tokens, start, matcher
            )
        except BaseException:
            model.close()
            if method is not None:
                method.close()
            if matcher is not None:
                matcher.close()
            raise
