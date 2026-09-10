"""Whole-model forward observations with same-artifact independent logits.

The case retains an accepted history and aborts each measured private advance
after completion. Reset neither shares the live KV tail with a checkpoint nor
includes prefix construction in a decode measurement.
"""

from __future__ import annotations

import hashlib
from contextlib import ExitStack
from enum import StrEnum
from pathlib import Path
from typing import NewType

import numpy as np
from pydantic import Field

from magnitude_engine.artifacts.identity import ArtifactIdentity
from magnitude_engine.data import TokenId
from magnitude_engine.models.qwen35.inputs import InputPlan
from magnitude_engine.models.qwen35.runtime import DenseRuntime
from magnitude_engine.models.sequence import LogitsSelection, ModelBatch, ModelRequest
from magnitude_engine.numerics.policy import NumericalFamily
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import Ticket
from performance.metrics import Latency, Record, Sample, TimingBoundary, TimingPass, Validation
from performance.model_accuracy import (
    FP32_MAX_ERROR,
    FP32_RELATIVE_RMS,
    PrecisionComparison,
    compare,
)

ReferenceIdentity = NewType("ReferenceIdentity", str)


class ReferenceRows(StrEnum):
    ALL = "all"
    LAST = "last"


class LogitsReference(Record):
    path: Path
    identity: ReferenceIdentity = Field(pattern=r"^[0-9a-f]{64}$")
    artifact: ArtifactIdentity = Field(pattern=r"^[0-9a-f]{64}$")
    rows: ReferenceRows = ReferenceRows.ALL
    numerics: NumericalFamily = NumericalFamily.REFERENCE_F32


class ModelWorkload(Record):
    reference: LogitsReference
    history_tokens: int = Field(default=0, ge=0)
    history_chunk: int = Field(default=512, gt=0)
    sequences: int = Field(default=1, gt=0)
    # Mixed references are controls, not interchangeable arithmetic trajectories.
    # The FP32 anchor must describe the identical tokens and artifact.
    accuracy_anchor: LogitsReference | None = None


class ModelMetrics(Record):
    completed_latency: Latency
    device_latency: Latency
    processed_tokens: int = Field(gt=0)
    commands_per_invocation: int = Field(gt=0)
    precision_comparison: PrecisionComparison | None = None

    @property
    def tokens_per_second(self) -> float | None:
        latency = self.completed_latency.median_seconds
        return self.processed_tokens / latency if latency else None


class ModelCase:
    def __init__(self, component: DenseRuntime, workload: ModelWorkload):
        reference = workload.reference
        if reference.artifact != component.artifact_identity:
            raise ValueError("model benchmark reference belongs to another artifact")
        control_family = (
            NumericalFamily.MIXED_BF16
            if component.numerics == NumericalFamily.MIXED_BF16_F32_RESIDUAL
            else component.numerics
        )
        if reference.numerics != control_family:
            raise ValueError("model benchmark reference belongs to another numerical family")
        mixed = reference.numerics == NumericalFamily.MIXED_BF16
        if mixed != (workload.accuracy_anchor is not None):
            raise ValueError(
                "mixed model observations require an explicit paired FP32 accuracy anchor"
            )
        content = reference.path.read_bytes()
        if hashlib.sha256(content).hexdigest() != reference.identity:
            raise ValueError("model benchmark reference checksum differs")
        if len(content) < 8:
            raise ValueError("model reference has no complete header")
        rows, vocabulary = map(int, np.frombuffer(content, "<i4", count=2))
        observed_rows = rows if reference.rows == ReferenceRows.ALL else 1
        if (
            not 0 <= workload.history_tokens < rows <= component.geometry.context_limit
            or vocabulary != component.geometry.vocabulary
            or len(content) != 8 + rows * 4 + observed_rows * vocabulary * 4
        ):
            raise ValueError("model reference geometry differs from the workload or binding")
        tokens = tuple(TokenId(int(value)) for value in np.frombuffer(content, "<i4", rows, 8))
        logits = np.frombuffer(content, "<f4", offset=8 + rows * 4).reshape(
            observed_rows, vocabulary
        )
        self.expected = logits[-1].astype(np.float64)
        if not np.isfinite(self.expected).all():
            raise ValueError("model reference contains nonfinite logits")
        self.anchor: np.ndarray | None = None
        self.precision_comparison: PrecisionComparison | None = None
        anchor = workload.accuracy_anchor
        if anchor is not None:
            if (
                anchor.numerics != NumericalFamily.REFERENCE_F32
                or anchor.artifact != component.artifact_identity
            ):
                raise ValueError(
                    "accuracy anchor requires FP32 interpretation of the same artifact"
                )
            anchor_content = anchor.path.read_bytes()
            if hashlib.sha256(anchor_content).hexdigest() != anchor.identity:
                raise ValueError("accuracy anchor checksum differs")
            anchor_rows = rows if anchor.rows == ReferenceRows.ALL else 1
            if (
                anchor_content[: 8 + rows * 4] != content[: 8 + rows * 4]
                or len(anchor_content) != 8 + rows * 4 + anchor_rows * vocabulary * 4
            ):
                raise ValueError("accuracy anchor tokens or geometry differ from the mixed control")
            anchor_values = (
                np.frombuffer(anchor_content, "<f4", offset=8 + rows * 4)
                .reshape(anchor_rows, vocabulary)[-1]
                .astype(np.float64)
            )
            if not np.isfinite(anchor_values).all():
                raise ValueError("accuracy anchor contains nonfinite logits")
            self.anchor = anchor_values
        self.component, self.workload = component, workload
        self.tokens = tokens[workload.history_tokens :]
        self.batch: ModelBatch | None = None
        self.command_count = 0
        with ExitStack() as cleanup:
            sequences = []
            for _ in range(workload.sequences):
                sequence = component.create(InputPlan.text(tokens))
                cleanup.callback(sequence.close)
                sequences.append(sequence)
            self.sequences = tuple(sequences)
            for start in range(0, workload.history_tokens, workload.history_chunk):
                end = min(start + workload.history_chunk, workload.history_tokens)
                batch = component.prepare(
                    tuple(
                        ModelRequest(sequence, tokens[start:end], LogitsSelection.NONE)
                        for sequence in self.sequences
                    )
                )
                try:
                    ticket = component.context.submit(batch.commands)
                    batch.submitted(ticket)
                    ticket.wait()
                    for advance in batch.advances:
                        advance.commit()
                finally:
                    batch.close()
            self._ownership = cleanup.pop_all()

    @property
    def observation_passes(self) -> tuple[TimingPass, ...]:
        return (
            (TimingPass.COMPLETED,)
            if self.component.context.backend == Backend.LLVM
            else (TimingPass.COMPLETED, TimingPass.DEVICE_EVENTS)
        )

    def reset(self) -> None:
        if self.batch is not None:
            self.batch.close()
            self.batch = None
        if any(sequence.position != self.workload.history_tokens for sequence in self.sequences):
            raise RuntimeError("model benchmark reset changed the accepted prefix")

    def invoke(self, *, timing: bool) -> Ticket:
        if self.batch is not None:
            raise RuntimeError("model benchmark requires reset before invocation")
        self.batch = self.component.prepare(
            tuple(
                ModelRequest(sequence, self.tokens, LogitsSelection.LAST)
                for sequence in self.sequences
            )
        )
        self.command_count = sum(command.dispatches for command in self.batch.commands)
        ticket = self.component.context.submit(self.batch.commands, timing=timing)
        self.batch.submitted(ticket)
        return ticket

    def validate(self, ticket: Ticket) -> Validation:
        self.precision_comparison = None
        if self.batch is None or self.batch.logits is None:
            raise RuntimeError("model benchmark has no pending logits")
        actual = (
            np.frombuffer(
                self.component.context.read(self.batch.logits, after=ticket),
                np.float32,
            )
            .astype(np.float64)
            .reshape(self.workload.sequences, -1)
        )
        if self.anchor is not None:
            if not np.isfinite(actual).all():
                return Validation(
                    passed=False,
                    method="Same-GGUF paired precision controls",
                    maximum_absolute_error=None,
                    maximum_bound_fraction=None,
                    failure="model output contains nonfinite logits",
                )
            self.precision_comparison = compare(self.anchor, self.expected, actual)
            return self.precision_comparison.validation()
        error = actual - self.expected
        finite = bool(np.isfinite(error).all())
        maximum = float(np.max(np.abs(error))) if finite else None
        rms_limit = FP32_RELATIVE_RMS * float(np.sqrt(np.mean(self.expected * self.expected)))
        fraction = (
            float(np.max(np.sqrt(np.mean(error * error, axis=1)))) / max(rms_limit, 1e-12)
            if finite
            else None
        )
        if fraction is not None and maximum is not None:
            fraction = max(fraction, maximum / FP32_MAX_ERROR)
        passed = bool(
            finite
            and maximum is not None
            and maximum < FP32_MAX_ERROR
            and fraction is not None
            and fraction < 1
            and np.all(actual.argmax(axis=1) == self.expected.argmax())
        )
        return Validation(
            passed=passed,
            method="same-GGUF independently decoded FP32 llama.cpp logits: max error < 0.003; "
            "each row RMS error < 1e-4 * reference RMS; matching most likely token per row",
            maximum_absolute_error=maximum,
            maximum_bound_fraction=fraction,
            failure=None if passed else "model logits differ from the qualified reference bound",
        )

    def assess(self, samples: tuple[Sample, ...]) -> ModelMetrics:
        device = tuple(s.device_seconds for s in samples if s.device_seconds is not None)
        return ModelMetrics(
            completed_latency=Latency(
                samples_seconds=tuple(
                    s.completed_seconds for s in samples if s.pass_kind == TimingPass.COMPLETED
                ),
                boundary=TimingBoundary.PREPARE_THROUGH_COMPLETION,
                missing_model_evidence=("no qualified matching whole-model time floor",),
            ),
            device_latency=Latency(
                samples_seconds=device,
                boundary=TimingBoundary.DEVICE_EXECUTION,
                unavailable_observations=() if device else ("device timestamps unavailable",),
            ),
            processed_tokens=len(self.tokens) * self.workload.sequences,
            commands_per_invocation=self.command_count,
            precision_comparison=self.precision_comparison,
        )

    def close(self) -> None:
        self.reset()
        self._ownership.close()


class ModelProcedure:
    def prepare(self, component: DenseRuntime, workload: ModelWorkload) -> ModelCase:
        return ModelCase(component, workload)
