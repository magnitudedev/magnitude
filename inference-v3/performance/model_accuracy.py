"""Paired precision controls for complete model outputs.

The independent mixed-precision model sets a workload-specific accuracy envelope
against an independent FP32 anchor. A candidate cannot set its own error budget.
This is numerical agreement on a finite workload, not a task-quality guarantee.
"""

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from performance.metrics import Record, Validation

type Vector = NDArray[np.float64]

# The existing FP32 model guard. The KL floor follows from Hoeffding's lemma:
# |candidate_logit - anchor_logit| <= delta implies KL(anchor || candidate)
# <= (2*delta)^2/8. No new reduced-precision epsilon is introduced.
FP32_MAX_ERROR = 0.003
FP32_RELATIVE_RMS = 1e-4


class LogitError(Record):
    maximum_absolute: float = Field(ge=0, allow_inf_nan=False)
    rms: float = Field(ge=0, allow_inf_nan=False)
    anchor_to_candidate_kl: float = Field(ge=0, allow_inf_nan=False)
    same_top_prediction: bool


class PrecisionLimits(Record):
    maximum_absolute: float = Field(gt=0, allow_inf_nan=False)
    rms: float = Field(gt=0, allow_inf_nan=False)
    anchor_to_candidate_kl: float = Field(gt=0, allow_inf_nan=False)


class PrecisionComparison(Record):
    control: LogitError
    limits: PrecisionLimits
    candidates: tuple[LogitError, ...] = Field(min_length=1)

    def validation(self) -> Validation:
        fraction = max(
            max(
                candidate.maximum_absolute / self.limits.maximum_absolute,
                candidate.rms / self.limits.rms,
                candidate.anchor_to_candidate_kl / self.limits.anchor_to_candidate_kl,
            )
            for candidate in self.candidates
        )
        passed = (
            self.control.same_top_prediction
            and all(candidate.same_top_prediction for candidate in self.candidates)
            and fraction <= 1
        )
        return Validation(
            passed=passed,
            method=(
                "Same-GGUF paired precision controls: candidate maximum/RMS logit error and "
                "KL(FP32 anchor || candidate) must not exceed the independent BF16 control's "
                "errors, with floors from the unchanged FP32 guard; control and each candidate "
                "must preserve the FP32 top prediction"
            ),
            maximum_absolute_error=max(candidate.maximum_absolute for candidate in self.candidates),
            maximum_bound_fraction=fraction,
            failure=None if passed else "model output exceeds the independent precision control",
        )


def log_probabilities(logits: Vector) -> Vector:
    shifted = logits - np.max(logits)
    return shifted - np.log(np.exp(shifted).sum())


def error(candidate: Vector, anchor: Vector) -> LogitError:
    if candidate.ndim != 1 or anchor.shape != candidate.shape or not candidate.size:
        raise ValueError("logit comparison requires equal nonempty vocabulary vectors")
    if not np.isfinite(candidate).all() or not np.isfinite(anchor).all():
        raise ValueError("logit comparison requires finite values")
    difference = candidate - anchor
    anchor_logp = log_probabilities(anchor)
    divergence = float(np.sum(np.exp(anchor_logp) * (anchor_logp - log_probabilities(candidate))))
    return LogitError(
        maximum_absolute=float(np.max(np.abs(difference))),
        rms=float(np.sqrt(np.mean(difference * difference))),
        # Roundoff in the FP64 distribution comparison can make exact equality
        # slightly negative. It cannot make a positive divergence pass a limit.
        anchor_to_candidate_kl=max(0.0, divergence),
        same_top_prediction=bool(candidate.argmax() == anchor.argmax()),
    )


def compare(anchor: Vector, control: Vector, candidates: Vector) -> PrecisionComparison:
    if candidates.ndim != 2 or candidates.shape[0] == 0:
        raise ValueError("model comparison requires one or more independent candidate rows")
    observed_control = error(control, anchor)
    return PrecisionComparison(
        control=observed_control,
        limits=PrecisionLimits(
            maximum_absolute=max(FP32_MAX_ERROR, observed_control.maximum_absolute),
            rms=max(
                FP32_RELATIVE_RMS * float(np.sqrt(np.mean(anchor * anchor))),
                observed_control.rms,
                1e-12,
            ),
            anchor_to_candidate_kl=max(
                FP32_MAX_ERROR**2 / 2, observed_control.anchor_to_candidate_kl
            ),
        ),
        candidates=tuple(error(candidate, anchor) for candidate in candidates),
    )
