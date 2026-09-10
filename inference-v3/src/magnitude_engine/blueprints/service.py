"""Continuous service construction keeps scheduling separate from model identity."""

from magnitude_engine.composition import Blueprint, blueprint
from magnitude_engine.models.sequence import ModelExecutor
from magnitude_engine.operations.sampling import SampleSelector
from magnitude_engine.service.engine import Engine
from magnitude_engine.service.policy import Limits

__all__ = ["ServiceLimits", "Continuous"]


@blueprint
class ServiceLimits(Blueprint[Limits]):
    max_requests: int = 128
    max_batch: int = 8
    prefill_tokens: int = 512
    decode_share: float = 0.5
    locality_seconds: float = 0.05

    @staticmethod
    def implementation():
        return Limits


@blueprint
class Continuous(Blueprint[Engine]):
    model: Blueprint[ModelExecutor]
    selector: Blueprint[SampleSelector]
    limits: Blueprint[Limits] = ServiceLimits()

    @staticmethod
    def implementation():
        return Engine
