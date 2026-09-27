"""Worker admission must finish constraint validation before allocating input state."""

from types import SimpleNamespace

import pytest
from test_constraints import vocabulary

from engine.generation.constraints import ConstraintPlan
from engine.generation.plain import Options
from engine.models.qwen35.runtime import DenseRuntime
from engine.service.engine import Engine
from engine.serving.runtime import Runtime


class Model(DenseRuntime):
    def __init__(self):
        self.geometry = SimpleNamespace(vocabulary=288)
        self.context = SimpleNamespace(check=lambda: None, check_thread=lambda: None)
        self.inputs_created = 0
        self.inputs_closed = 0

    def input(self, plan):
        self.inputs_created += 1

        def close():
            self.inputs_closed += 1

        return SimpleNamespace(model=self, prompt=plan.tokens, close=close)


def runtime():
    binding = vocabulary()
    model = Model()
    engine = Engine(model)
    ready = SimpleNamespace(
        tokenizer=SimpleNamespace(config=binding.tokenizer.config),
        properties=SimpleNamespace(artifact_identity=binding.tokenizer.config.artifact_identity),
    )
    owner = Runtime(engine, ready)
    options = Options(max_tokens=5, stop_tokens=binding.tokenizer.stop_tokens)
    plan = ConstraintPlan(
        artifact_identity=binding.tokenizer.config.artifact_identity,
        template_identity="fixture",
        grammar='root ::= "yes"',
    )
    return owner, options, plan


def test_compile_and_identity_errors_precede_source_creation_and_queue_admission():
    owner, options, plan = runtime()
    for invalid in (
        plan.model_copy(update={"artifact_identity": "other"}),
        plan.model_copy(update={"converter_sha256": "other"}),
        plan.model_copy(update={"grammar": "invalid grammar"}),
        plan.model_copy(update={"initial_prefix": "no"}),
    ):
        with pytest.raises(ValueError):
            owner.admit((10,), options, invalid)
        assert owner.model.inputs_created == 0
        assert not owner.engine.requests
    owner.engine.close()


def test_admitted_requests_share_only_vocabulary_and_cleanup_failed_admission():
    owner, options, plan = runtime()
    first = owner.admit((10,), options, plan)
    second = owner.admit((10,), options, plan)
    left = owner.engine.requests[first].constraint
    right = owner.engine.requests[second].constraint
    assert left is not right
    assert left.vocabulary is right.vocabulary
    left.stage((ord("y"),)).commit()
    assert right.position == 0
    assert ConstraintPlan.model_validate_json(plan.model_dump_json()) == plan
    with pytest.raises(ValueError, match="EOS identities"):
        owner.admit((10,), Options(max_tokens=5), plan)
    assert owner.model.inputs_created == 3
    assert owner.model.inputs_closed == 1
    owner.engine.close()
    assert owner.model.inputs_closed == 3
