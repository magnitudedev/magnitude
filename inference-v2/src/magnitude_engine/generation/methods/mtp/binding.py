from dataclasses import dataclass

from magnitude_engine.generation.contracts import MethodFactory
from magnitude_engine.models.executor.contracts import ExecutorFactory
from magnitude_engine.models.residency import BoundExecutor, ModelResources

from .runtime import MTPMethod


@dataclass(eq=False)
class MTP(MethodFactory):
    drafter: ExecutorFactory
    max_draft_tokens: int | None

    def bind(self, target: BoundExecutor, resources: ModelResources):
        head = self.drafter.load(resources)
        contract = head.program.drafting
        if contract is None or contract.target is not target.program.program:
            raise ValueError("drafter must borrow this generation target's vocabulary and features")
        if contract.target_feature not in target.program.program.features:
            raise ValueError("target does not expose the drafter's required feature")
        capacity = self.max_draft_tokens or contract.capacity
        if capacity > contract.capacity:
            raise ValueError("draft allowance exceeds the head's supported capacity")
        method = MTPMethod(
            target=target.model,
            head=head.model,
            target_feature=contract.target_feature,
            project=contract.vocabulary.project,
            capacity=capacity,
            budget=resources.budget,
            identity=head.program.descriptor.path,
        )
        return method, capacity, "mtp"
