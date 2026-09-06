from dataclasses import dataclass

from magnitude_engine.resources.io.reader import PositionalReader

from ..loading.parameters import resident_experts
from ..loading.partitions import ExpertPartition, OperationResources
from .bank import ExpertBank, ExpertSource, ProjectionSource
from .computation import GatedExpertMath
from .contracts import ExpertFactory
from .streaming import StreamedExperts


class Resident(ExpertFactory):
    def excluded(self, partition: ExpertPartition) -> frozenset[str]:
        return frozenset()

    def bind(self, partition: ExpertPartition, resources: OperationResources):
        return resident_experts(partition.up, partition.gate, partition.down, partition.activation)


@dataclass(eq=False)
class Streamed(ExpertFactory):
    slots: int
    reader: PositionalReader

    def source(self, partition: ExpertPartition) -> ExpertSource:
        projections = []
        encodings = []
        for projection in ("up_proj", "gate_proj", "down_proj"):
            name = partition.name + "." + projection
            projections.append(
                ProjectionSource(
                    *(partition.tensors[name + "." + c] for c in ("weight", "scales", "biases"))
                )
            )
            encodings.append(partition.encodings[name + ".weight"])
        if len(set(encodings)) != 1:
            raise ValueError("streamed expert banks require compatible projection encodings")
        return ExpertSource(projections[0], projections[1], projections[2], encodings[0])

    def excluded(self, partition: ExpertPartition) -> frozenset[str]:
        self.source(partition)
        return frozenset(partition.tensors)

    def bind(self, partition: ExpertPartition, resources: OperationResources):
        source = self.source(partition)
        bank = resources.own(
            ExpertBank(
                source,
                min(self.slots, source.experts),
                resources.budget,
                owner=partition.name,
            )
        )
        geometry = (
            source.encoding,
            tuple(
                (tensor.dtype, tensor.shape)
                for projection in source.projections
                for tensor in projection.components
            ),
        )
        scratch = resources.shared(
            self,
            geometry,
            lambda: resources.own(
                ExpertBank(
                    source,
                    source.experts,
                    resources.budget,
                    owner=partition.name + ".prefill",
                )
            ),
        )
        return StreamedExperts(
            source,
            GatedExpertMath(partition.activation),
            bank=bank,
            scratch=scratch,
            reader=self.reader,
        )
