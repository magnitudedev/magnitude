from dataclasses import dataclass

from magnitude_engine.artifacts.tensors import TensorRegion
from magnitude_engine.models.embeddings.streaming import StreamedEmbedding
from magnitude_engine.models.embeddings.table import AffineRowTable
from magnitude_engine.resources.io.reader import PositionalReader

from ..loading.parameters import resident_embedding
from ..loading.partitions import EmbeddingPartition, OperationResources
from .contracts import EmbeddingFactory


class Resident(EmbeddingFactory):
    def excluded(self, partition: EmbeddingPartition) -> frozenset[str]:
        return frozenset()

    def bind(self, partition: EmbeddingPartition, resources: OperationResources):
        return resident_embedding(partition.module)[0]


@dataclass(eq=False)
class Streamed(EmbeddingFactory):
    cache_bytes: int
    max_pending: int
    reader: PositionalReader

    def excluded(self, partition: EmbeddingPartition) -> frozenset[str]:
        self.table(partition)  # Validate representation before model allocation.
        return frozenset() if partition.retained_for_readout else frozenset(partition.tensors)

    @staticmethod
    def table(partition: EmbeddingPartition) -> AffineRowTable:
        if partition.encoding is None:
            raise ValueError("streamed row tables require an affine encoding")
        tensors = tuple(
            partition.tensors[partition.name + "." + c] for c in ("weight", "scales", "biases")
        )
        if len({len(t.pieces) for t in tensors}) != 1:
            raise ValueError("embedding components must have aligned physical row shards")
        shards = []
        for pieces in zip(*(t.pieces for t in tensors), strict=True):
            records = []
            for tensor, piece in zip(tensors, pieces, strict=True):
                row_bytes = tensor.nbytes // tensor.shape[0]
                if piece.size % row_bytes:
                    raise ValueError("embedding physical shards must end at row boundaries")
                records.append(
                    TensorRegion(
                        tensor.name,
                        (piece.size // row_bytes, tensor.shape[1]),
                        tensor.dtype,
                        piece,
                    )
                )
            shards.append(tuple(records))
        return AffineRowTable(tuple(shards), partition.encoding)

    def bind(self, partition: EmbeddingPartition, resources: OperationResources):
        return resources.own(
            StreamedEmbedding(
                self.table(partition),
                self.reader,
                resources.budget,
                cache_bytes=self.cache_bytes,
                max_pending=self.max_pending,
                owner=partition.name,
            )
        )
