"""One resident encoded representation shared by every computational consumer."""

import math
from contextlib import ExitStack

from magnitude_engine.artifacts.gguf import ByteOrder, Encoding
from magnitude_engine.artifacts.model import GGUFArtifact
from magnitude_engine.numerics.encoded_layout import EncodedLayout, storage_bytes
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import DeviceContext, DType, Prepared, Tensor, TensorSpec


class ResidentWeight:
    def __init__(
        self,
        artifact: GGUFArtifact,
        context: DeviceContext,
        tensor_name: str,
        layout: EncodedLayout | None = None,
    ):
        descriptor = artifact.directory.tensor(tensor_name)
        if artifact.directory.byte_order != ByteOrder.LITTLE:
            raise ValueError("encoded kernels require little-endian GGUF weights")
        self.artifact, self.context, self.descriptor = artifact, context, descriptor
        if layout is None:
            layout = (
                EncodedLayout.AFFINE_PLANES
                if context.backend == Backend.METAL
                and context.subgroup_width == 32
                and descriptor.encoding in (Encoding.Q4_K, Encoding.Q5_K, Encoding.Q6_K)
                and len(descriptor.shape) == 2
                and descriptor.shape[1] % 512 == 0
                else EncodedLayout.GGUF
            )
        self.layout = layout
        self.nbytes = storage_bytes(math.prod(descriptor.shape), descriptor.encoding, layout)
        if layout == EncodedLayout.GGUF:
            self._storage = context.upload_source(
                TensorSpec((self.nbytes,), DType.U8),
                artifact.source,
                artifact.directory.data_offset + descriptor.offset,
            )
        else:
            if context.backend != Backend.METAL or context.subgroup_width != 32:
                raise ValueError("affine plane packing requires a qualified Metal endpoint")
            self._storage = self._pack()

    def _pack(self) -> Tensor:
        from magnitude_engine.numerics.planar_affine import pack

        elements = math.prod(self.descriptor.shape)
        context = self.context
        # Final backing plus at most 4096 original superblocks and an offset.
        # Chunk offsets are runtime operands, not distinct compiled kernels.
        chunk_blocks = 4096
        with ExitStack() as cleanup:
            target = context.allocate(TensorSpec((self.nbytes // 4,), DType.U32))
            cleanup.callback(target.close)
            for first_block in range(0, elements // 256, chunk_blocks):
                count = min(chunk_blocks, elements // 256 - first_block)
                kernel = context.specialize(pack, count, elements, self.descriptor.encoding)
                with ExitStack() as chunk:
                    source = context.upload_source(
                        kernel.signature[0],
                        self.artifact.source,
                        self.artifact.directory.data_offset
                        + self.descriptor.offset
                        + first_block * self.descriptor.encoding.block_bytes,
                    )
                    chunk.callback(source.close)
                    offset = context.indices((first_block * 256,))
                    chunk.callback(offset.close)
                    context.submit((Prepared(context, kernel, (source, target, offset)),)).wait()
            owned = target.view(TensorSpec((self.nbytes,), DType.U8))
        return owned

    def acquire(self, spec: TensorSpec) -> Tensor:
        if spec.nbytes != self.nbytes:
            raise ValueError("consumer representation differs from resident encoded weight size")
        return self._storage.view(spec)

    def close(self) -> None:
        self._storage.close()
