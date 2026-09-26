"""Encode floating projection records with bounded source residency."""

import math

import mlx.core as mx

from magnitude_engine.resources.budget import MemoryBudget
from magnitude_engine.resources.io.reader import PositionalReader

from .layouts import LogicalTensor
from .materialization import ResidentMaterializer, ResidentTensors
from .quantization import AffineEncoding
from .tensors import DTYPE_BYTES


class AffineMaterializer:
    """Reserve final storage first, then read/encode one projection at a time.

    This avoids keeping a floating head checkpoint resident alongside its encoded
    replacement. Non-projection tensors retain their original dtype and convention.
    """

    def __init__(
        self,
        budget: MemoryBudget,
        reader: PositionalReader,
        encodings: dict[str, AffineEncoding],
        *,
        owner: str,
    ):
        self.budget, self.reader, self.encodings, self.owner = budget, reader, encodings, owner

    def materialize(self, tensors: dict[str, LogicalTensor]) -> ResidentTensors:
        if not self.encodings.keys() <= tensors.keys():
            raise ValueError("encoding plan names tensors outside its partition")
        final_bytes = 0
        for name, tensor in tensors.items():
            encoding = self.encodings.get(name)
            if encoding is None:
                final_bytes += tensor.nbytes
                continue
            if (
                not name.endswith(".weight")
                or len(tensor.shape) < 2
                or tensor.dtype not in ("F32", "F16", "BF16")
                or tensor.shape[-1] % encoding.group_size
                or tensor.shape[-1] % (32 // encoding.bits)
            ):
                raise ValueError(f"projection cannot use the requested affine encoding: {name}")
            if any(name[:-7] + suffix in tensors for suffix in (".scales", ".biases")):
                raise ValueError("floating projection partition already has encoding metadata")
            rows, width = math.prod(tensor.shape[:-1]), tensor.shape[-1]
            final_bytes += rows * width * encoding.bits // 8
            final_bytes += 2 * rows * (width // encoding.group_size) * DTYPE_BYTES[tensor.dtype]
        reservation = self.budget.reserve(self.owner, final_bytes)
        arrays = {}
        reader = ResidentMaterializer(self.budget, self.reader, owner=f"{self.owner}.source")
        try:
            for name, tensor in tensors.items():
                source = reader.materialize({name: tensor})
                try:
                    encoding = self.encodings.get(name)
                    if encoding is None:
                        arrays[name] = source.arrays[name]
                    else:
                        weight, scales, biases = mx.quantize(
                            source.arrays[name],
                            bits=encoding.bits,
                            group_size=encoding.group_size,
                            mode="affine",
                        )
                        mx.eval(weight, scales, biases)
                        arrays[name] = weight
                        arrays[name[:-7] + ".scales"] = scales
                        arrays[name[:-7] + ".biases"] = biases
                finally:
                    source.close()
        except BaseException:
            arrays.clear()
            reservation.close()
            raise
        return ResidentTensors(arrays, reservation)
