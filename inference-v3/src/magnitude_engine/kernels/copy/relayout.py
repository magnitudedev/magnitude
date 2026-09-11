"""Residency-time permutation from a format codec to canonical quantized bytes."""

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.launch import serial
from magnitude_engine.weights.representation import (
    Affine,
    Codebook,
    DirectCoefficients,
    HierarchicalCoefficients,
    canonical_layout,
)


def relayout(
    elements: int,
    representation: Affine | Codebook,
    codec,
    staged_tiles: int,
    *,
    capability: Capability,
):
    """Permute complete source tiles into their final private allocation.

    ``Range`` contains the number of valid staged tiles and the first logical
    destination tile. The source allocation is fixed-size so all chunks of one
    encoding share a specialization; the final chunk is zero padded.
    """
    layout = canonical_layout(representation, elements)
    coefficients = representation.coefficients
    tile_elements = codec.block_elements
    if tile_elements <= 0 or codec.block_bytes <= 0 or staged_tiles <= 0:
        raise ValueError("invalid quantized import geometry")
    if isinstance(coefficients, HierarchicalCoefficients):
        if tile_elements != coefficients.supergroup:
            raise ValueError("source and canonical hierarchy tiles differ")
    elif tile_elements != representation.group:
        raise ValueError("direct affine import is grouped by its coefficient extent")

    low_bits = (
        representation.code.low_bits
        if isinstance(representation, Affine)
        else representation.code_bits
    )
    high_bits = representation.code.high_bits if isinstance(representation, Affine) else 0
    low_bytes = tile_elements * low_bits // 8
    high_bytes = tile_elements * high_bits // 8
    groups = tile_elements // representation.group
    cpu = serial(capability)

    @T.macro
    def write(A, B, source_tile, target_tile):
        source_base = source_tile * codec.block_bytes
        target_base = target_tile * layout.tile_bytes if layout.hierarchical else 0
        low_base = target_base + layout.low
        if not layout.hierarchical:
            low_base += target_tile * low_bytes
        for byte_index in T.serial(low_bytes):
            packed = T.alloc_local((1,), "uint32")
            packed[0] = 0
            for slot in T.unroll(8 // low_bits, explicit=True):
                index = byte_index * (8 // low_bits) + slot
                packed[0] |= (codec.code(A, source_base, index) & ((1 << low_bits) - 1)) << (
                    slot * low_bits
                )
            B[low_base + byte_index] = packed[0].astype("uint8")

        if high_bits:
            high_base = target_base + layout.high
            if not layout.hierarchical:
                high_base += target_tile * high_bytes
            for byte_index in T.serial(high_bytes):
                packed = T.alloc_local((1,), "uint32")
                packed[0] = 0
                for slot in T.unroll(8 // high_bits, explicit=True):
                    index = byte_index * (8 // high_bits) + slot
                    packed[0] |= (
                        (codec.code(A, source_base, index) >> low_bits) & ((1 << high_bits) - 1)
                    ) << (slot * high_bits)
                B[high_base + byte_index] = packed[0].astype("uint8")

        if isinstance(coefficients, DirectCoefficients):
            scale_base = layout.scales + target_tile * coefficients.scale_dtype.itemsize
            for byte_index in T.unroll(coefficients.scale_dtype.itemsize, explicit=True):
                B[scale_base + byte_index] = codec.scale_byte(A, source_base, byte_index)
            if coefficients.bias_dtype is not None:
                bias_base = layout.biases + target_tile * coefficients.bias_dtype.itemsize
                for byte_index in T.unroll(coefficients.bias_dtype.itemsize, explicit=True):
                    B[bias_base + byte_index] = codec.bias_byte(A, source_base, byte_index)
        else:
            scale_bytes = groups * coefficients.local_scale_bits // 8
            for byte_index in T.serial(scale_bytes):
                B[target_base + layout.scales + byte_index] = 0
            for group in T.unroll(groups, explicit=True):
                value = codec.local_scale(A, source_base, group)
                bit = group * coefficients.local_scale_bits
                byte = bit // 8
                shift = bit % 8
                B[target_base + layout.scales + byte] |= ((value << shift) & 255).astype("uint8")
                if shift + coefficients.local_scale_bits > 8:
                    B[target_base + layout.scales + byte + 1] |= (value >> (8 - shift)).astype(
                        "uint8"
                    )

            if coefficients.local_bias_bits is not None:
                bias_bytes = groups * coefficients.local_bias_bits // 8
                for byte_index in T.serial(bias_bytes):
                    B[target_base + layout.biases + byte_index] = 0
                for group in T.unroll(groups, explicit=True):
                    value = codec.local_bias(A, source_base, group)
                    bit = group * coefficients.local_bias_bits
                    byte = bit // 8
                    shift = bit % 8
                    B[target_base + layout.biases + byte] |= ((value << shift) & 255).astype(
                        "uint8"
                    )
                    if shift + coefficients.local_bias_bits > 8:
                        B[target_base + layout.biases + byte + 1] |= (value >> (8 - shift)).astype(
                            "uint8"
                        )

            for byte_index in T.unroll(coefficients.super_scale_dtype.itemsize, explicit=True):
                B[target_base + layout.super_scale + byte_index] = codec.scale_byte(
                    A, source_base, byte_index
                )
            if coefficients.super_bias_dtype is not None:
                for byte_index in T.unroll(coefficients.super_bias_dtype.itemsize, explicit=True):
                    B[target_base + layout.super_bias + byte_index] = codec.bias_byte(
                        A, source_base, byte_index
                    )

    @T.prim_func
    def main(
        A: T.Tensor((staged_tiles * codec.block_bytes,), "uint8"),
        B: T.Tensor((layout.nbytes,), "uint8"),
        Range: T.Tensor((2,), "int32"),
    ):
        if cpu:
            for tile in T.Parallel(staged_tiles):
                if tile < Range[0]:
                    write(A, B, tile, Range[1] + tile)
        else:
            with T.Kernel(T.ceildiv(staged_tiles, 128), threads=128) as block:
                tile = block * 128 + T.get_thread_binding()
                if tile < Range[0]:
                    write(A, B, tile, Range[1] + tile)

    return main
