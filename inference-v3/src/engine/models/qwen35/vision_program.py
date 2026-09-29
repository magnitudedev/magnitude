"""Compile the image encoder from its semantic equations and weight roles."""

from collections.abc import Mapping

import ops
from engine.models.qwen35.tensor_program import ProgramDefinition
from engine.models.qwen35.vision import (
    Affine,
    Normalization,
    VisionBlock,
    merge_patches,
    vision_block,
)
from engine.models.qwen35.vision_description import VisionDescription


def weight_roles(description: VisionDescription):
    yield description.positions
    pairs = [description.patch]
    for block in description.blocks:
        pairs.extend((block.norm1, block.qkv, block.projection, block.norm2, block.up, block.down))
    pairs.extend((description.merger_norm, description.merger_up, description.merger_down))
    for pair in pairs:
        yield pair.weight
        yield pair.bias


def define(
    description: VisionDescription, weight_specs: Mapping[str, ops.TensorSpec], rows: int
) -> ProgramDefinition:
    g = description.geometry
    image = g.image
    if rows <= 0 or rows % image.merge**2:
        raise ValueError("image encoding requires complete spatial merge groups")
    arguments = [
        ops.Argument(ops.TensorSpec((rows, image.patch_width), ops.DType.F32), "pixels"),
        ops.Argument(ops.TensorSpec((rows, 2), ops.DType.I32), "coordinates"),
        ops.Argument(ops.TensorSpec((rows, 2), ops.DType.I32), "visible"),
    ]
    arguments.extend(
        ops.Argument(ops.TensorSpec((rows,), ops.DType.I32), f"position_indices.{i}")
        for i in range(4)
    )
    arguments.extend(
        ops.Argument(ops.TensorSpec((rows, 1), ops.DType.F32), f"position_weights.{i}")
        for i in range(4)
    )
    arguments.extend(
        ops.Argument(ops.TensorSpec((1,), ops.DType.I32), f"qkv_selector.{i}") for i in range(3)
    )
    constants = {
        role.name: ops.Argument(weight_specs[role.name], role.name, ops.ValueKind.CONSTANT)
        for role in weight_roles(description)
    }

    @ops.formula(id="qwen35.vision", version=1, metric="tokens", rows="pixels")
    def function(pixels, coordinates, visible, *controls, **bound):
        def affine(role):
            return Affine(bound[role.weight.name], bound[role.bias.name])

        def norm(role):
            return Normalization(bound[role.weight.name], bound[role.bias.name])

        indices, coefficients, selectors = controls[:4], controls[4:8], controls[8:]
        # Processor patches are channel/time/height/width. The artifact's
        # convolution kernels are time/height/width/channel within each output.
        patches = ops.reshape(
            pixels, (rows, image.channels, image.temporal_patch, image.patch, image.patch)
        )
        patches = ops.reshape(ops.transpose(patches, (0, 2, 3, 4, 1)), pixels.shape)
        hidden = affine(description.patch)(ops.cast(patches, ops.DType.BF16))
        table = bound[description.positions.name]
        positions = tuple(
            ops.embedding(index, table) * ops.cast(coefficient, table.dtype)
            for index, coefficient in zip(indices, coefficients, strict=True)
        )
        hidden = hidden + (((positions[0] + positions[1]) + positions[2]) + positions[3])
        for block in description.blocks:
            hidden = vision_block(
                hidden,
                coordinates,
                selectors,
                visible,
                VisionBlock(
                    norm(block.norm1),
                    affine(block.qkv),
                    affine(block.projection),
                    norm(block.norm2),
                    affine(block.up),
                    affine(block.down),
                    g.heads,
                ),
            )
        features = merge_patches(
            hidden,
            norm(description.merger_norm),
            affine(description.merger_up),
            affine(description.merger_down),
            image.merge,
        )
        return ops.cast(features, ops.DType.F32)

    return ProgramDefinition(
        function,
        ops.Signature(tuple(arguments), constants),
        ops.CompileOptions(mode="prefill"),
        dict(weight_specs),
    )
