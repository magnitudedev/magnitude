"""Interpret the MLX Qwen image tower without executing its model library."""

from math import isqrt

from pydantic import BaseModel, ConfigDict, PositiveInt

from engine.data import TokenId
from engine.models.qwen35.preparation import ImageGeometry
from engine.models.qwen35.vision_description import (
    AffineWeights,
    VisionBlockWeights,
    VisionDescription,
    VisionGeometry,
)
from engine.weights.formats.mlx_safetensors import MLXFormat


class VisionConfig(BaseModel):
    model_config = ConfigDict(strict=True)
    in_channels: PositiveInt
    temporal_patch_size: PositiveInt
    patch_size: PositiveInt
    spatial_merge_size: PositiveInt
    hidden_size: PositiveInt
    intermediate_size: PositiveInt
    out_hidden_size: PositiveInt
    num_heads: PositiveInt
    depth: PositiveInt
    num_position_embeddings: PositiveInt
    hidden_act: str
    deepstack_visual_indexes: list[int]


class TextWidth(BaseModel):
    model_config = ConfigDict(strict=True)
    hidden_size: PositiveInt


class ImageConfig(BaseModel):
    model_config = ConfigDict(strict=True)
    image_token_id: TokenId
    vision_start_token_id: TokenId
    vision_end_token_id: TokenId
    vision_config: VisionConfig
    text_config: TextWidth


def describe(artifact: MLXFormat) -> VisionDescription:
    config = ImageConfig.model_validate(artifact.config)
    vision = config.vision_config
    if vision.deepstack_visual_indexes or vision.hidden_act != "gelu_pytorch_tanh":
        raise ValueError("unsupported Qwen image tower")
    side = isqrt(vision.num_position_embeddings)
    if side**2 != vision.num_position_embeddings:
        raise ValueError("vision position table must be square")
    image = ImageGeometry(
        channels=vision.in_channels,
        temporal_patch=vision.temporal_patch_size,
        patch=vision.patch_size,
        merge=vision.spatial_merge_size,
        image_token=config.image_token_id,
        start_token=config.vision_start_token_id,
        end_token=config.vision_end_token_id,
    )
    g = VisionGeometry(
        image=image,
        hidden=vision.hidden_size,
        intermediate=vision.intermediate_size,
        output=vision.out_hidden_size,
        heads=vision.num_heads,
        depth=vision.depth,
        table_side=side,
    )
    if g.output != config.text_config.hidden_size:
        raise ValueError("image projection differs from decoder width")

    def weight(name, shape):
        return artifact.descriptor("vision_tower." + name, shape)

    def affine(name, output, inputs=None):
        return AffineWeights(
            weight=weight(name + ".weight", (output,) if inputs is None else (output, inputs)),
            bias=weight(name + ".bias", (output,)),
        )

    patch_name = "vision_tower.patch_embed.proj.weight"
    if artifact.tensors[patch_name].spec.shape != (
        g.hidden,
        image.temporal_patch,
        image.patch,
        image.patch,
        image.channels,
    ):
        raise ValueError("MLX vision patch kernel requires output/time/height/width/channel order")
    merged = g.hidden * image.merge**2
    return VisionDescription(
        geometry=g,
        patch=affine("patch_embed.proj", g.hidden, image.patch_width),
        positions=weight("pos_embed.weight", (side**2, g.hidden)),
        blocks=tuple(
            VisionBlockWeights(
                norm1=affine(f"blocks.{i}.norm1", g.hidden),
                qkv=affine(f"blocks.{i}.attn.qkv", 3 * g.hidden, g.hidden),
                projection=affine(f"blocks.{i}.attn.proj", g.hidden, g.hidden),
                norm2=affine(f"blocks.{i}.norm2", g.hidden),
                up=affine(f"blocks.{i}.mlp.linear_fc1", g.intermediate, g.hidden),
                down=affine(f"blocks.{i}.mlp.linear_fc2", g.hidden, g.intermediate),
            )
            for i in range(g.depth)
        ),
        merger_norm=affine("merger.norm", g.hidden),
        merger_up=affine("merger.linear_fc1", merged, merged),
        merger_down=affine("merger.linear_fc2", g.output, merged),
    )
