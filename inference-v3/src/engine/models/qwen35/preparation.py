"""Qwen's CPU image preparation and exact decoder-span interpretation."""

import json
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np
from pydantic import PositiveInt

from engine.data import Record, TokenId
from engine.inputs.layout import ConditioningIdentity, InputLayout, InputPosition, InputSpan
from engine.inputs.media import PreparedMedia, PreparedTensor
from engine.models.qwen35.inputs import InputPlan


class ImageGeometry(Record):
    channels: PositiveInt
    temporal_patch: PositiveInt
    patch: PositiveInt
    merge: PositiveInt
    image_token: TokenId
    start_token: TokenId
    end_token: TokenId

    @property
    def patch_width(self):
        return self.channels * self.temporal_patch * self.patch**2


@dataclass(frozen=True)
class ImagePatches:
    identity: ConditioningIdentity
    grid: tuple[int, int, int]
    pixels: PreparedTensor


@dataclass(frozen=True)
class PreparedInput:
    plan: InputPlan
    images: tuple[ImagePatches, ...]


def spatial_controls(grid: tuple[int, int, int], merge: int, table_side: int):
    """Geometry-only rotary positions and bilinear lookup controls in patch order."""
    t, h, w = grid
    if t != 1 or merge <= 0 or table_side <= 0 or min(h, w) < merge or h % merge or w % merge:
        raise ValueError("invalid image geometry for spatial controls")
    row, col = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    def patch_order(value):
        return value.reshape(h // merge, merge, w // merge, merge).transpose(0, 2, 1, 3).reshape(-1)

    coordinates = np.stack((patch_order(row), patch_order(col)), axis=-1).astype(np.int32)
    # align_corners=True, as in the bound Qwen processor/model contract.
    y = np.linspace(0, table_side - 1, h, dtype=np.float32)
    x = np.linspace(0, table_side - 1, w, dtype=np.float32)
    y0, x0 = y.astype(np.int32), x.astype(np.int32)
    y1, x1 = np.minimum(y0 + 1, table_side - 1), np.minimum(x0 + 1, table_side - 1)
    dy, dx = y - y0, x - x0
    indices, coefficients = [], []
    for yi, yw in ((y0, 1 - dy), (y1, dy)):
        for xi, xw in ((x0, 1 - dx), (x1, dx)):
            indices.append(patch_order(yi[:, None] * table_side + xi[None, :]).astype(np.int32))
            coefficients.append(patch_order(yw[:, None] * xw[None, :]).astype(np.float32)[:, None])
    return coordinates, tuple(indices), tuple(coefficients)


def interpret(
    tokens: tuple[TokenId, ...], media: PreparedMedia, *, processor: str, geometry: ImageGeometry
) -> PreparedInput:
    """Validate all media and placeholders before constructing executable input.

    Physical token offsets and rotary coordinates deliberately advance by
    different amounts across an image. Content identity includes preparation.
    """
    if media.processor != processor:
        raise ValueError("image processor differs from the bound artifact")
    fields = {tensor.name: tensor for tensor in media.tensors}
    if set(fields) != {"pixel_values", "image_grid_thw"}:
        raise ValueError("Qwen prepared tensors differ from the image contract")
    pixels, grids = fields["pixel_values"].array(), fields["image_grid_thw"].array()
    if (
        pixels.dtype != np.dtype("<f4")
        or pixels.ndim != 2
        or pixels.shape[1] != geometry.patch_width
        or grids.dtype != np.dtype("<i8")
        or grids.ndim != 2
        or grids.shape[1] != 3
        or not 1 <= grids.shape[0] <= 16
        or not np.isfinite(pixels).all()
    ):
        raise ValueError("invalid Qwen image tensor geometry or values")
    merge = geometry.merge
    # Python integer products cannot wrap before checking the payload extent.
    sizes = grids.tolist()
    if any(t != 1 or h < merge or w < merge or h % merge or w % merge for t, h, w in sizes):
        raise ValueError("image grids require one frame and merge-aligned spatial dimensions")
    if sum(t * h * w for t, h, w in sizes) != len(pixels):
        raise ValueError("image grids must cover their pixel patches exactly")
    coordinates = []
    images, spans = [], []
    cursor = pixel_start = rotary = 0
    for t, h, w in sizes:
        patches, count = t * h * w, t * h * w // merge**2
        try:
            start = tokens.index(geometry.image_token, cursor)
        except ValueError as error:
            raise ValueError("image has no corresponding prompt span") from error
        end = start + count
        if (
            start == 0
            or tokens[start - 1] != geometry.start_token
            or tokens[start:end] != (geometry.image_token,) * count
            or end >= len(tokens)
            or tokens[end] != geometry.end_token
        ):
            raise ValueError("image prompt span differs from feature geometry")
        coordinates.extend((p, p, p) for p in range(rotary, rotary + start - cursor))
        base = rotary + start - cursor
        coordinates.extend(
            (base + frame, base + row, base + col)
            for frame in range(t)
            for row in range(h // merge)
            for col in range(w // merge)
        )
        values = PreparedTensor.from_array(
            "pixel_values", pixels[pixel_start : pixel_start + patches]
        )
        digest = sha256(processor.encode())
        digest.update(np.asarray((t, h, w), dtype="<i8").tobytes())
        digest.update(values.data)
        identity = ConditioningIdentity(digest.hexdigest())
        images.append(ImagePatches(identity, (t, h, w), values))
        spans.append(
            InputSpan(start=InputPosition(start), end=InputPosition(end), identity=identity)
        )
        rotary = base + max(t, h // merge, w // merge)
        cursor, pixel_start = end, pixel_start + patches
    if geometry.image_token in tokens[cursor:]:
        raise ValueError("image prompt span has no source pixels")
    coordinates.extend((p, p, p) for p in range(rotary, rotary + len(tokens) - cursor))
    return PreparedInput(
        InputPlan(
            tokens=tokens,
            layout=InputLayout(count=len(tokens), spans=tuple(spans)),
            coordinates=tuple(coordinates),
            continuation=rotary + len(tokens) - cursor,
        ),
        tuple(images),
    )


def preparation_identity(directory: Path) -> str:
    digest = sha256(b"qwen35-image-preparation-v1")
    for content in (
        (directory / "config.json").read_bytes(),
        (directory / "preprocessor_config.json").read_bytes(),
        version("transformers").encode(),
        version("pillow").encode(),
    ):
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.hexdigest()


class ImagePreparation:
    """Artifact-bound PIL processor; request data never becomes processor state."""

    def __init__(self, directory: Path, pieces: tuple[str, ...]):
        from transformers.models.auto.image_processing_auto import AutoImageProcessor

        directory = directory.expanduser().resolve()
        config_bytes = (directory / "config.json").read_bytes()
        config = json.loads(config_bytes)
        vision = config["vision_config"]
        self.geometry = ImageGeometry(
            channels=vision["in_channels"],
            temporal_patch=vision["temporal_patch_size"],
            patch=vision["patch_size"],
            merge=vision["spatial_merge_size"],
            image_token=config["image_token_id"],
            start_token=config["vision_start_token_id"],
            end_token=config["vision_end_token_id"],
        )
        self.identity = preparation_identity(directory)
        self.image_token = pieces[self.geometry.image_token]
        self.marker = "".join(
            pieces[token]
            for token in (
                self.geometry.start_token,
                self.geometry.image_token,
                self.geometry.end_token,
            )
        )
        self.processor = AutoImageProcessor.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False, backend="pil"
        )

    def prepare(self, text: str, images: tuple) -> tuple[str, PreparedMedia]:
        parts = text.split(self.image_token)
        if not images or len(parts) != len(images) + 1:
            raise ValueError("prompt must contain one placeholder per source image")
        data = self.processor(images=list(images), return_tensors="np")
        if set(data) != {"pixel_values", "image_grid_thw"}:
            raise ValueError("image processor output differs from the bound contract")
        grids = np.asarray(data["image_grid_thw"])
        if grids.shape != (len(images), 3):
            raise ValueError("image processor grids differ from source count")
        counts = [int(t) * int(h) * int(w) // self.geometry.merge**2 for t, h, w in grids]
        media = PreparedMedia(
            self.identity,
            tuple(
                PreparedTensor.from_array(name, np.asarray(value)) for name, value in data.items()
            ),
        )
        expanded = parts[0] + "".join(
            self.image_token * count + part for count, part in zip(counts, parts[1:], strict=True)
        )
        return expanded, media
