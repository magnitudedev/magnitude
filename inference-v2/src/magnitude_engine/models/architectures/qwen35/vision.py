"""Qwen image encoding, input geometry, and semantic continuation ownership."""

from __future__ import annotations

import hashlib
from contextlib import ExitStack
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx_vlm.models.qwen3_5.vision import VisionModel

from magnitude_engine.artifacts.blueprint import Local
from magnitude_engine.artifacts.identity import processor_identity
from magnitude_engine.artifacts.source import LocalArtifact
from magnitude_engine.components import component, component_id
from magnitude_engine.models.computation import Computation
from magnitude_engine.models.context import InputCheckpoint
from magnitude_engine.models.embeddings.replacement import EmbeddingReplacement
from magnitude_engine.models.features import FeatureLease, FeatureSet
from magnitude_engine.models.inputs import ModelInputs
from magnitude_engine.models.loading.parameters import affine_encodings, load_resident_parameters
from magnitude_engine.models.preparation import PreparedMedia
from magnitude_engine.models.prompt import InputSpan, Prompt
from magnitude_engine.models.residency import ModelResources

from .blueprint import Images
from .inputs import QwenInputs
from .preparation import QwenImages


@dataclass(frozen=True)
class ImageInput:
    span: InputSpan
    grid: tuple[int, int, int]
    pixels: np.ndarray


@component("MODEL:QWEN35.VISION:VLM:STANDARD")
class QwenVision:
    def __init__(
        self, artifact: LocalArtifact, configuration, tensors, reader, resources: ModelResources
    ):
        self.owner, self.budget = resources.owner, resources.budget
        self.cache = resources.input_features
        self.config = configuration
        self.processor = Images(artifact=Local(path=artifact.path))
        self.identity = processor_identity(artifact.directory, component_id(QwenImages))
        self.model = VisionModel(configuration.vision_config)
        self.model.eval()
        vision = {name.removeprefix("vision_tower."): t for name, t in tensors.items()}
        settings = artifact.configuration().get("quantization", {})
        self.parameters = load_resident_parameters(
            self.model,
            vision,
            affine_encodings(vision, settings, prefix="vision_tower."),
            budget=self.budget,
            reader=reader,
            owner="qwen35.vision.weights",
        )
        self.dtype = self.model.patch_embed.proj.weight.dtype
        self.closed = False

    def close(self):
        if not self.closed:
            self.closed = True
            self.model = None
            self.parameters.close()

    def prepare(self, tokens: tuple[int, ...], media: PreparedMedia):
        if self.closed or media.processor != self.identity:
            raise ValueError("Qwen media processor differs from the bound artifact")
        tensors = {t.name: t for t in media.tensors}
        if set(tensors) != {"pixel_values", "image_grid_thw"}:
            raise ValueError("Qwen image tensors differ from the model input contract")
        pixels, grids = tensors["pixel_values"].array(), tensors["image_grid_thw"].array()
        config = self.config.vision_config
        pixel_width = config.in_channels * config.temporal_patch_size * config.patch_size**2
        if (
            pixels.dtype != np.float32
            or pixels.ndim != 2
            or pixels.shape[1] != pixel_width
            or (
                grids.dtype != np.int64
                or grids.ndim != 2
                or grids.shape[1] != 3
                or not 1 <= grids.shape[0] <= 16
                or not np.isfinite(pixels).all()
            )
        ):
            raise ValueError("invalid Qwen image pixel or grid geometry")
        merge = config.spatial_merge_size
        if any(
            t != 1 or h < merge or w < merge or h % merge or w % merge for t, h, w in grids.tolist()
        ):
            raise ValueError("Qwen images require positive spatial grids aligned to merging")
        if int(np.prod(grids, axis=1).sum()) != len(pixels):
            raise ValueError("Qwen image grids do not cover their pixel patches exactly")
        if config.out_hidden_size != self.config.text_config.hidden_size:
            raise ValueError("Qwen vision output width differs from its decoder")
        coordinates = np.empty((3, len(tokens)), dtype=np.int32)
        images = []
        cursor = pixel_start = next_position = 0
        for t, h, w in grids.tolist():
            patches = t * h * w
            count = patches // merge**2
            try:
                start = tokens.index(self.config.image_token_id, cursor)
            except ValueError as error:
                raise ValueError("Qwen image has no matching prompt span") from error
            end = start + count
            if (
                start == 0
                or tokens[start - 1] != self.config.vision_start_token_id
                or (
                    tokens[start:end] != (self.config.image_token_id,) * count
                    or end >= len(tokens)
                    or tokens[end] != self.config.vision_end_token_id
                )
            ):
                raise ValueError("Qwen image placeholder span differs from its feature geometry")
            values = pixels[pixel_start : pixel_start + patches]
            identity = hashlib.sha256(
                self.identity.encode()
                + np.array((t, h, w), dtype=np.int64).tobytes()
                + values.tobytes()
            ).digest()
            span = InputSpan(start, end, identity)
            images.append(ImageInput(span, (t, h, w), values))
            text_count = start - cursor
            coordinates[:, cursor:start] = next_position + np.arange(text_count)[None]
            base = next_position + text_count
            coordinates[:, start:end] = (
                np.indices((t, h // merge, w // merge)).reshape(3, -1) + base
            )
            next_position = base + max(t, h // merge, w // merge)
            cursor, pixel_start = end, pixel_start + patches
        if self.config.image_token_id in tokens[cursor:]:
            raise ValueError("Qwen prompt contains an image span without source pixels")
        coordinates[:, cursor:] = next_position + np.arange(len(tokens) - cursor)[None]
        prompt = Prompt(tokens, tuple(image.span for image in images))
        return prompt, QwenSource(self, prompt, coordinates, tuple(images))


@dataclass
class EncodeImage:
    encoder: QwenVision
    image: ImageInput
    output: FeatureLease

    @property
    def owner(self):
        return self.encoder.owner

    @property
    def batch_key(self):
        return id(self.encoder)

    def reserve(self, rows: tuple[Computation, ...]):
        assert all(isinstance(row, EncodeImage) for row in rows)
        typed = tuple(row for row in rows if isinstance(row, EncodeImage))
        config = self.encoder.config.vision_config
        patches = sum(len(row.image.pixels) for row in typed)
        scratch = 4 * (
            patches * (12 * config.hidden_size + 2 * config.intermediate_size)
            + config.num_heads * sum(len(row.image.pixels) ** 2 for row in typed)
        )
        return self.encoder.budget.reserve("qwen35.vision.scratch", scratch)

    def run_batch(self, rows: tuple[Computation, ...], scope):
        if self.encoder.closed:
            raise RuntimeError("Qwen encoder is closed")
        assert self.encoder.model is not None
        typed = tuple(row for row in rows if isinstance(row, EncodeImage))
        if len(typed) != len(rows):
            raise ValueError("Qwen encoder batch contains unrelated computations")
        for row in typed:
            scope.acquire(row.output.fork)
        pixels = mx.concatenate(
            [mx.array(row.image.pixels).astype(self.encoder.dtype) for row in typed]
        )
        grids = mx.array([row.image.grid for row in typed], dtype=mx.int32)
        features, deepstack = self.encoder.model(pixels, grids)
        if deepstack:
            raise ValueError("Qwen3.5 binding does not provide deepstack decoder injection")
        outputs = []
        start = 0
        for row in typed:
            count = row.image.span.end - row.image.span.start
            value = mx.array(features[start : start + count])[None]
            if value.nbytes > row.output.feature.reservation.size:
                raise MemoryError("Qwen feature exceeded its declared allocation")
            row.output.feature.value = value
            outputs.append((value,))
            start += count
        if start != features.shape[0]:
            raise ValueError("Qwen encoder output does not align with image spans")
        return tuple(outputs)


@dataclass(frozen=True)
class QwenSource:
    encoder: QwenVision
    prompt: Prompt
    coordinates: np.ndarray
    images: tuple[ImageInput, ...]

    def delta(self, position):
        count = min(position, len(self.prompt.tokens))
        return 0 if count == 0 else int(self.coordinates[:, :count].max()) + 1 - count

    def bind(self, checkpoint: InputCheckpoint | None):
        if checkpoint is not None and (
            not isinstance(checkpoint, QwenCheckpoint)
            or checkpoint.closed
            or checkpoint.delta != self.delta(checkpoint.position)
            or checkpoint.identities != self.identities(checkpoint.position)
        ):
            raise ValueError("Qwen input continuation differs from the prepared prompt")
        saved = {} if checkpoint is None else checkpoint.features
        return QwenContext(self, {key: lease.fork() for key, lease in saved.items()})

    def identities(self, position: int) -> tuple[bytes, ...]:
        return tuple(image.span.identity for image in self.images if image.span.start < position)


class QwenContinuation:
    cache_hits = 0

    def __init__(self, delta: int, identities: tuple[bytes, ...] = ()):
        self.delta = delta
        self.identities = identities

    def boundary(self, position):
        return type(position) is int and position >= 0

    def acquire(self, position, count):
        return ExitStack()

    def prepare(self, position, count):
        yield from ()

    def assemble(self, inputs, position):
        return ModelInputs(
            inputs.tokens,
            inputs.conditioning,
            QwenInputs(mx.array([position + self.delta], mx.int32)),
        )

    def batch_key(self, position, count):
        return None

    def checkpoint(self, position):
        return QwenCheckpoint(position, self.delta, {}, self.identities)

    def close(self):
        pass


class QwenContext(QwenContinuation):
    def __init__(self, source: QwenSource, features: dict[bytes, FeatureLease]):
        super().__init__(
            source.delta(len(source.prompt.tokens)), source.identities(len(source.prompt.tokens))
        )
        self.source = source
        self.features = FeatureSet(
            source.encoder, source.encoder.budget, source.encoder.cache, features
        )

    @property
    def cache_hits(self):
        return self.features.cache_hits

    def prepare(self, position, count):
        self.features.keep(
            {image.span.identity for image in self.source.images if image.span.end > position}
        )
        for image in self.source.images:
            if image.span.start >= position + count or image.span.end <= position:
                continue
            size = (
                (image.span.end - image.span.start)
                * self.source.encoder.config.vision_config.out_hidden_size
                * 4
            )
            yield from self.features.prepare(
                image.span.identity,
                size,
                lambda lease, image=image: EncodeImage(self.source.encoder, image, lease),
            )

    def acquire(self, position, count):
        return self.features.pin(
            image.span.identity
            for image in self.source.images
            if image.span.start < position + count and image.span.end > position
        )

    def assemble(self, inputs, position):
        length = len(self.source.prompt.tokens)
        if position >= length:
            return super().assemble(inputs, position)
        end = position + inputs.count
        coordinates = self.source.coordinates[:, position : min(end, length)]
        if end > length:
            tail = np.broadcast_to(np.arange(length, end) + self.delta, (3, end - length))
            coordinates = np.concatenate([coordinates, tail], axis=1)
        embeddings = []
        for image in self.source.images:
            start, stop = max(position, image.span.start), min(end, image.span.end)
            if start < stop:
                value = self.features.value(image.span.identity)
                embeddings.append(
                    EmbeddingReplacement(
                        start - position,
                        value[:, start - image.span.start : stop - image.span.start],
                    )
                )
        return ModelInputs(
            inputs.tokens,
            inputs.conditioning,
            QwenInputs(mx.array(coordinates[:, None], mx.int32), tuple(embeddings)),
        )

    def checkpoint(self, position):
        partial = {
            image.span.identity
            for image in self.source.images
            if image.span.start < position < image.span.end
        }
        return QwenCheckpoint(
            position,
            self.source.delta(position),
            self.features.checkpoint(partial),
            self.source.identities(position),
        )

    def close(self):
        self.features.close()


class QwenCheckpoint:
    reclaimable = True

    def __init__(
        self,
        position: int,
        delta: int,
        features: dict[bytes, FeatureLease],
        identities: tuple[bytes, ...] = (),
    ):
        self.position, self.delta, self.features = position, delta, features
        self.identities = identities
        self.closed = False

    def retained_storage(self):
        return tuple(
            storage for feature in self.features.values() for storage in feature.retained_storage()
        )

    def restore(self):
        if self.closed:
            raise ValueError("Qwen input checkpoint is closed")
        if self.features:
            raise ValueError("partial Qwen images require their prepared input source to resume")
        return QwenContinuation(self.delta, self.identities)

    def close(self):
        if not self.closed:
            for feature in self.features.values():
                feature.close()
            self.features.clear()
            self.closed = True
