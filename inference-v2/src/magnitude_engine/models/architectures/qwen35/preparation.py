"""CPU-only Qwen image interpretation; no language model or video processor is constructed."""

import numpy as np

from magnitude_engine.artifacts.source import LocalArtifact
from magnitude_engine.components import component
from magnitude_engine.models.preparation import (
    ImagePreparation,
    PreparedMedia,
    PreparedTensor,
)


@component("MODEL:QWEN35.PREPARATION:MAG:IMAGES")
class QwenImages(ImagePreparation):
    def __init__(self, artifact: LocalArtifact):
        super().__init__(artifact)
        self.image_token_id = artifact.configuration()["image_token_id"]

    def process(self, text, images, tokenizer):
        processor = self.processor
        token = tokenizer.convert_ids_to_tokens(self.image_token_id)
        parts = text.split(token)
        if len(parts) != len(images) + 1:
            raise ValueError("Qwen prompt must contain exactly one placeholder per source image")
        data = processor(images=images, return_tensors="np")
        if set(data) != {"pixel_values", "image_grid_thw"}:
            raise ValueError("Qwen image processor output differs from its bound contract")
        counts = np.prod(data["image_grid_thw"], axis=1) // processor.merge_size**2
        expanded = parts[0] + "".join(
            token * int(count) + part for count, part in zip(counts, parts[1:], strict=True)
        )
        return expanded, PreparedMedia(
            self.identity,
            tuple(
                PreparedTensor.from_array(name, np.asarray(value)) for name, value in data.items()
            ),
        )
