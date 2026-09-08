"""CPU-only Gemma image preparation and explicit image-block expansion."""

import numpy as np

from magnitude_engine.components import component
from magnitude_engine.models.preparation import (
    ImagePreparation,
    PreparedMedia,
    PreparedTensor,
)


@component("MODEL:GEMMA4.PREPARATION:MAG:IMAGES")
class GemmaImages(ImagePreparation):
    def process(self, text, images, tokenizer):
        processor = self.processor
        parts = text.split(tokenizer.image_token)
        if len(parts) != len(images) + 1:
            raise ValueError("Gemma prompt must contain exactly one placeholder per source image")
        data = processor(images=images, return_tensors="np")
        if set(data) != {"pixel_values", "image_position_ids", "num_soft_tokens_per_image"}:
            raise ValueError("Gemma image processor output differs from its bound contract")
        counts = data["num_soft_tokens_per_image"]
        expanded = parts[0] + "".join(
            tokenizer.boi_token + tokenizer.image_token * int(count) + tokenizer.eoi_token + part
            for count, part in zip(counts, parts[1:], strict=True)
        )
        return expanded, PreparedMedia(
            self.identity,
            tuple(
                PreparedTensor.from_array(name, np.asarray(value)) for name, value in data.items()
            ),
        )
