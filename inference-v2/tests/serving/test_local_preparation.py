"""Exact CPU preparation, realistic source sizes and concurrent bound-processor use."""

import io
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image
from transformers.models.auto.image_processing_auto import AutoImageProcessor

from magnitude_engine.artifacts.source import LocalArtifact
from magnitude_engine.artifacts.tokenizer import TokenizerArtifact
from magnitude_engine.models.architectures.gemma4.preparation import GemmaImages
from magnitude_engine.models.architectures.qwen35.preparation import QwenImages
from magnitude_engine.models.preparation import PreparedMedia
from magnitude_engine.worker.framing import Frame, read_frame, write_frame


@pytest.mark.model
@pytest.mark.parametrize("family", ["qwen35", "gemma4"])
def test_native_preparation_quality_survives_large_inputs_transport_and_concurrent_reuse(family):
    path = os.environ.get(f"MAGNITUDE_TEST_{family.upper()}_VISION")
    if not path:
        pytest.skip("requires the local vision artifact")
    artifact = LocalArtifact(path)
    tokenizer = TokenizerArtifact.load(Path(path)).tokenizer
    # Texture exercises every pixel channel/value; this is not a compressible-color transport test.
    pixels = np.random.default_rng(19).integers(0, 256, (1200, 1600, 3), dtype=np.uint8)
    photo = Image.fromarray(pixels)
    sources = [photo] if family == "qwen35" else [photo.resize((224, 112))] * 16
    token = tokenizer.convert_ids_to_tokens(artifact.configuration()["image_token_id"])
    text = ("before " + token + " after ") * len(sources)
    with patch.object(
        AutoImageProcessor, "from_pretrained", wraps=AutoImageProcessor.from_pretrained
    ) as load:
        processor = (QwenImages if family == "qwen35" else GemmaImages)(artifact)
        expanded, prepared = processor.process(text, sources, tokenizer)
        # Separate CPU calls share configuration, never mutable per-request operands.
        with ThreadPoolExecutor(2) as pool:
            again = tuple(pool.map(lambda _: processor.process(text, sources, tokenizer), range(2)))
        assert load.call_count == 1
    assert all(item == (expanded, prepared) for item in again)
    assert sum(len(b) for b in prepared.buffers) > 32 << 20
    reference = AutoImageProcessor.from_pretrained(path, backend="pil", local_files_only=True)
    expected = reference(images=sources, return_tensors="np")
    assert {t.name for t in prepared.tensors} == set(expected)
    for tensor in prepared.tensors:
        assert np.array_equal(tensor.array(), expected[tensor.name])
    stream = io.BytesIO()
    write_frame(stream, Frame("g", prepared.encode(), prepared.buffers))
    stream.seek(0)
    transported = read_frame(stream, "g")
    restored = PreparedMedia.decode(transported.message, transported.buffers)
    assert restored == prepared and restored.identity() == prepared.identity()
