"""Build the production parameter container while keeping an independent LM oracle."""

from dataclasses import asdict

import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_vlm.models import qwen3_5, qwen3_5_moe


def vision_language_parameters(reference):
    architecture = qwen3_5_moe if reference.args.num_experts else qwen3_5
    model = architecture.LanguageModel(architecture.TextConfig.from_dict(asdict(reference.args)))
    quantized = {
        path: module for path, module in reference.named_modules() if hasattr(module, "bits")
    }
    nn.quantize(
        model,
        class_predicate=lambda path, _: (
            {"bits": quantized[path].bits, "group_size": quantized[path].group_size}
            if path in quantized
            else False
        ),
    )
    model.load_weights(tree_flatten(reference.parameters()))
    model.eval()
    return model
