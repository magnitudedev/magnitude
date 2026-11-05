"""What Nemotron 3.5 headers state differently from the first Nemotron-H converters."""
from types import SimpleNamespace

import pytest

from magnitude_references.families.nemotron_h import NemotronHReference
from magnitude_references.gguf import Package

KEYS = {
    "block_count": 3, "embedding_length": 8, "attention.head_count": 2, "attention.key_length": 4,
    "attention.head_count_kv": [0, 2, 0], "feed_forward_length": [0, 0, 16], "ssm.inner_size": 8, "ssm.state_size": 4,
    "ssm.time_step_rank": 2, "ssm.group_count": 1, "ssm.conv_kernel": 3, "expert_count": 4, "expert_weights_norm": True,
    "expert_weights_scale": 1.0, "expert_feed_forward_length": 16, "expert_shared_feed_forward_length": 16,
    "context_length": 128,
}


class Header:
    architecture = "nemotron_h_moe"
    key = Package.key

    def __init__(self, keys: dict):
        self.metadata = {f"{self.architecture}.{key}": SimpleNamespace(value=value) for key, value in keys.items()}


def reference(**keys) -> NemotronHReference:
    weights = SimpleNamespace(has=lambda name: name != "output.weight", shape=lambda name: (32, 8))
    return NemotronHReference(Header({**KEYS, **keys}), weights)


def test_the_epsilon_is_read_from_whichever_key_the_converter_wrote():
    assert reference(**{"attention.layer_norm_epsilon": 1e-5}, expert_used_count=2).eps == 1e-5
    assert reference(**{"attention.layer_norm_rms_epsilon": 1e-6}, expert_used_count=2).eps == 1e-6
    with pytest.raises(ValueError, match="neither"):
        reference(expert_used_count=2)
    with pytest.raises(ValueError, match="not one"):
        reference(**{"attention.layer_norm_epsilon": 1e-5, "attention.layer_norm_rms_epsilon": 1e-6}, expert_used_count=2)


def test_the_experts_each_layer_selects_may_be_stated_per_layer():
    assert reference(**{"attention.layer_norm_epsilon": 1e-5}, expert_used_count=[0, 0, 3]).experts_used == [0, 0, 3]
    assert reference(**{"attention.layer_norm_epsilon": 1e-5}, expert_used_count=2).experts_used == [2, 2, 2]
