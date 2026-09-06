from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AdapterConfiguration:
    model: Path
    served_model: str
    base_path: Path
    context_capacity: int
    max_concurrent_requests: int


def validate_configuration(config: AdapterConfiguration) -> None:
    if not config.model.is_dir():
        raise ValueError(f"model snapshot is not a directory: {config.model}")
    if not (config.model / "config.json").is_file():
        raise ValueError(f"model snapshot has no config.json: {config.model}")
    if (
        not config.served_model
        or config.served_model in {".", ".."}
        or "/" in config.served_model
        or "\\" in config.served_model
    ):
        raise ValueError("served model must be a non-empty discovery-safe name")
    if config.context_capacity <= 0 or config.max_concurrent_requests <= 0:
        raise ValueError("context and concurrency must be positive")


def write_configuration(config: AdapterConfiguration) -> Path:
    validate_configuration(config)
    config.base_path.mkdir(parents=True, exist_ok=True)
    models = config.base_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    link = models / config.served_model
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(config.model.resolve(), target_is_directory=True)

    settings = {
        "version": "1.0",
        "server": {
            "host": "127.0.0.1",
            "log_level": "info",
            "burst_decode_mode": "off",
            "preserve_mid_system_cache": False,
            "distributed_inference_enabled": False,
        },
        "model": {
            "model_dirs": [str(models.resolve())],
            "model_fallback": False,
            "hide_helper_models": True,
        },
        "memory": {"prefill_memory_guard": False},
        "scheduler": {
            "max_concurrent_requests": config.max_concurrent_requests,
            "chunked_prefill": False,
            "decode_fairness": True,
        },
        "cache": {"enabled": False, "hot_cache_only": False},
        "huggingface": {"hf_cache_enabled": False},
        "sampling": {"max_context_window": config.context_capacity},
    }
    (config.base_path / "settings.json").write_text(
        json.dumps(settings, indent=2, sort_keys=True) + "\n"
    )

    model_settings = {
        "version": 1,
        "models": {
            config.served_model: {
                "model_alias": config.served_model,
                "max_context_window": config.context_capacity,
                "is_pinned": True,
                "is_default": True,
                "specprefill_enabled": False,
                "turboquant_kv_enabled": False,
                "qwen35_ane_prefill_enabled": False,
                "vlm_mtp_enabled": False,
                "mtp_enabled": False,
                "dflash_enabled": False,
                "dflash_max_ctx": None,
                "dflash_in_memory_cache": False,
                "dflash_ssd_cache": False,
                "dflash_draft_window_size": None,
                "dflash_draft_sink_size": 0,
                "dflash_verify_mode": None,
            }
        },
    }
    (config.base_path / "model_settings.json").write_text(
        json.dumps(model_settings, indent=2, sort_keys=True) + "\n"
    )
    (config.base_path / "benchmark-adapter.json").write_text(
        json.dumps(
            {
                "served_model": config.served_model,
                "model": str(config.model.resolve()),
                "context_capacity": config.context_capacity,
                "max_concurrent_requests": config.max_concurrent_requests,
                "speculative_backend": "none",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    os.environ["OMLX_BASE_PATH"] = str(config.base_path.resolve())
    return link
