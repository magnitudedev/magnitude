"""Pinned resident model substitutions under the same ordinary phase contracts."""

from pathlib import Path

from magnitude_engine import blueprints as bp

from .single_session import decode, prefill


def resident(repository: str, revision: str, memory_bytes: int = 28 << 30) -> bp.engine.Engine:
    cache = Path.home() / ".cache/huggingface/hub" / ("models--" + repository.replace("/", "--"))
    artifact = bp.model.artifacts.Local(path=str(cache / "snapshots" / revision))
    return bp.engine.Engine(
        generation=bp.generation.Generation(target=bp.model.auto(artifact)),
        scheduler=bp.engine.scheduling.TimeShared(max_active=1, prefill_tokens=512),
        memory=bp.engine.memory.Budgeted(limit_bytes=memory_bytes),
        prefixes=bp.engine.prefixes.Radix(
            retention=bp.engine.prefixes.LeastRecentlyUsed(max_entries=0),
        ),
    )


qwen27 = resident(
    "mlx-community/Qwen3.8-27B-4bit",
    "3e6447f082e89cc7f0bc6e5441afd38dfce760ff",
)
gemma26 = resident(
    "mlx-community/gemma-4-26B-A4B-it-qat-4bit",
    "0e3cbab38ce568cf6e23543010d08d03b731910c",
)
qwen35_q8 = resident(
    "mlx-community/Qwen3.6-35B-A3B-8bit",
    "e06a74e6236a60c8367e1a3214e83d8b61b637b0",
    memory_bytes=48 << 30,
)

qwen27_prefill = prefill(1024, composition=qwen27, model="qwen38-27b-q4")
qwen27_decode = decode(1024, 4, composition=qwen27, model="qwen38-27b-q4")
gemma26_prefill = prefill(1024, composition=gemma26, model="gemma4-26b-qat-q4")
gemma26_decode = decode(1024, 4, composition=gemma26, model="gemma4-26b-qat-q4")
qwen35_q8_prefill = prefill(1024, composition=qwen35_q8, model="qwen36-35b-q8")
qwen35_q8_decode = decode(1024, 4, composition=qwen35_q8, model="qwen36-35b-q8")
