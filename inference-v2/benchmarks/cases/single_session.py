"""Separate fixed-query prefill and ordinary decode for the declared model composition."""

from benchmarks.contracts import Experiment
from benchmarks.subjects import ModelPrefill, PlainDecode
from magnitude_engine import blueprints as bp

from .qwen36 import artifact


def engine():
    return bp.engine.Engine(
        generation=bp.generation.Generation(target=bp.model.auto(artifact)),
        scheduler=bp.engine.scheduling.TimeShared(max_active=1, prefill_tokens=512),
        prefixes=bp.engine.prefixes.Radix(
            retention=bp.engine.prefixes.LeastRecentlyUsed(max_entries=0),
        ),
    )


def prefill(
    context: int, *, composition: bp.engine.Engine | None = None, model: str = "qwen36",
) -> Experiment:
    return Experiment(
        identity=f"model.native-{model}-prefill-at-{context}",
        subject=ModelPrefill(
            engine=composition or engine(), prefix_tokens=context - 512, input_tokens=512,
        ),
        characteristic="MECHANISM-CAUSAL-PREFILL",
        measurement_width="integrated",
        claim=(
            "Completed 512-token causal prefill ending at the declared context, with the declared "
            "resident model execution. Prefix preparation, checkpoint restore and continuation "
            "validation are outside timing. This is a mechanism diagnostic. "
            "task accuracy and hardware ceilings need separate qualification."
        ),
        warmup=1,
        repetitions=3,
        run_class="diagnostic",
        invariants=(
            "fixed input tokens and prefix",
            "committed state length checked",
            "finite repeatable continuation logits",
            "full model-state completion",
        ),
    )


def decode(
    context: int, allowance: int = 1, *,
    composition: bp.engine.Engine | None = None, model: str = "qwen36",
) -> Experiment:
    return Experiment(
        identity=f"generation.native-{model}-decode-at-{context}-span-{allowance}",
        subject=PlainDecode(
            engine=composition or engine(), prompt_tokens=context,
            output_tokens=64, token_allowance=allowance,
        ),
        characteristic="MECHANISM-ORDINARY-DECODE",
        measurement_width="integrated",
        claim=(
            f"64 ordinary decode tokens with a service allowance of {allowance}, "
            "including model/state, "
            "sampling and publication to the generation caller. The first step consumes the final "
            "prompt token. Bulk prefill, restore, scheduling, HTTP and semantic validation "
            "are outside timing. No drafter or grammar is used. "
            "Task accuracy and hardware ceilings need separate qualification."
        ),
        warmup=1,
        repetitions=3,
        run_class="diagnostic",
        invariants=(
            "same fixed context checkpoint",
            "no prefill inside timing",
            "64 repeatable greedy tokens",
            "one causal forward per output token",
            "no speculative or forced tokens",
            "completed model work",
        ),
    )


native_prefill_1k = prefill(1024)
native_prefill_4k = prefill(4096)
native_prefill_16k = prefill(16384)
native_decode_1k = decode(1024)
native_decode_4k = decode(4096)
native_decode_16k = decode(16384)

native_causal_1k = decode(1024, 4)
native_causal_4k = decode(4096, 4)
native_causal_16k = decode(16384, 4)
