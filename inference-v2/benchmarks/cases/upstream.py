"""Stock-loaded neural controls matching the single_session trace inputs."""

from benchmarks.contracts import Experiment
from benchmarks.subjects.upstream.blueprint import UpstreamDecode, UpstreamPrefill

from .qwen36 import artifact


def decode(context: int, service_tokens: int) -> Experiment:
    return Experiment(
        identity=f"upstream.qwen36-decode-at-{context}-service-{service_tokens}",
        subject=UpstreamDecode(
            artifact=artifact, prompt_tokens=context, output_tokens=64,
            service_tokens=service_tokens,
        ),
        characteristic="MECHANISM-ORDINARY-DECODE",
        measurement_width="component",
        claim=(
            "Stock-loaded MLX-VLM text forward with greedy feedback, 64 inputs and outputs. "
            "Loading and prefix preparation are excluded. State is completed at each declared "
            "service boundary. This isolates neural execution, not the upstream server."
        ),
        comparison="compare output token IDs and completed service times with single_session",
        warmup=1,
        repetitions=3,
        run_class="diagnostic",
        invariants=(
            "64 repeatable greedy outputs", "no speculative lookahead beyond output budget",
        ),
    )


def prefill(context: int) -> Experiment:
    return Experiment(
        identity=f"upstream.qwen36-prefill-at-{context}",
        subject=UpstreamPrefill(
            artifact=artifact, prefix_tokens=context - 512, input_tokens=512,
        ),
        characteristic="MECHANISM-CAUSAL-PREFILL",
        measurement_width="component",
        claim=(
            "Stock-loaded MLX-VLM completed 512-token text prefill, with fixed-prefix "
            "preparation and continuation validation excluded. No engine state or scheduler."
        ),
        comparison="same fixed prefix and query as single_session native prefill",
        warmup=1,
        repetitions=3,
        run_class="diagnostic",
        invariants=("finite repeatable continuation", "complete cache state"),
    )


prefill_1k = prefill(1024)
prefill_4k = prefill(4096)
prefill_16k = prefill(16384)
bounded_decode_1k = decode(1024, 4)
bounded_decode_4k = decode(4096, 4)
bounded_decode_16k = decode(16384, 4)
continuous_decode_1k = decode(1024, 64)
continuous_decode_4k = decode(4096, 64)
continuous_decode_16k = decode(16384, 64)
