"""Thin access to the shared fixture package; no second content or tokenizer pipeline."""

import asyncio
from pathlib import Path


def tokens(artifact, *, fixture, context_tokens, continuation_tokens):
    from benchmark_fixtures.preparation import Fixture, Tokenization, prepare

    tokenizer = Tokenization(Path(artifact))
    prepared = asyncio.run(
        prepare(
            Fixture(
                identity=fixture,
                context_tokens=context_tokens,
                continuation_tokens=continuation_tokens,
            ),
            tokenizer,
        )
    )

    return prepared.model_copy(
        update={"provenance": {**prepared.provenance, "eos_tokens": sorted(tokenizer.eos_tokens)}}
    )


def record_inputs(run, **values):
    """Keep supplied inputs reproducible and prevent same-shape evidence collisions."""
    from performance.records import digest

    run.record["inputs"] = values
    run.record["workload"]["input_digest"] = digest(values)
