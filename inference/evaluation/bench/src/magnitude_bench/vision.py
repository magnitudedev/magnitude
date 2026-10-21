"""Qualify catalog image serving with and without its drafter on fixed fixtures.

Run through magnitude-remote, which records source, artifacts and host evidence.
This checks observable answers and speculative serving, not distributional quality.
"""

import argparse
import asyncio
from pathlib import Path

import httpx
from magnitude_fixtures.canonical import digest
from magnitude_fixtures.vision import fixtures

from .adapters.native import Native
from .client import measure
from .models import Acquisition, Target, prepare
from .options import EngineOptions, MagnitudeOptions
from .requests import Request
from .store import RunStore


async def run(args):
    acquisition = Acquisition.model_validate_json(args.artifacts.read_text())
    if acquisition.projector is None or acquisition.drafter is None:
        raise ValueError("vision qualification requires the catalog projector and drafter")
    target = Target(
        engine="magnitude", reference=str(acquisition.target), sha256=acquisition.target_sha256
    )
    artifact = prepare(target)
    store = RunStore(args.output / "vision")
    cases = fixtures()
    (store.path / "fixtures.jsonl").write_text("".join(f.model_dump_json() + "\n" for f in cases))
    store.event(
        "inputs",
        catalog=acquisition.model_dump(mode="json"),
        fixtures_sha256=digest([f.model_dump(mode="json") for f in cases]),
    )
    observations = []
    try:
        for drafting in (False, True):
            options = EngineOptions(
                drafting=drafting,
                vision=True,
                magnitude=MagnitudeOptions(
                    binary=args.binary,
                    device=args.magnitude_device,
                    kernel_cache_root=args.cache,
                    tuning_seconds=10,
                ),
            )
            adapter = Native(target, artifact, acquisition, store, options)
            await adapter.prepare()
            store.event("adapter", identity=adapter.identity)
            async with (
                adapter.launch(8192, 1, "draft" if drafting else "plain") as engine,
                httpx.AsyncClient(timeout=None, trust_env=False) as client,
            ):
                for fixture in cases:
                    request = Request(
                        id=fixture.id + ("-draft" if drafting else "-plain"),
                        section="single",
                        session=fixture.id,
                        checkpoint=0,
                        family=fixture.family,
                        fixture_id=fixture.id,
                        messages=fixture.messages[: fixture.generation.prefix],
                        tools=fixture.tools,
                        generation=fixture.generation,
                    )
                    engine.started(request.id)
                    observation = await measure(
                        client,
                        engine.endpoint,
                        engine.model,
                        request,
                        lambda event, request_id=request.id: store.record_stream(
                            "stream.jsonl", {"request_id": request_id, **event}
                        ),
                        extensions=adapter.extensions,
                    )
                    engine.finished(request.id)
                    observation = adapter.verify_observation(observation)
                    store.append("observations.jsonl", observation.model_dump(mode="json"))
                    observations.append(observation)
                    print(observation.model_dump_json(), flush=True)
        success = all(o.outcome == "valid" for o in observations)
        store.complete(
            {
                "status": "passed" if success else "failed",
                "observations": [o.model_dump(mode="json") for o in observations],
            },
            "# Image serving qualification\n\n"
            + "\n".join(f"- {o.request_id}: {o.outcome}" for o in observations)
            + "\n",
        )
        return 0 if success else 1
    finally:
        store.close_streams()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--magnitude-device", default=MagnitudeOptions().device)
    raise SystemExit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
