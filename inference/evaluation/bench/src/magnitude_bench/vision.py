"""Qualify catalog image serving with and without its drafter on fixed image conversations.

A capability check that multimodal input is accepted and answered, not a benchmark request: its
cases are the bench's own, never a template.

Run through `mrig serve`, which records source, artifacts and host evidence.
This checks observable answers and speculative serving, not distributional quality.
"""

import argparse
import asyncio
import base64
import struct
import zlib
from pathlib import Path

import httpx
from magnitude_fixtures import Generation
from magnitude_fixtures.canonical import digest

from .adapters.native import Native
from .client import measure
from .models import Acquisition, Target, prepare
from .options import EngineOptions, MagnitudeOptions
from .requests import Request
from .store import RunStore


def image(width: int, height: int, colors: tuple[tuple[int, int, int], ...]) -> str:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    row = b"\0" + b"".join(
        bytes(colors[min(x * len(colors) // width, len(colors) - 1)]) for x in range(width)
    )
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(row * height))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


#: The cases: (name, the conversation's messages, its question's expected values).
Case = tuple[str, list[dict], dict[str, str]]


def cases() -> tuple[Case, ...]:
    """Deterministic image conversations with exact, visible color answers."""
    red, blue = (255, 0, 0), (0, 0, 255)
    asked = (
        ("solid", [image(224, 224, (red,))], "What color is the image?", {"color": "red"}),
        (
            "details",
            [image(896, 896, (red, blue))],
            "What colors are the left and right halves of the image?",
            {"left": "red", "right": "blue"},
        ),
        (
            "ordered",
            [image(224, 224, (red,)), image(224, 224, (blue,)), image(224, 224, (red,))],
            "What color is each image, in order?",
            {"first": "red", "second": "blue", "third": "red"},
        ),
        (
            "history",
            [image(224, 224, (blue,))],
            "What colors are the earlier image and the current image?",
            {"earlier": "red", "current": "blue"},
        ),
    )
    results = []
    for name, urls, question, values in asked:
        content: list[dict] = [{"type": "image_url", "image_url": {"url": url}} for url in urls]
        content.append(
            {
                "type": "text",
                "text": question
                + " Reply only with a JSON object using these keys: "
                + ", ".join(values)
                + ". Use lowercase color names as string values. Do not use Markdown.",
            }
        )
        messages: list[dict] = [{"role": "user", "content": content}]
        if name == "history":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image(224, 224, (red,))}},
                        {"type": "text", "text": "Describe this image."},
                    ],
                },
                {"role": "assistant", "content": "The image is red."},
                *messages,
            ]
        results.append((name, messages, values))
    return tuple(results)


def vision_request(case: Case, drafting: bool) -> Request:
    name, messages, values = case
    return Request(
        id=name + ("-draft" if drafting else "-plain"),
        section="vision",
        session=name,
        checkpoint=0,
        template=None,
        lane=0,
        key="vision",
        messages=messages,
        tools=[],
        generation=Generation(prefix=len(messages), max_tokens=80, expected={"values": values}),
    )


async def run(args):
    acquisition = Acquisition.model_validate_json(args.artifacts.read_text())
    if acquisition.projector is None or acquisition.drafter is None:
        raise ValueError("vision qualification requires the catalog projector and drafter")
    target = Target(
        engine="magnitude", reference=str(acquisition.target), sha256=acquisition.target_sha256
    )
    artifact = prepare(target)
    store = RunStore(args.output / "vision")
    asked = cases()
    store.event(
        "inputs",
        catalog=acquisition.model_dump(mode="json"),
        cases_sha256=digest([list(case) for case in asked]),
    )
    observations = []
    try:
        for drafting in (False, True):
            options = EngineOptions(
                inference=args.inference,
                drafting=drafting,
                vision=True,
                magnitude=MagnitudeOptions.of(
                    args.inference,
                    binary=args.binary,
                    device=args.magnitude_device,
                    cache_dir=args.cache,
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
                for case in asked:
                    request = vision_request(case, drafting)
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
    parser.add_argument("--inference", type=Path, required=True, help="the checkout's inference/ tree")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--magnitude-device", default=MagnitudeOptions.model_fields["device"].default)
    raise SystemExit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
