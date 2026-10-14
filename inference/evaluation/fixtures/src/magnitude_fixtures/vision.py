"""Deterministic image conversations with exact, visible color answers."""

import base64
import hashlib
import json
import struct
import zlib

from .fixture import FixedSizing, Fixture, Generation, Source


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


def fixtures() -> tuple[Fixture, ...]:
    red, blue = (255, 0, 0), (0, 0, 255)
    cases = (
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
    for name, urls, question, values in cases:
        answer = json.dumps(values, separators=(",", ":"))
        content = [{"type": "image_url", "image_url": {"url": url}} for url in urls]
        content.append(
            {
                "type": "text",
                "text": question
                + " Reply only with a JSON object using these keys: "
                + ", ".join(values)
                + ". Use lowercase color names as string values. Do not use Markdown.",
            }
        )
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": answer, "scored": True},
        ]
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
        results.append(
            Fixture(
                id="vision/" + name,
                family="vision",
                source=Source(
                    name="deterministic-rgb-png/1",
                    digest=hashlib.sha256(
                        json.dumps(messages, sort_keys=True).encode()
                    ).hexdigest(),
                ),
                messages=messages,
                generation=Generation(
                    prefix=len(messages) - 1, max_tokens=80, expected={"values": values}
                ),
                sizing=FixedSizing(),
            )
        )
    return tuple(results)
