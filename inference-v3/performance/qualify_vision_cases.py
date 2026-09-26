"""Public serving qualification for image order, repeated identity and follow-up turns."""

import argparse
import asyncio
import base64
import io
import json
import re
from pathlib import Path

import httpx
from PIL import Image, ImageDraw
from qualify_vision_serving import image_part

from engine.platform.backend import Backend
from engine.serving.app import create_app
from engine.serving.runtime import Config


def diagram():
    image = Image.new("RGB", (192, 128), "white")
    drawing = ImageDraw.Draw(image)
    drawing.ellipse((12, 34, 72, 94), fill="red")
    drawing.polygon(((135, 24), (97, 102), (179, 102)), fill="blue")
    with io.BytesIO() as encoded:
        image.save(encoded, format="PNG")
        payload = base64.b64encode(encoded.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + payload}}


def text(value):
    return {"type": "text", "text": value}


def cases():
    red, blue = image_part("red"), image_part("blue")
    order = text(
        "Name the solid colors of the two images in order. Output only the two color names."
    )
    return (
        (
            "diagram",
            [
                {
                    "role": "user",
                    "content": [
                        diagram(),
                        text(
                            "What shape is the red object on the left? "
                            "Answer with only the shape name."
                        ),
                    ],
                }
            ],
            ("circle",),
        ),
        ("red-blue", [{"role": "user", "content": [red, blue, order]}], ("red", "blue")),
        ("blue-red", [{"role": "user", "content": [blue, red, order]}], ("blue", "red")),
        ("repeated-red", [{"role": "user", "content": [red, red, order]}], ("red", "red")),
        (
            "follow-up",
            [
                {"role": "user", "content": [red, text("What solid color is this?")]},
                {"role": "assistant", "content": "red"},
                {
                    "role": "user",
                    "content": [
                        blue,
                        text(
                            "What solid color is this NEW image? Answer with only the color name."
                        ),
                    ],
                },
            ],
            ("blue",),
        ),
    )


async def main(args):
    app = create_app(
        Config(
            target=str(args.target),
            memory_bytes=12 << 30,
            context_tokens=1024,
            parallel_sequences=3,
            prefill_tokens=64,
            output_capacity=32,
            backend=Backend(args.backend),
        )
    )
    evidence = []
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=None
        ) as client:

            async def ask(case):
                name, messages, expected = case
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "magnitude",
                        "temperature": 0,
                        "max_tokens": 24,
                        "reasoning_effort": "none",
                        "messages": messages,
                    },
                )
                result = {"case": name, "status": response.status_code, "body": response.json()}
                evidence.append(result)
                args.output.write_text(json.dumps(evidence, indent=2) + "\n")
                print(json.dumps(result), flush=True)
                assert response.status_code == 200, result
                content = result["body"]["choices"][0]["message"]["content"]
                words = re.findall(r"[a-z]+", content.lower())
                assert (
                    tuple(word for word in words if word in {"red", "blue", "circle"}) == expected
                ), result

            fixtures = cases()
            await ask(fixtures[0])
            await asyncio.gather(*(ask(case) for case in fixtures[1:4]))
            await ask(fixtures[4])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", choices=("metal", "cuda"), default="metal")
    asyncio.run(main(parser.parse_args()))
