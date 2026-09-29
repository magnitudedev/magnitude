"""Exercise image bytes through HTTP preparation, scheduling and generation."""

import argparse
import asyncio
import base64
import io
import json
from pathlib import Path

import httpx
from PIL import Image

from engine.platform.backend import Backend
from engine.serving.app import create_app
from engine.serving.runtime import Config


def image_part(color):
    with io.BytesIO() as buffer:
        Image.new("RGB", (64, 96), color).save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}}


async def main(artifact, output, backend):
    app = create_app(
        Config(
            target=str(artifact),
            memory_bytes=8 << 30,
            context_tokens=512,
            parallel_sequences=3,
            prefill_tokens=128,
            backend=Backend(backend),
        )
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=None
        ) as client:

            async def ask(content):
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "magnitude",
                        "temperature": 0,
                        "max_tokens": 16,
                        "reasoning_effort": "none",
                        "messages": [{"role": "user", "content": content}],
                    },
                )
                result = {"status": response.status_code, "body": response.json()}
                print(json.dumps(result), flush=True)
                return result

            prompt = {
                "type": "text",
                "text": "What is the single solid color of this image? "
                "Answer with only the color name.",
            }
            first = await ask([image_part("red"), prompt])
            mixed = await asyncio.gather(
                ask([image_part("blue"), prompt]),
                ask([image_part("red"), prompt]),
                ask("Reply with only the word LANTERN."),
            )
            results = [first, *mixed]
            output.write_text(json.dumps(results, indent=2) + "\n")
            for result, expected in zip(results, ("red", "blue", "red", "lantern"), strict=True):
                assert result["status"] == 200, result
                text = result["body"]["choices"][0]["message"]["content"]
                assert expected in text.lower(), (expected, text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", choices=("metal", "cuda"), default="metal")
    args = parser.parse_args()
    asyncio.run(main(args.artifact, args.output, args.backend))
