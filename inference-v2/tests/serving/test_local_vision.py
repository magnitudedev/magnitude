"""Opt-in real-artifact image qualification across the host/worker/API boundary."""

import base64
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from magnitude_engine import blueprints as bp
from magnitude_engine.artifacts.tokenizer import TokenizerArtifact
from magnitude_engine.serving.app import create_app
from magnitude_engine.serving.session import ChatService
from magnitude_engine.worker.host import Worker
from tests.worker.test_worker import compose_engine


def image_request(color, size=(112, 112)):
    data = io.BytesIO()
    Image.new("RGB", size, color).save(data, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()
    return {
        "model": "vision-test",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the main color in this image? Answer in one word.",
                    },
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
        "max_tokens": 8,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }


@pytest.mark.model
@pytest.mark.parametrize("owned", [True, False])
@pytest.mark.parametrize("family", ["qwen35", "gemma4"])
def test_images_are_row_local_and_semantic_prefix_reuse_is_safe(owned, family):
    path = os.environ.get(f"MAGNITUDE_TEST_{family.upper()}_VISION")
    if not path:
        pytest.skip(f"set MAGNITUDE_TEST_{family.upper()}_VISION to a local full artifact")
    artifact = TokenizerArtifact.load(Path(path))
    head = os.environ.get("MAGNITUDE_TEST_MTP") if owned and family == "qwen35" else None
    engine = compose_engine(
        path,
        head=head,
        context_tokens=8192,
        max_active=4,
        prefill_tokens=32,
        memory_bytes=int(os.environ.get("MAGNITUDE_TEST_MEMORY_GIB", "28")) << 30,
    )
    if family == "gemma4":
        program = bp.model.programs.gemma4.Program(artifact=bp.model.artifacts.Local(path=path))
        engine = replace(
            engine,
            generation=bp.generation.Generation(
                target=bp.model.Executor(
                    program=program,
                    state=bp.model.state.PagedHybrid(),
                )
            ),
        )
    if not owned:
        program = bp.model.upstream.mlx_vlm.ModelLoader(
            artifact=bp.model.artifacts.Local(path=path)
        )
        engine = replace(
            engine,
            generation=bp.generation.Generation(
                target=bp.model.Executor(
                    program=program,
                    state=bp.model.state.Native(source=program),
                )
            ),
        )
    elif os.environ.get("MAGNITUDE_TEST_STREAMED") or os.environ.get("MAGNITUDE_TEST_EXPERTS"):
        program = engine.generation.target.program
        if os.environ.get("MAGNITUDE_TEST_STREAMED"):
            program = replace(program, embedding=bp.model.embeddings.Streamed())
        if os.environ.get("MAGNITUDE_TEST_EXPERTS"):
            assert family == "qwen35"
            program = replace(
                program,
                feedforward=bp.model.feedforward.qwen35.MoE(
                    experts=bp.model.experts.Streamed(slots=8),
                ),
            )
        generation = replace(
            engine.generation,
            target=replace(engine.generation.target, program=program),
        )
        if head:
            method = generation.method
            head_program = replace(method.drafter.program, target=program)
            generation = replace(
                generation,
                method=replace(
                    method,
                    drafter=replace(
                        method.drafter,
                        program=head_program,
                        state=replace(method.drafter.state, source=head_program),
                    ),
                ),
            )
        engine = replace(engine, generation=generation)
    with Worker.start(engine=engine) as host:
        assert host.properties["image_processor"] is not None
        if head:
            assert host.properties["speculative_backend"] == "mtp"
        service = ChatService(host, artifact, "vision-test")
        with TestClient(create_app(service)) as client:

            def infer(body):
                response = client.post("/v1/chat/completions", json=body)
                assert response.status_code == 200, response.text + host.stderr
                return response.json()

            with ThreadPoolExecutor(max_workers=2) as pool:
                red, blue = tuple(
                    pool.map(infer, [image_request("red"), image_request("blue", (224, 112))])
                )
            assert "red" in red["choices"][0]["message"]["content"].lower(), red
            assert "blue" in blue["choices"][0]["message"]["content"].lower(), blue
            details = red["usage"]["prompt_tokens_details"]
            assert details["media_tokens"] > 0
            assert details["media_tokens"] + details["text_tokens"] == red["usage"]["prompt_tokens"]
            assert red["timings"]["preparation_ms"] > 0
            if head:
                assert red["timings"]["draft_n"] > 0
            warm = infer(image_request("red"))
            assert warm["choices"][0]["message"] == red["choices"][0]["message"]
            assert warm["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
            green = infer(image_request("lime"))
            assert "green" in green["choices"][0]["message"]["content"].lower(), green
            assert green["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
            assert green["timings"]["preparation_ms"] > 0
            changed = image_request("red")
            changed["messages"][0]["content"][0]["text"] = (
                "Identify the color shown. Answer in one word."
            )
            reuse = infer(changed)
            assert "red" in reuse["choices"][0]["message"]["content"].lower(), reuse
            assert reuse["timings"]["cached_input_features"] == 1
            assert reuse["timings"]["preparation_ms"] == 0
            multiple = image_request("red")
            multiple["messages"][0]["content"][0]["text"] = (
                "Name the colors of the first and second images, in order."
            )
            multiple["messages"][0]["content"].append(
                image_request("blue")["messages"][0]["content"][1]
            )
            multiple["max_tokens"] = 64
            answer = infer(multiple)["choices"][0]["message"]["content"].lower()
            assert (
                "red" in answer and "blue" in answer and answer.index("red") < answer.index("blue")
            ), answer
            conversation = image_request("red")
            conversation["messages"].extend(
                [
                    {"role": "assistant", "content": "Red."},
                    {"role": "user", "content": "What color was the image? Answer in one word."},
                ]
            )
            followup = infer(conversation)
            assert "red" in followup["choices"][0]["message"]["content"].lower(), followup
            # Ordinary photo geometry exceeds the former prepared-JSON limit.
            large = image_request("red", (1600, 1200))
            large["stream"] = True
            large["stream_options"] = {"include_usage": True}
            streamed = client.post("/v1/chat/completions", json=large)
            assert streamed.status_code == 200, streamed.text + host.stderr
            records = [line[6:] for line in streamed.text.splitlines() if line.startswith("data: ")]
            assert records[-1] == "[DONE]", streamed.text
            chunks = [json.loads(record) for record in records[:-1]]
            assert (
                "red"
                in "".join(
                    choice.get("delta", {}).get("content", "") or ""
                    for chunk in chunks
                    for choice in chunk.get("choices", [])
                ).lower()
            ), streamed.text
            assert any(
                chunk.get("usage", {}).get("prompt_tokens", 0) > 0
                for chunk in chunks
                if chunk.get("usage") is not None
            )
            if family == "gemma4":
                many = image_request("red")
                many["messages"][0]["content"].extend(
                    image_request("red")["messages"][0]["content"][1] for _ in range(15)
                )
                result = infer(many)
                assert "red" in result["choices"][0]["message"]["content"].lower(), result
            text = infer(
                {
                    "model": "vision-test",
                    "messages": [{"role": "user", "content": "Say hello."}],
                    "max_tokens": 8,
                    "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            )
            assert "hello" in text["choices"][0]["message"]["content"].lower(), text
