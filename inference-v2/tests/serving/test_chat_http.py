import json
import os
import signal
import socket
import subprocess
import sys
from threading import Event, Thread

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient
from tokenizers import Tokenizer, decoders, models
from transformers import PreTrainedTokenizerFast

from magnitude_engine.artifacts.tokenizer import TokenizerArtifact
from magnitude_engine.serving.app import create_app
from magnitude_engine.serving.session import ChatService
from magnitude_engine.worker.host import Worker
from tests.models.architectures.qwen35.test_construction import artifact_pair
from tests.worker.test_worker import compose_engine


@pytest.fixture
def service(tmp_path):
    target, head, _, _ = artifact_pair(tmp_path)
    vocab = {chr(32 + i): i for i in range(127)}
    vocab["[EOS]"] = 127
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens(["[EOS]"])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="[EOS]")
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    tokenizer.save_pretrained(target)
    (target / "generation_config.json").write_text('{"eos_token_id":127}')
    config = compose_engine(
        str(target),
        str(head),
        memory_bytes=64 << 20,
        context_tokens=256,
        output_capacity=2,
    )
    with Worker(config, startup_timeout=20) as host:
        yield ChatService(host, TokenizerArtifact.load(target), "test")


def payload(**changes):
    return {
        "model": "test",
        "messages": [{"role": "user", "content": "x"}],
        "max_tokens": 40,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"schema": {"const": {"answer": 7}}},
        },
        **changes,
    }


def parse_sse(text):
    events = [block.removeprefix("data: ") for block in text.strip().split("\n\n")]
    assert events[-1] == "[DONE]" and events.count("[DONE]") == 1
    return [json.loads(event) for event in events[:-1]]


def test_http_nonstream_and_stream_use_real_worker_and_agree_on_semantic_output(service):
    with TestClient(create_app(service)) as client:
        assert client.get("/health").json()["status"] == "ready"
        assert client.get("/v1/models").json()["data"][0]["id"] == "test"
        complete = client.post("/v1/chat/completions", json=payload())
        assert complete.status_code == 200, complete.text
        output = complete.json()
        assert json.loads(output["choices"][0]["message"]["content"]) == {"answer": 7}
        streamed = client.post(
            "/v1/chat/completions",
            json=payload(stream=True, stream_options={"include_usage": True}),
        )
        assert streamed.status_code == 200, streamed.text
        events = parse_sse(streamed.text)
        assert not any("error" in event for event in events), events
        assert len({event["id"] for event in events}) == 1
        text = "".join(
            choice.get("delta", {}).get("content", "")
            for event in events
            for choice in event["choices"]
        )
        assert text == output["choices"][0]["message"]["content"]
        terminal = events[-1]
        assert terminal["choices"] == [] and events[-2]["choices"][0]["finish_reason"]
        usage, timings = terminal["usage"], terminal["timings"]
        assert usage["prompt_tokens"] == timings["cache_n"] + timings["prompt_n"]
        assert usage["completion_tokens"] == timings["predicted_n"]
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        assert 0 <= timings["draft_n_accepted"] <= timings["draft_n"]
        native = terminal["engine"]["native"]
        assert native["forced_tokens"] > 0
        assert timings["prompt_ms"] + timings["predicted_ms"] == pytest.approx(
            (native["prefill_ns"] + native["decode_ns"]) / 1e6
        )
        assert not service.host._requests


def test_http_admission_errors_and_zero_token_completion_are_request_local(service):
    with TestClient(create_app(service)) as client:
        for change, status in (
            ({"model": "missing"}, 404),
            ({"max_tokens": 1000}, 400),
            ({"max_tokens": True}, 422),
            ({"n": 2}, 422),
            ({"n": True}, 422),
            ({"logprobs": True}, 422),
            ({"stop": ""}, 422),
        ):
            response = client.post("/v1/chat/completions", json=payload(**change))
            assert response.status_code == status and "error" in response.json(), response.text
        response = client.post("/v1/chat/completions", json=payload(max_tokens=0))
        assert response.status_code == 200
        assert response.json()["choices"][0]["finish_reason"] == "length"
        assert response.json()["usage"]["completion_tokens"] == 0
        assert client.get("/health").status_code == 200 and not service.host._requests


def test_http_string_stop_cancels_worker_and_retains_native_work_evidence(service):
    with TestClient(create_app(service)) as client:
        response = client.post(
            "/v1/chat/completions",
            json=payload(stop="answer", stream=True, stream_options={"include_usage": True}),
        )
        events = parse_sse(response.text)
        assert not any("error" in event for event in events), events
        text = "".join(
            choice.get("delta", {}).get("content", "")
            for event in events
            for choice in event["choices"]
        )
        assert "answer" not in text and "7" not in text
        assert events[-1]["engine"]["string_stop"] == "answer"
        assert events[-1]["engine"]["native"]["reason"] == "cancelled"
        assert events[-2]["choices"][0]["finish_reason"] == "stop"
        assert not service.host._requests


def test_http_health_tracks_worker_loss(service):
    with TestClient(create_app(service)) as client:
        os.kill(service.host.process.pid, signal.SIGKILL)
        service.host.process.wait(timeout=3)
        assert client.get("/health").status_code == 503
        assert client.post("/v1/chat/completions", json=payload()).status_code == 503


def test_server_import_does_not_initialize_mlx():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import magnitude_engine.serving.__main__; "
            "assert not any(k == 'mlx' or k.startswith('mlx.') for k in sys.modules)",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_tcp_disconnect_cancels_only_its_request_and_keeps_peer_service_healthy(
    service, monkeypatch
):
    submitted, cancelled = Event(), Event()
    submit = service.host.submit

    def observe(*args, **kwargs):
        request = submit(*args, **kwargs)
        cancel = request.cancel

        def record_cancel(*args, **kwargs):
            result = cancel(*args, **kwargs)
            cancelled.set()
            return result

        request.cancel = record_cancel
        submitted.set()
        return request

    monkeypatch.setattr(service.host, "submit", observe)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    address = f"http://127.0.0.1:{listener.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(create_app(service), log_level="error", lifespan="off"))
    thread = Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        with httpx.Client(base_url=address, timeout=10) as client:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json=payload(
                    stream=True,
                    max_tokens=180,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "schema": {"const": "a" * 1000},
                        },
                    },
                ),
            ) as response:
                assert response.status_code == 200
                assert next(line for line in response.iter_lines() if line.startswith("data: "))
                assert submitted.wait(5)
            assert cancelled.wait(5), "disconnect did not complete worker cancellation"
            assert not service.host._requests
            peer = client.post("/v1/chat/completions", json=payload())
            assert peer.status_code == 200
            assert json.loads(peer.json()["choices"][0]["message"]["content"]) == {"answer": 7}
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
    assert not thread.is_alive()
