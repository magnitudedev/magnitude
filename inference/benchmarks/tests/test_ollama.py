import json

import pytest

from magnitude_benchmarks.adapters import ADAPTERS
from magnitude_benchmarks.adapters.ollama_native import translate
from magnitude_benchmarks.session_bench import validation
from magnitude_benchmarks.session_bench.models import Artifact, ArtifactFile, Target, prepare
from magnitude_benchmarks.session_bench.options import EngineOptions, OllamaOptions, Schedule
from magnitude_benchmarks.session_bench.policy import project_root
from magnitude_benchmarks.session_bench.results import RunStore, public_command

# Final chunks as Ollama 0.35.1 streamed them for one prompt sent twice to a loaded model.
COLD = {
    "done": True,
    "done_reason": "length",
    "load_duration": 1039684541,
    "prompt_eval_count": 8001,
    "prompt_eval_cached_count": 0,
    "prompt_eval_duration": 11553573000,
    "eval_count": 32,
    "eval_duration": 612988000,
}
CACHED = {**COLD, "prompt_eval_cached_count": 7997, "prompt_eval_duration": 53837000}
OVERFLOW = (
    '{"error":"{\\"error\\":{\\"code\\":400,\\"message\\":\\"request (8001 tokens) exceeds the '
    'available context size (4096 tokens), try increasing it\\",\\"type\\":'
    '\\"exceed_context_size_error\\",\\"n_prompt_tokens\\":8001,\\"n_ctx\\":4096}}"}'
)


def adapter(engine, kind, tmp_path, options=None):
    path = tmp_path / ("model.gguf" if kind == "gguf" else "4b-mlx")
    path.write_bytes(b"never loaded")
    reference = str(path) if kind == "gguf" else "ollama:qwen3.5:4b-mlx"
    artifact = Artifact(
        reference=reference,
        path=path,
        kind=kind,
        context_limit=262144,
        metadata={},
        files=(ArtifactFile(path=path.name, size=12, sha256="0" * 64, mtime_ns=0),),
    )
    value = ADAPTERS[engine](
        project_root(),
        Target(engine=engine, reference=reference),
        artifact,
        RunStore(tmp_path, "test", {}),
        options or EngineOptions(ollama=OllamaOptions(models=tmp_path / "store")),
    )
    value.executable = str(tmp_path / "ollama")
    value.store_directory = tmp_path / "store"
    return value


def test_ollama_counters_become_native_timings_the_bench_accepts():
    cold = translate.terminal(COLD)
    assert cold["timings"]["prompt_n"] == 8001 and cold["timings"]["cache_n"] == 0
    assert cold["timings"]["prompt_ms"] == pytest.approx(11553.573)
    assert cold["timings"]["predicted_ms"] == pytest.approx(612.988)
    validation.terminal({"choices": [], **cold})

    # The prompt count includes reused tokens; the duration covers only the evaluated ones.
    cached = translate.terminal(CACHED)
    assert cached["timings"]["cache_n"] == 7997 and cached["timings"]["prompt_n"] == 4
    assert cached["usage"]["prompt_tokens"] == 8001
    assert cached["usage"]["prompt_tokens_details"]["cached_tokens"] == 7997
    validation.terminal({"choices": [], **cached})

    with pytest.raises(ValueError):
        translate.terminal({k: v for k, v in COLD.items() if k != "prompt_eval_duration"})
    with pytest.raises(ValueError):
        translate.terminal({**COLD, "prompt_eval_cached_count": 9000})


def test_ollama_request_fixes_context_and_disables_thinking():
    body = {
        "model": "session-bench",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 256,
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = translate.chat_request(body, "stored-name", 66048)
    assert request["model"] == "stored-name" and request["think"] is False
    assert request["keep_alive"] == -1 and request["stream"] is True
    assert request["options"] == {
        "num_ctx": 66048,
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
        "num_predict": 256,
    }
    assert "tools" not in request


def test_ollama_overflow_rejection_states_the_prompt_size():
    assert translate.overflow_prompt_tokens(400, OVERFLOW) == 8001
    assert translate.overflow_prompt_tokens(500, OVERFLOW) is None
    assert translate.overflow_prompt_tokens(400, '{"error":"model not found"}') is None


def test_ollama_stream_chunks_become_deltas():
    assert translate.delta({"message": {"content": "Call"}}, 0) == {"content": "Call"}
    assert translate.delta({"message": {"thinking": "hm"}}, 0) == {"reasoning_content": "hm"}
    call = {"function": {"name": "echo", "arguments": {"value": 7}}}
    delta = translate.delta({"message": {"tool_calls": [call]}}, 2)
    assert delta["tool_calls"][0]["index"] == 2
    assert json.loads(delta["tool_calls"][0]["function"]["arguments"]) == {"value": 7}
    assert translate.finish_reason({"done_reason": "length"}, False) == "length"
    assert translate.finish_reason({"done_reason": "stop"}, True) == "tool_calls"
    with pytest.raises(ValueError):
        translate.finish_reason({"done_reason": "load"}, False)


def test_answer_prefill_sends_ollamas_glimmer_prompt_through_the_raw_route(tmp_path):
    body = {
        "messages": [
            {"role": "system", "content": "Reading session."},
            {"role": "user", "content": "Copy it."},
            {"role": "assistant", "content": "Copied."},
            {"role": "user", "content": "Again."},
        ],
        "max_tokens": 256,
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    chat = translate.chat_request(body, "muse-glimmer:30b", 65808)
    raw = translate.prefilled_request(chat, "llama-server")
    # The MLX runner does not add the leading token on the raw route; the llama.cpp runner does.
    assert translate.prefilled_request(chat, "mlx")["prompt"] == "<|begin_of_text|>" + raw["prompt"]
    assert raw["prompt"] == (
        "<|start|>system<|message|>Reading session.\n\nReasoning strength: none."
        '\n\n# Valid recipients: "self", "user".<|eot|>'
        "<|start|>user<|message|>Copy it.<|eot|>"
        "<|start|>assistant to=user<|message|>Copied.<|eot|>"
        "<|start|>user<|message|>Again.<|eot|>"
        "<|start|>assistant to=user<|message|>"
    )
    assert raw["raw"] is True and raw["truncate"] is False and "think" not in raw
    assert raw["options"] == chat["options"] and raw["keep_alive"] == chat["keep_alive"]
    assert translate.delta({"response": "Call"}, 0) == {"content": "Call"}
    for messages in (
        body["messages"][1:],
        [{"role": "system", "content": "Reasoning strength: high."}, body["messages"][1]],
        [*body["messages"], {"role": "tool", "content": "x"}],
    ):
        with pytest.raises(ValueError):
            translate.glimmer_prompt(messages)
    with pytest.raises(ValueError):
        translate.prefilled_request({**chat, "tools": [{"type": "function"}]}, "llama-server")

    options = EngineOptions(ollama=OllamaOptions(models=tmp_path / "store", answer_prefill=True))
    selected = adapter("ollama-registry", "registry", tmp_path, options)
    assert "--answer-prefill" in selected.argv(1234, 65536, 1, tmp_path / "engine")
    plain = adapter("ollama-registry", "registry", tmp_path)
    assert "--answer-prefill" not in plain.argv(1234, 65536, 1, tmp_path / "engine")
    command = public_command([selected.target], ("session",), (65536,), (), 1, options=options)
    assert "--ollama-answer-prefill" in command


@pytest.mark.parametrize(
    ("engine", "kind", "model_format", "runner"),
    [
        ("ollama", "gguf", "gguf", "llama-server"),
        ("ollama-mlx", "registry", "safetensors", "mlx"),
        ("ollama-registry", "registry", "gguf", "llama-server"),
    ],
)
def test_ollama_readiness_requires_context_runner_format_and_full_gpu(
    engine, kind, model_format, runner, tmp_path
):
    value = adapter(engine, kind, tmp_path)
    data = {
        "ready": True,
        "served_model": "session-bench",
        "context_capacity": 66048,
        "max_concurrent_requests": 1,
        "model_format": model_format,
        "runner": runner,
        "speculation": "off",
        "context_headroom": 0,
        "answer_prefill": False,
        # A tag with a separate draft model loads two models; Ollama's sizes then cover a
        # fraction of the load, so the llama.cpp runner is judged by its own load lines.
        "gpu_layers": [[36, 36], [5, 5]] if runner == "llama-server" else [],
        "mlx_device": "gpu" if runner == "mlx" else None,
        "size_bytes": 703709838 if runner == "llama-server" else 5853119774,
        "size_vram_bytes": 703709838 if runner == "llama-server" else 5853119774,
    }
    value.verify_ready(data, 66048, 1)
    placement = (
        ({"gpu_layers": [[35, 36], [5, 5]]}, {"gpu_layers": [[36, 36], [0, 5]]}, {"gpu_layers": []})
        if runner == "llama-server"
        else ({"size_vram_bytes": 1}, {"mlx_device": "cpu"}, {"mlx_device": None})
    )
    for change in (
        {"context_capacity": 4096},
        {"model_format": "gguf" if model_format == "safetensors" else "safetensors"},
        *placement,
        {"runner": None},
        {"speculation": "default"},
        {"context_headroom": 16},
        {"answer_prefill": True},
        {"max_concurrent_requests": 4},
    ):
        with pytest.raises(ValueError):
            value.verify_ready({**data, **change}, 66048, 1)


def test_ollama_launch_selects_source_and_leaves_defaults_alone(tmp_path):
    imported = adapter("ollama", "gguf", tmp_path)
    argv = imported.argv(1234, 66048, 1, tmp_path / "engine")
    assert argv[argv.index("--gguf") + 1] == str(tmp_path / "model.gguf")
    assert argv[argv.index("--context-capacity") + 1] == "66048"
    assert argv[argv.index("--store") + 1] == str(tmp_path / "store")
    assert argv[argv.index("--flash-attention") + 1] == "auto"
    assert argv[argv.index("--speculation") + 1] == "off"
    assert argv[argv.index("--context-headroom") + 1] == "0"
    roomy = adapter(
        "ollama",
        "gguf",
        tmp_path,
        EngineOptions(ollama=OllamaOptions(models=tmp_path / "store", context_headroom=16)),
    )
    argv = roomy.argv(1234, 66048, 1, tmp_path / "engine")
    assert argv[argv.index("--context-headroom") + 1] == "16"
    # A launch at the model's context limit cannot allocate beyond it.
    argv = roomy.argv(1234, 262144, 1, tmp_path / "engine")
    assert argv[argv.index("--context-headroom") + 1] == "0"
    assert "--kv-cache-type" not in argv and "--registry-model" not in argv

    options = EngineOptions(
        ollama=OllamaOptions(models=tmp_path / "store", kv_cache_type="q8_0", flash_attention="on")
    )
    pulled = adapter("ollama-mlx", "registry", tmp_path, options)
    argv = pulled.argv(1234, 66048, 1, tmp_path / "engine")
    assert argv[argv.index("--registry-model") + 1] == "qwen3.5:4b-mlx"
    assert argv[argv.index("--kv-cache-type") + 1] == "q8_0"
    assert "--gguf" not in argv
    command = public_command([pulled.target], ("session",), (65536,), (), 1, options=options)
    assert "--ollama-kv-cache-type q8_0" in command and "--ollama-flash-attention on" in command


def test_passes_replace_the_balanced_schedule_and_are_reproduced(tmp_path):
    assert Schedule().blocks(1, 1) == 2 and Schedule().blocks(3, 2) == 6
    assert Schedule(passes=1).blocks(3, 2) == 1
    options = EngineOptions(schedule=Schedule(passes=1))
    target = Target(engine="magnitude", reference=str(tmp_path / "model.gguf"))
    assert "--passes 1" in public_command([target], ("session",), (65536,), (), 1, options=options)
    assert "--passes" not in public_command([target], ("session",), (65536,), (), 1)


def test_registry_artifact_is_pinned_by_its_manifest(tmp_path, monkeypatch):
    store = tmp_path / "store"
    blobs = store / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "sha256-aa").write_text(
        json.dumps({"model_format": "safetensors", "file_type": "nvfp4", "renderer": "qwen3.5"})
    )
    (blobs / "sha256-bb").write_text(
        json.dumps({"model_type": "qwen3_5", "text_config": {"max_position_embeddings": 262144}})
    )
    manifest = store / "manifests/registry.ollama.ai/library/qwen3.5/4b-mlx"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "config": {"digest": "sha256:aa"},
                "layers": [
                    {"name": "config.json", "digest": "sha256:bb", "size": 10},
                    {"name": "w", "digest": "sha256:cc", "size": 90},
                ],
            }
        )
    )
    monkeypatch.setenv("OLLAMA_MODELS", str(store))
    artifact = prepare(Target(engine="ollama-mlx", reference="ollama:qwen3.5:4b-mlx"))
    assert artifact.kind == "registry" and artifact.context_limit == 262144
    assert artifact.metadata["quantization"] == "nvfp4" and artifact.metadata["bytes"] == 100
    artifact.verify_unchanged()
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ValueError):
        artifact.verify_unchanged()

    with pytest.raises(ValueError):
        prepare(Target(engine="ollama-mlx", reference="ollama:qwen3.5:missing"))
    with pytest.raises(ValueError):
        prepare(Target(engine="ollama-mlx", reference=str(tmp_path)))
    with pytest.raises(ValueError):
        prepare(Target(engine="ollama", reference="ollama:qwen3.5:4b-mlx"))


def test_a_prompt_ollama_truncates_is_counted_from_its_warning():
    line = (
        b'time=2026-10-06T07:42:32.879Z level=WARN source=llama_server.go:320 msg="truncating '
        b'input prompt" limit=2058 prompt=5406 keep=4 new=2058'
    )
    assert translate.truncated_prompt_tokens(line) == 5406
    assert translate.truncated_prompt_tokens(b"slot print_timing: id  0 | task 26 |") is None


def test_llama_runner_speculation_is_read_from_its_launch_line_and_slot_timings():
    launch = (
        b'time=2026-10-06T06:06:06.403Z level=INFO source=llama_server.go:436 msg="starting '
        b'llama-server" cmd="/o/llama-server --model /o/blob -c 8192 --spec-type draft-mtp '
        b'--spec-draft-n-max 2 --flash-attn auto"'
    )
    assert "--spec-type draft-mtp" in translate.llama_launch(launch)
    assert translate.llama_launch(b"slot print_timing: id  0 | task 0 |") is None
    timing = (
        b"slot print_timing: id  0 | task 0 | draft acceptance = 0.97059 (  132 accepted /   "
        b"136 generated), mean len =  3.00"
    )
    assert translate.speculation_stats(timing) == (136, 132)
    mlx = b'level=INFO msg="speculative decode stats" drafted=15 accepted=14'
    assert translate.speculation_stats(mlx) == (15, 14)


def test_registry_gguf_artifact_reads_its_context_from_the_model_layer(tmp_path, monkeypatch):
    from gguf import GGUFWriter

    store = tmp_path / "store"
    blobs = store / "blobs"
    blobs.mkdir(parents=True)
    writer = GGUFWriter(str(blobs / "sha256-dd"), "qwen35")
    writer.add_context_length(262144)
    writer.add_file_type(15)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()
    (blobs / "sha256-aa").write_text(json.dumps({"model_format": "gguf", "file_type": "Q4_K_M"}))
    manifest = store / "manifests/registry.ollama.ai/library/qwen3.5/4b-q4_K_M"
    manifest.parent.mkdir(parents=True)
    layer = {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:dd", "size": 90}
    manifest.write_text(json.dumps({"config": {"digest": "sha256:aa"}, "layers": [layer]}))
    monkeypatch.setenv("OLLAMA_MODELS", str(store))
    reference = "ollama:qwen3.5:4b-q4_K_M"
    artifact = prepare(Target(engine="ollama-registry", reference=reference))
    assert artifact.kind == "registry" and artifact.context_limit == 262144
    assert artifact.metadata["architecture"] == "qwen35"
    assert artifact.metadata["quantization"] == "Q4_K_M"
    assert artifact.metadata["model_sha256"] == "dd"
    # Each registry engine serves one stored format.
    with pytest.raises(ValueError):
        prepare(Target(engine="ollama-mlx", reference=reference))


def test_ollama_prompt_identity_ignores_sampling_and_records_reported_context():
    body = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 256, "seed": 42}
    assert translate.prompt_identity(body) == translate.prompt_identity({**body, "seed": 7})
    changed = {**body, "messages": [{"role": "user", "content": "hello!"}]}
    assert translate.prompt_identity(body) != translate.prompt_identity(changed)
    assert translate.terminal(COLD, 66048)["timings"]["n_ctx"] == 66048
    validation.terminal({"choices": [], **translate.terminal(COLD, 66048)})


def test_ollama_speculation_statistics_become_draft_counters():
    line = (
        b"time=2026-10-06T05:00:00Z level=INFO source=speculate_stats.go:61 "
        b'msg="speculative decode stats" iterations=20 drafted=57 accepted=41 acceptance=0.72'
    )
    assert translate.speculation_stats(line) == (57, 41)
    assert translate.speculation_stats(b'msg="Loaded draft model" arrays=58') is None
    evidence = translate.terminal(COLD, 66048, (57, 41))
    assert evidence["timings"]["draft_n"] == 57 and evidence["timings"]["draft_n_accepted"] == 41
    validation.terminal({"choices": [], **evidence})
    assert "draft_n" not in translate.terminal(COLD, 66048)["timings"]


def test_gpu_placement_is_read_from_the_runners_load_lines():
    assert translate.offloaded_layers(b"load_tensors: offloaded 36/36 layers to GPU") == (36, 36)
    assert translate.offloaded_layers(b"load_tensors: offloaded 12/61 layers to GPU") == (12, 61)
    assert translate.offloaded_layers(b"load_tensors:  MTL0_Mapped model buffer size") is None
    line = b'level=INFO msg="MLX engine initialized" "MLX version"=0.32.3-0-g64ea011 device=gpu'
    assert translate.mlx_device(line) == "gpu"
    assert translate.mlx_device(b'msg="mlx runner is ready" port=57428') is None
