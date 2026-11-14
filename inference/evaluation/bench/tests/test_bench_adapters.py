import hashlib

import pytest
from bench_support import TREE
from magnitude_bench.adapters import ADAPTERS
from magnitude_bench.adapters.omlx_native import instrumentation
from magnitude_bench.models import Acquisition, Drafter, Target, prepare
from magnitude_bench.options import EngineOptions, LlamaOptions, MagnitudeOptions, SourceBuild
from magnitude_bench.record import DRAFT_WIDTH_TRACE
from magnitude_bench.store import RunStore


def make(engine, artifact_path, tmp_path, acquisition, options=None):
    target = Target(engine=engine, reference=str(artifact_path))
    artifact = prepare(Target(engine="mlx-vlm", reference=str(artifact_path)))
    arguments = (target, artifact, acquisition, RunStore(tmp_path / "run"), options or EngineOptions(inference=TREE))
    return ADAPTERS[engine](*arguments)


def flag(argv, name):
    return argv[argv.index(name) + 1]


@pytest.mark.parametrize("engine", ["magnitude", "mlx-vlm", "omlx", "llama.cpp"])
def test_readiness_requires_matching_loaded_capacity(engine, artifact_path, tmp_path, acquisition):
    # Readiness checks stay lightweight and don't load an engine or a model.
    adapter = make(engine, artifact_path, tmp_path, acquisition)
    if engine == "magnitude":
        data = {"ready": True, "model": "magnitude-bench", "context_tokens": 40000, "vocabulary": 9,
                "drafting": {"method": "dflash", "proposals": 16, "policy": "Adaptive"}}
    elif engine == "mlx-vlm":
        data = {
            "status": "healthy",
            "loaded_model": str(artifact_path),
            "effective_context_limit": 40000,
            "apc_enabled": False,
            "continuous_batching_enabled": True,
        }
    elif engine == "omlx":
        data = {
            "ready": True,
            "served_model": "magnitude-bench",
            "loaded": True,
            "context_capacity": 40000,
            "max_concurrent_requests": 4,
            "speculative_backend": "none",
        }
    else:
        data = {"total_slots": 4, "default_generation_settings": {
            "n_ctx": 40000, "params": {"speculative.types": "draft-dflash"}}}
    adapter.verify_ready(data, 40000, 4)
    with pytest.raises(ValueError):
        adapter.verify_ready(data, 41000, 4)


def test_readiness_refuses_an_engine_not_serving_the_runs_drafter(artifact_path, tmp_path, acquisition):
    """Each engine states what it decodes with at readiness; a run that drafts with the catalog's
    drafter refuses an engine serving anything else (a drafter it could not admit serves plain),
    and a run that drafts nothing expects plain decoding."""
    plain = EngineOptions(inference=TREE, drafting=False)
    magnitude = lambda method: {"ready": True, "model": "magnitude-bench", "context_tokens": 40000,
                                "vocabulary": 9, "drafting": {"method": method, "proposals": 0, "policy": None}}
    llama = lambda types: {"total_slots": 1, "default_generation_settings": {
        "n_ctx": 40000, "params": {"speculative.types": types}}}
    for engine, served, refused in (("magnitude", magnitude("dflash"), magnitude("plain")),
                                    ("llama.cpp", llama("draft-dflash"), llama("none"))):
        drafting = make(engine, artifact_path, tmp_path, acquisition)
        drafting.verify_ready(served, 40000, 1)
        with pytest.raises(ValueError, match="serves"):
            drafting.verify_ready(refused, 40000, 1)
        undrafted = make(engine, artifact_path, tmp_path, acquisition, plain)
        undrafted.verify_ready(refused, 40000, 1)
        with pytest.raises(ValueError, match="serves"):
            undrafted.verify_ready(served, 40000, 1)


def test_magnitude_serves_the_catalog_drafter_and_the_run_tuning(
    artifact_path, tmp_path, acquisition
):
    options = EngineOptions(
        inference=TREE,
        magnitude=MagnitudeOptions.of(
            TREE,
            binary=tmp_path / "engine", device="cpu", mtp_proposals=2, tuning_seconds=30,
            fresh_tuning=True, draft_budget='{"generated":2,"verified":1}',
        )
    )
    adapter = make("magnitude", artifact_path, tmp_path, acquisition, options)
    argv = adapter.argv(1234, 8192, 2, tmp_path)
    assert argv[0] == str(tmp_path / "engine")
    assert "--no-projector" in argv and flag(argv, "--context-tokens") == "8192"
    assert flag(argv, "--method") == "dflash"
    assert flag(argv, "--draft") == str(acquisition.drafter.path)
    assert flag(argv, "--mtp-proposals") == "2" and flag(argv, "--device") == "cpu"
    assert flag(argv, "--draft-budget") == options.magnitude.draft_budget
    assert flag(argv, "--tuning-budget") == "30.0"
    assert flag(argv, "--tuning-trace") == str(tmp_path / "run" / "tuning.jsonl")
    assert "--fresh-tuning" in argv
    # Only the run's first launch tunes afresh; later launches reuse what it stored.
    assert "--fresh-tuning" not in adapter.argv(1234, 8192, 2, tmp_path)

    embedded = acquisition.model_copy(
        update={"drafter": Drafter(method="mtp", path=acquisition.target, sha256="0" * 64)}
    )
    argv = make("magnitude", artifact_path, tmp_path, embedded).argv(1234, 8192, 1, tmp_path)
    assert flag(argv, "--method") == "mtp" and "--draft" not in argv
    assert "--tuning-budget" not in argv and "--fresh-tuning" not in argv

    # A drafting engine traces its controller's widths into its log; a plain one doesn't.
    assert adapter.environment()[DRAFT_WIDTH_TRACE] == "1"
    plain = make("magnitude", artifact_path, tmp_path, acquisition, EngineOptions(inference=TREE, drafting=False))
    argv = plain.argv(1234, 8192, 1, tmp_path)
    assert flag(argv, "--method") == "plain" and "--draft" not in argv
    assert DRAFT_WIDTH_TRACE not in plain.environment()
    without = acquisition.model_copy(update={"drafter": None})
    argv = make("magnitude", artifact_path, tmp_path, without).argv(1234, 8192, 1, tmp_path)
    assert flag(argv, "--method") == "plain"


async def test_magnitude_measures_another_builds_binary(artifact_path, tmp_path, acquisition):
    binary = tmp_path / "origin-engine"
    binary.write_bytes(b"origin")
    binary.chmod(0o755)
    build = SourceBuild(commit="b2f79f00f", dirty_digest=None)
    options = EngineOptions(
        inference=TREE,
        magnitude=MagnitudeOptions.of(TREE, binary=binary, build=build, tuning_trace=False)
    )
    adapter = make("magnitude", artifact_path, tmp_path, acquisition, options)
    await adapter.prepare()
    assert adapter.identity["binary"] == str(binary)
    assert adapter.identity["binary_sha256"] == hashlib.sha256(b"origin").hexdigest()
    assert adapter.identity["options"]["build"] == {"commit": "b2f79f00f", "dirty_digest": None}
    argv = adapter.argv(1234, 8192, 1, tmp_path)
    assert argv[0] == str(binary)
    assert not any(argument.startswith("--tuning") or argument == "--fresh-tuning"
                   for argument in argv)
    # A tuning request the binary could not record is refused.
    with pytest.raises(ValueError, match="tuning trace"):
        MagnitudeOptions.of(TREE, tuning_trace=False, tuning_seconds=30)
    with pytest.raises(ValueError, match="tuning trace"):
        MagnitudeOptions.of(TREE, tuning_trace=False, fresh_tuning=True)


async def test_magnitude_serves_from_exactly_the_cache_given(artifact_path, tmp_path, acquisition):
    binary = tmp_path / "engine"
    binary.write_bytes(b"engine")
    binary.chmod(0o755)
    exact = EngineOptions(inference=TREE, magnitude=MagnitudeOptions.of(TREE, binary=binary, cache_dir=tmp_path / "c"))
    adapter = make("magnitude", artifact_path, tmp_path, acquisition, exact)
    await adapter.prepare()
    assert adapter.identity["kernel_cache"] == str(tmp_path / "c")
    assert flag(adapter.argv(1234, 8192, 1, tmp_path), "--cache-dir") == str(tmp_path / "c")
    unset = make("magnitude", artifact_path, tmp_path, acquisition,
                 EngineOptions(inference=TREE, magnitude=MagnitudeOptions.of(TREE, binary=binary)))
    await unset.prepare()
    assert unset.identity["kernel_cache"] is None and "--cache-dir" not in unset.argv(1234, 8192, 1, tmp_path)


def test_a_request_the_drafter_proposed_nothing_for_is_a_valid_measurement():
    """An engine's draft controller may choose not to draft a request: its response omits the
    draft counters (`engine/serving/src/chat.rs`, which states them only when it proposed some),
    which reports none proposed, not a fault. Such a request is measured, and adds nothing to
    the drafted totals."""
    from magnitude_bench import report, validation

    def response(drafted):
        timings = {"cache_n": 0, "prompt_n": 10, "predicted_n": 64, "prompt_ms": 5.0, "predicted_ms": 50.0,
                   **({"draft_n": drafted[0], "draft_n_accepted": drafted[1]} if drafted else {})}
        payload = {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 64, "total_tokens": 74,
                                            "prompt_tokens_details": {"cached_tokens": 0}}, "timings": timings}
        return validation.terminal(payload)

    declined, drafted = response(None), response((7, 5))
    assert "draft_n" not in declined["timings"]
    rows = [{"target": "magnitude", "section": "context", "checkpoint": 4096, "concurrency": 1, "family": "prose",
             "phase": "measured", "timing_basis": "native",
             "observation": {"outcome": "valid", "terminal": terminal, "ttft_ms": 5, "completed_ms": 55}}
            for terminal in (declined, drafted)]
    (row,) = report.summarize(rows, "completed", "c", "id", 2)["rows"]
    assert row["eligible"] == 2 and row["draft_proposed"] == 7 and row["draft_accepted"] == 5


def test_llama_allocates_context_headroom_within_the_model_limit(
    artifact_path, tmp_path, acquisition
):
    options = EngineOptions(inference=TREE, llama=LlamaOptions(context_headroom=1024))
    adapter = make("llama.cpp", artifact_path, tmp_path, acquisition, options)
    adapter.executable = str(tmp_path / "llama-server")
    limit = adapter.artifact.context_limit
    context = limit - 4096
    argv = adapter.argv(1234, context, 2, tmp_path)
    assert flag(argv, "--ctx-size") == str((context + 1024) * 2)
    adapter.verify_ready(
        {"total_slots": 2, "default_generation_settings": {
            "n_ctx": context + 1024, "params": {"speculative.types": "draft-dflash"}}}, context, 2
    )
    with pytest.raises(ValueError):
        adapter.verify_ready(
            {"total_slots": 2, "default_generation_settings": {"n_ctx": context}}, context, 2
        )
    # At the model's limit there is no room to add.
    assert flag(adapter.argv(1234, limit, 1, tmp_path), "--ctx-size") == str(limit)


def test_llama_serves_the_catalog_drafter(artifact_path, tmp_path, acquisition):
    adapter = make("llama.cpp", artifact_path, tmp_path, acquisition)
    adapter.executable = str(tmp_path / "llama-server")
    argv = adapter.argv(1234, 65536, 1, tmp_path)
    assert flag(argv, "--model") == str(artifact_path)
    assert flag(argv, "--ctx-size") == "65536" and flag(argv, "--parallel") == "1"
    assert flag(argv, "--n-gpu-layers") == "99"
    assert flag(argv, "--model-draft") == str(acquisition.drafter.path)
    assert flag(argv, "--spec-type") == "draft-dflash"
    assert flag(argv, "--spec-draft-n-max") == "3"
    assert flag(argv, "--n-gpu-layers-draft") == "99"
    assert "--no-cache-prompt" in argv and "--no-context-shift" in argv

    embedded = acquisition.model_copy(
        update={"drafter": Drafter(method="mtp", path=acquisition.target, sha256="0" * 64)}
    )
    adapter = make("llama.cpp", artifact_path, tmp_path, embedded)
    adapter.executable = str(tmp_path / "llama-server")
    argv = adapter.argv(1234, 65536, 1, tmp_path)
    assert flag(argv, "--spec-type") == "draft-mtp" and "--model-draft" not in argv

    adapter = make("llama.cpp", artifact_path, tmp_path, acquisition, EngineOptions(inference=TREE, drafting=False))
    adapter.executable = str(tmp_path / "llama-server")
    assert "--spec-type" not in adapter.argv(1234, 65536, 1, tmp_path)


async def test_llama_refuses_a_drafter_it_cannot_serve(artifact_path, tmp_path, acquisition):
    unknown = acquisition.model_copy(
        update={"drafter": Drafter(method="eagle", path=tmp_path / "d.gguf", sha256="1" * 64)}
    )
    with pytest.raises(ValueError, match="--drafting off"):
        await make("llama.cpp", artifact_path, tmp_path, unknown).prepare()


def test_llama_serves_dflash2_drafters_on_its_dflash_implementation(
    artifact_path, tmp_path, acquisition
):
    dflash2 = acquisition.model_copy(
        update={"drafter": Drafter(method="dflash2", path=tmp_path / "d.gguf", sha256="1" * 64)}
    )
    adapter = make("llama.cpp", artifact_path, tmp_path, dflash2)
    adapter.executable = str(tmp_path / "llama-server")
    assert flag(adapter.argv(1234, 65536, 1, tmp_path), "--spec-type") == "draft-dflash"


def test_gguf_engines_are_counted_by_the_harness(artifact_path, tmp_path, acquisition):
    for engine in ("magnitude", "llama.cpp", "ollama"):
        gguf = Target(engine=engine, reference=str(acquisition.target), sha256="0" * 64)
        adapter = ADAPTERS[engine](
            gguf, prepare(Target(engine="mlx-vlm", reference=str(artifact_path))), acquisition,
            RunStore(tmp_path / "run"), EngineOptions(inference=TREE),
        )
        assert adapter.host_rendered
        if engine != "ollama":
            with pytest.raises(NotImplementedError, match="counted by magnitude-harness"):
                adapter.counter()
    for engine in ("mlx-vlm", "omlx"):
        assert not make(engine, artifact_path, tmp_path, acquisition).host_rendered
    registry = ADAPTERS["ollama-registry"](
        Target(engine="ollama-registry", reference="ollama:a:b"),
        prepare(Target(engine="mlx-vlm", reference=str(artifact_path))), acquisition,
        RunStore(tmp_path / "run"), EngineOptions(inference=TREE),
    )
    assert registry.host_rendered
    mlx_tag = ADAPTERS["ollama-mlx"](
        Target(engine="ollama-mlx", reference="ollama:a:b"),
        prepare(Target(engine="mlx-vlm", reference=str(artifact_path))), acquisition,
        RunStore(tmp_path / "run"), EngineOptions(inference=TREE),
    )
    assert not mlx_tag.host_rendered


def test_omlx_native_enrichment_keeps_evidence_and_checks_phase_boundary():
    metric = instrumentation.RequestMetrics(prompt_ms=10)
    instrumentation._record_batched_service_time(metric, 5, has_uncached_prompt_token=True)
    instrumentation._record_batched_service_time(metric, 7, has_uncached_prompt_token=False)
    assert metric.prompt_ms == 15
    assert metric.generation_ms == 7
    key = "test-native-timing"
    instrumentation._metrics[key] = metric
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    try:
        value = instrumentation._terminal_payload({"usage": usage, "choices": []}, key)
        assert value["usage"] == usage
        assert value["timings"]["predicted_ms"] == 7
        usage["total_tokens"] = 99
        with pytest.raises(RuntimeError, match="disagrees"):
            instrumentation._terminal_payload({"usage": usage}, key)
    finally:
        instrumentation._metrics.pop(key)


async def test_ollama_mlx_tag_counts_with_its_tokenizer_json(artifact_path, tmp_path, acquisition):
    from types import SimpleNamespace

    from magnitude_bench.adapters.ollama import Ollama
    from magnitude_bench.adapters.ollama_native.prompts import PromptFormat
    from magnitude_bench.sizing import chat
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    adapter = Ollama(
        Target(engine="ollama-mlx", reference="ollama:test:tag"),
        prepare(Target(engine="mlx-vlm", reference=str(artifact_path))),
        acquisition,
        RunStore(tmp_path / "run"),
        EngineOptions(inference=TREE),
    )
    adapter.tag = SimpleNamespace(layer=lambda **_: path)
    adapter.format = PromptFormat("qwen3.8", False)
    body = chat([{"role": "user", "content": "hello"}], [])
    async with adapter.counter() as lengths:
        measured = await lengths([body, body])
    text = adapter.format.chat_text({"messages": body["messages"], "think": False})
    assert measured == [len(tokenizer.encode(text, add_special_tokens=False).ids)] * 2


def test_acquisition_reads_the_contract_object(tmp_path):
    value = {
        "model": "qwen3.5-4b:gguf:q4",
        "target": "/m/target.gguf",
        "target_sha256": "0" * 64,
        "drafter": {"method": "mtp", "path": "/m/target.gguf", "sha256": "0" * 64},
        "projector": None,
    }
    acquisition = Acquisition.model_validate(value)
    assert acquisition.embedded_drafter
    with pytest.raises(ValueError):
        Acquisition.model_validate({**value, "extra": 1})


def test_explicit_mtp_keeps_its_separate_catalog_head(artifact_path, tmp_path, acquisition):
    mtp = acquisition.model_copy(update={"drafter": acquisition.drafter.model_copy(update={"method": "mtp"})})
    options = EngineOptions(inference=TREE, magnitude=MagnitudeOptions.of(TREE, method="mtp"))
    argv = make("magnitude", artifact_path, tmp_path, mtp, options).argv(1234, 8192, 1, tmp_path)
    assert flag(argv, "--method") == "mtp"
    assert flag(argv, "--draft") == str(mtp.drafter.path)
    # An override away from a DFlash catalog entry uses the embedded MTP head.
    argv = make("magnitude", artifact_path, tmp_path, acquisition, options).argv(1234, 8192, 1, tmp_path)
    assert "--draft" not in argv
