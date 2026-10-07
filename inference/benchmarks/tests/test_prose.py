import json
import shlex
from pathlib import Path

import httpx
import pytest
from test_client import Fragments, terminal_event
from test_runner import FixtureAdapter

from magnitude_benchmarks.fixtures.prose_history import (
    CONTINUATION_WORDS,
    Prose,
    ProseHistory,
)
from magnitude_benchmarks.session_bench import cli, runner
from magnitude_benchmarks.session_bench.client import measure
from magnitude_benchmarks.session_bench.models import Target
from magnitude_benchmarks.session_bench.results import public_command
from magnitude_benchmarks.session_bench.suites import SECTIONS, compile_plan


@pytest.fixture
def source():
    return Prose("".join(f"word{i} " for i in range(15000)), {"sha256": "pinned-test-source"})


async def count(context):
    return len(json.dumps(context.model_dump(mode="json")))


async def test_all_schedules_share_prose_policy_and_preserve_history(source):
    plan = await compile_plan(
        source,
        source.identity,
        tuple(SECTIONS),
        (4096, 65536),
        counter=count,
        sizing_identity="test",
    )
    assert plan == await compile_plan(
        source,
        source.identity,
        tuple(SECTIONS),
        (4096, 65536),
        counter=count,
        sizing_identity="test",
    )
    seen = {}
    for request in plan.requests:
        assert request.workload == "prose"
        assert request.output_limit == 256
        assert request.tools == request.expected == []
        assert "tools" not in request.body("model")
        assert "tool_choice" not in request.body("model")
        assert request.fixture_provenance["actual_context_tokens"] >= request.checkpoint
        for dependency in request.depends_on:
            assert dependency in seen
            previous = seen[dependency]
            if request.session == previous.session:
                assert request.messages[: len(previous.messages)] == previous.messages
                assert request.messages[len(previous.messages)]["role"] == "assistant"
        seen[request.id] = request
    assert len(plan.warmup.messages[-1]["content"]) <= 1024
    assert plan.warmup.body("model")["max_tokens"] == 256


async def test_prose_continuation_uses_next_book_words_and_exhaustion_fails(source):
    history = ProseHistory(source, "one")
    first = await history.prepare(4096, count, "test")
    end = first.provenance["passage_end_word"]
    history.complete()
    assert history.messages[-1]["content"] == history.passage(end, end + CONTINUATION_WORDS)
    second = await history.prepare(16384, count, "test")
    assert second.provenance["passage_start_word"] == end + CONTINUATION_WORDS
    with pytest.raises(ValueError, match="exceeds"):
        await history.prepare(10**9, count, "test")


@pytest.fixture
def book():
    front = " ".join(f"front{i}" for i in range(300))
    paragraphs = [
        " ".join(f"“paragraph{paragraph}word{word}’s”" for word in range(80))
        for paragraph in range(200)
    ]
    return f"{front}CHAPTER 1. Loomings.\n\nCall me Ishmael. " + "\n\n".join(paragraphs)


async def word_count(context):
    # A renderer with explicit framing overhead, shared by complete-context and
    # marginal copy sizing; paragraph selection cannot count framing as source.
    assert all(message["content"] for message in context.messages)
    return 7 + sum(5 + len(message["content"].split()) for message in context.messages)


async def test_prose_repeat_places_passage_before_final_copy_instruction(book):
    source = Prose(book, {"sha256": "pinned-test-source"}, "prose-repeat")
    assert source.identity != Prose(book, {"sha256": "pinned-test-source"}).identity
    prepared = await ProseHistory(source, "one").prepare(4096, count, "test")
    messages = prepared.content.messages
    assert [message["role"] for message in messages] == ["system", "assistant", "user"]
    passage, instruction = messages[-2]["content"], messages[-1]["content"]
    assert passage.startswith('Call me Ishmael. "paragraph0word0\'s" ')
    assert "\n\n" in passage
    assert not any(quote in passage for quote in "‘’“”")
    assert "repeat the supplied passage verbatim to the end." in instruction
    assert instruction.endswith("Output only the passage text.")
    assert len(instruction) < len(passage)
    assert prepared.provenance["fixture"] == "prose.moby-dick.repeat"
    assert prepared.provenance["recipe"] == "prose-repeat-history-v2"
    assert prepared.tokens >= 4096


@pytest.mark.parametrize("output_tokens", [64, 256, 512])
async def test_prose_repeat_selects_latest_sufficient_paragraph_deterministically(book, output_tokens):
    source = Prose(book, {"sha256": "pinned"}, "prose-repeat")
    starts = []
    for checkpoint in (1024, 2048, 4096):
        history = ProseHistory(source, "one", output_tokens=output_tokens)
        prepared = await history.prepare(checkpoint, word_count, "words")
        again = await ProseHistory(source, "one", output_tokens=output_tokens).prepare(
            checkpoint, word_count, "words"
        )
        assert prepared == again
        provenance = prepared.provenance
        start, end = provenance["copy_start_word"], provenance["passage_end_word"]
        selected = history.passage(start, end)
        supplied = prepared.content.messages[-2]["content"]
        paragraphs = supplied.split("\n\n")
        suffixes = ["\n\n".join(paragraphs[index:]) for index in range(len(paragraphs))]
        qualifying = [suffix for suffix in suffixes if len(suffix.split()) >= output_tokens]
        assert selected == qualifying[-1]
        assert provenance["copy_available_tokens"] == len(selected.split())
        assert provenance["copy_available_tokens"] >= output_tokens
        assert provenance["copy_output_tokens"] == output_tokens
        assert " ".join(selected.split()[:6]) in prepared.content.messages[-1]["content"]
        assert start >= provenance["passage_start_word"]
        starts.append(start)
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)


async def test_prose_repeat_session_copies_selected_suffix_then_moves_to_new_text(book):
    history = ProseHistory(Prose(book, {"sha256": "pinned"}, "prose-repeat"), "one")
    first = await history.prepare(4096, count, "test")
    start, end = first.provenance["copy_start_word"], first.provenance["passage_end_word"]
    history.complete()
    assert history.messages[:-1] == first.content.messages
    assert history.messages[-1] == {"role": "assistant", "content": history.passage(start, end)}
    second = await history.prepare(8192, count, "test")
    assert second.provenance["passage_start_word"] == end
    assert second.content.messages[: len(history.messages)] == history.messages
    assert [message["role"] for message in second.content.messages[-2:]] == ["assistant", "user"]


async def test_prose_repeat_schedules_preserve_shared_history_and_output_budget(book):
    source = Prose(book, {"sha256": "pinned"}, "prose-repeat")
    plan = await compile_plan(
        source, source.identity, tuple(SECTIONS), (4096,), counter=count, sizing_identity="test"
    )
    seen = {}
    for request in plan.requests:
        assert request.fixture_provenance["copy_output_tokens"] == request.output_limit == 256
        assert request.fixture_provenance["copy_available_tokens"] >= request.output_limit
        assert [message["role"] for message in request.messages[-2:]] == ["assistant", "user"]
        for dependency in request.depends_on:
            previous = seen[dependency]
            if previous.session == request.session:
                assert request.messages[: len(previous.messages)] == previous.messages
                assert request.messages[len(previous.messages)]["role"] == "assistant"
        seen[request.id] = request


async def test_prose_repeat_single_paragraph_uses_start_without_fallback():
    source = Prose("Call me Ishmael. " + " ".join(f"word{i}" for i in range(4000)), {}, "prose-repeat")
    prepared = await ProseHistory(source, "one").prepare(2048, word_count, "words")
    assert prepared.provenance["copy_start_word"] == prepared.provenance["passage_start_word"]
    assert prepared.provenance["copy_available_tokens"] >= 256


async def test_prose_repeat_exhaustion_fails_clearly():
    source = Prose("Call me Ishmael. " + " ".join(f"word{i}" for i in range(300)), {}, "prose-repeat")
    with pytest.raises(ValueError, match="(?i)(remaining|enough|short|exceed|budget)"):
        await ProseHistory(source, "one", output_tokens=1024).prepare(0, word_count, "words")


@pytest.mark.parametrize(
    "finish,tokens,outcome",
    [
        ("stop", 4, "valid"),
        ("length", 256, "valid"),
        ("length", 4, "truncated"),
        ("length", 257, "protocol-error"),
        ("tool_calls", 4, "protocol-error"),
    ],
)
async def test_prose_completion_budget_is_not_tool_truncation(source, finish, tokens, outcome):
    plan = await compile_plan(
        source, source.identity, ("single",), (4096,), counter=count, sizing_identity="test"
    )
    terminal = terminal_event()
    terminal["usage"].update(completion_tokens=tokens, total_tokens=10 + tokens)
    terminal["timings"]["predicted_n"] = tokens
    events = [
        {"id": "response-1", "choices": [{"index": 0, "delta": {"content": "Continuation."}}]},
        {"id": "response-1", "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
        terminal,
        "[DONE]",
    ]

    def response(request):
        body = json.loads(request.content)
        assert body["max_tokens"] == 256 and "tools" not in body
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Fragments(events)
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        result = await measure(client, "http://test", "model", plan.requests[0], lambda _: None)
    assert result.outcome == outcome


@pytest.mark.parametrize(
    "deltas,reason",
    [
        ([{"reasoning_content": "Copy it."}, {"content": "Continuation."}], "reasoning text"),
        ([{"reasoning_content": "Copy it."}], "reasoning text"),
        ([{}], "no answer text"),
    ],
)
async def test_prose_refuses_reasoning_and_missing_answers(source, deltas, reason):
    plan = await compile_plan(
        source, source.identity, ("single",), (4096,), counter=count, sizing_identity="test"
    )
    events = [
        *({"id": "response-1", "choices": [{"index": 0, "delta": delta}]} for delta in deltas),
        {"id": "response-1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        terminal_event(),
        "[DONE]",
    ]

    def response(request):
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Fragments(events)
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        result = await measure(client, "http://test", "model", plan.requests[0], lambda _: None)
    assert result.outcome == "invalid" and reason in result.error
    assert result.ttft_ms is None or deltas[-1].get("content")


@pytest.mark.parametrize("workload", ["prose-continue", "prose-repeat"])
def test_prose_command_roundtrip_and_incompatible_filters(tmp_path, workload):
    target = Target(engine="magnitude", reference=str(tmp_path))
    command = public_command(
        [target], ("context",), (65536,), ("simple-python",), 1, prose=workload
    )
    args = cli.parser().parse_args(shlex.split(command)[4:])
    assert args.workload == workload and args.category is None
    assert "--category" not in command
    for flag in ("--case", "--category"):
        argv = ["run", "--target", f"magnitude={tmp_path}", "--workload", workload, flag, "all"]
        assert cli.main(argv) == 2
    assert runner.capacity([{"request": 65536}], [66000], 256) == 65792


async def test_prose_real_process_run_records_mode_and_no_bfcl(
    tmp_path, artifact_path, monkeypatch, source
):
    async def prepare():
        return source.text, source.provenance

    async def forbidden(*args):
        raise AssertionError("prose must not acquire BFCL")

    monkeypatch.setattr(runner.prose_source, "prepare", prepare)
    monkeypatch.setattr(runner.corpus, "prepare", forbidden)
    monkeypatch.setitem(runner.ADAPTERS, "mlx-vlm", FixtureAdapter)
    result = await runner.run(
        tmp_path,
        [Target(engine="mlx-vlm", reference=str(artifact_path))],
        ("single", "session", "fork"),
        (4096,),
        (),
        1,
        None,
        lambda _: None,
        prose="prose-continue",
    )
    assert result["status"] == "completed", result
    assert result["workload"] == "prose-continue"
    path = Path(result["path"])
    assert "--workload prose-continue" in (path / "command.txt").read_text()
    report = (path / "report.md").read_text()
    assert "no answer-quality scoring" in report
    assert "BFCL" not in report
    assert all(row["actual_completion_tokens"] == [4] for row in result["rows"])


async def test_repeat_warmup_includes_its_own_supplied_passage(book):
    source = Prose(book, {"sha256": "pinned"}, "prose-repeat")
    plan = await compile_plan(
        source, source.identity, ("context",), (4096,),
        counter=word_count, sizing_identity="words",
    )
    warmup = plan.warmup
    assert warmup.id == "warmup"
    assert [m["role"] for m in warmup.messages] == ["system", "assistant", "user"]
    assert warmup.fixture_provenance["copy_available_tokens"] >= warmup.output_limit
    assert warmup.messages[0] != plan.requests[0].messages[0]
    assert warmup.fixture_provenance["actual_context_tokens"] < 4096
