import json
import shlex
from pathlib import Path

import httpx
import pytest
from test_client import Fragments, terminal_event
from test_runner import FixtureAdapter

from benchmark_fixtures.prose_history import CONTINUATION_WORDS, Prose, ProseHistory
from session_bench import cli, runner
from session_bench.client import measure
from session_bench.models import Target
from session_bench.results import public_command
from session_bench.suites import SECTIONS, compile_plan


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


def test_prose_command_roundtrip_and_incompatible_filters(tmp_path):
    target = Target(engine="magnitude", reference=str(tmp_path))
    command = public_command([target], ("context",), (65536,), ("simple-python",), 1, prose=True)
    args = cli.parser().parse_args(shlex.split(command)[4:])
    assert args.prose and args.category is None
    assert "--category" not in command
    for flag in ("--case", "--category"):
        assert cli.main(["run", "--target", f"magnitude={tmp_path}", "--prose", flag, "all"]) == 2
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
    monkeypatch.setitem(runner.ADAPTERS, "magnitude", FixtureAdapter)
    result = await runner.run(
        tmp_path,
        [Target(engine="magnitude", reference=str(artifact_path))],
        ("single", "session", "fork"),
        (4096,),
        (),
        1,
        None,
        lambda _: None,
        prose=True,
    )
    assert result["status"] == "completed", result
    assert result["workload"] == "prose"
    path = Path(result["path"])
    assert "--prose" in (path / "command.txt").read_text()
    report = (path / "report.md").read_text()
    assert "no answer-quality scoring" in report
    assert "BFCL" not in report
    assert all(row["actual_completion_tokens"] == [4] for row in result["rows"])
