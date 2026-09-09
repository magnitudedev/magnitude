import json
import shlex
from pathlib import Path

import httpx
import pytest
from test_client import Fragments, terminal_event
from test_runner import FixtureAdapter

from benchmark_fixtures.contexts import Context
from benchmark_fixtures.ruler import RulerFixture
from session_bench import cli, report, runner
from session_bench.client import measure
from session_bench.models import Target
from session_bench.results import public_command
from session_bench.sessions import Request
from session_bench.suites import SECTIONS, compile_plan


async def count(context):
    return len(json.dumps(context.model_dump(mode="json")))


async def plan_for(sections=("session",), contexts=(4096, 16384, 4096)):
    fixture = RulerFixture(seed=42, variant="multiquery", queries=4)
    return await compile_plan(
        fixture, fixture.identity, sections, contexts, counter=count, sizing_identity="test"
    )


async def test_all_schedules_resize_without_answer_leakage_and_keep_wire_contract():
    plan = await plan_for(tuple(SECTIONS))
    assert plan == await plan_for(tuple(SECTIONS))
    assert plan.parallel_sequences == 8
    seen = set()
    for request in plan.requests:
        assert request.id not in seen
        assert set(request.depends_on) <= seen
        seen.add(request.id)
        assert request.workload == "retrieval"
        assert request.output_limit == 1024
        assert request.tools == []
        assert all(message["role"] != "assistant" for message in request.messages)
        assert [message["role"] for message in request.messages] == ["system", "user"]
        body = request.body("test")
        assert "tools" not in body and "tool_choice" not in body
        saved = Request.model_validate_json(request.model_dump_json())
        assert json.dumps(saved.body("test")) == json.dumps(body)
    session = [r for r in plan.requests if r.section == "session"]
    assert session[0].messages == session[2].messages
    assert session[0].expected == session[1].expected == session[2].expected
    assert (
        session[0].messages[-1]["content"].split("\n\n")[-1]
        == session[1].messages[-1]["content"].split("\n\n")[-1]
    )
    assert plan.warmup.expected != session[0].expected
    assert plan.warmup.messages[0] != session[0].messages[0]
    assert plan.warmup.fixture_provenance["actual_context_tokens"] == await count(
        Context(messages=plan.warmup.messages)
    )


@pytest.mark.parametrize("mode", ["correct", "wrong", "malformed", "truncated", "protocol"])
async def test_streaming_scores_only_complete_retrieval_responses(mode):
    request = (await plan_for(("single",))).requests[0]
    answer = dict(request.expected.values)
    if mode == "wrong":
        answer[next(iter(answer))] = "wrong"
    output = "not JSON" if mode == "malformed" else json.dumps(answer)
    terminal = terminal_event()
    events = [
        {"id": "response-1", "choices": [{"index": 0, "delta": {"content": output[:8]}}]},
        {"id": "response-1", "choices": [{"index": 0, "delta": {"content": output[8:]}}]},
        {
            "id": "response-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "length" if mode == "truncated" else "stop",
                }
            ],
        },
        terminal,
        "[DONE]",
    ]
    if mode == "protocol":
        events.pop()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=Fragments(events)
            )
        )
    ) as client:
        result = await measure(client, "http://test", "model", request, lambda _: None)
    assert (
        result.outcome
        == {
            "correct": "valid",
            "wrong": "invalid",
            "malformed": "invalid",
            "truncated": "truncated",
            "protocol": "protocol-error",
        }[mode]
    )
    if mode in ("truncated", "protocol"):
        assert result.retrieval is None
    else:
        assert result.retrieval.correct == {"correct": 4, "wrong": 3, "malformed": 0}[mode]


def test_command_roundtrip_preserves_configuration_and_resize_order(tmp_path):
    fixture = RulerFixture(seed=17, variant="multiquery", queries=3)
    command = public_command(
        [Target(engine="magnitude", reference=str(tmp_path))],
        ("context",),
        (4096, 16384, 4096),
        (),
        2,
        retrieval=fixture,
        needle_depth=0.8,
    )
    args = cli.parser().parse_args(shlex.split(command)[4:])
    assert args.retrieval and not args.prose and args.category is None
    assert (args.retrieval_seed, args.retrieval_variant, args.retrieval_queries) == (
        17,
        "multiquery",
        3,
    )
    assert args.needle_depth == 0.8
    assert cli.contexts(args.context, preserve_order=True) == (4096, 16384, 4096)
    for flags in (
        ("--retrieval", "--category", "all"),
        ("--retrieval-seed", "3"),
        ("--retrieval", "--needle-depth", "nan"),
    ):
        assert cli.main(["run", "--target", f"magnitude={tmp_path}", *flags]) == 2


async def test_retrieval_real_process_run_and_score_denominators(
    tmp_path, artifact_path, monkeypatch
):
    async def forbidden(*args, **kwargs):
        raise AssertionError("retrieval must not acquire BFCL or prose")

    monkeypatch.setattr(runner.corpus, "prepare", forbidden)
    monkeypatch.setattr(runner.prose_source, "prepare", forbidden)
    monkeypatch.setitem(runner.ADAPTERS, "magnitude", FixtureAdapter)
    progress = []
    result = await runner.run(
        tmp_path,
        [Target(engine="magnitude", reference=str(artifact_path))],
        ("single", "session", "fork"),
        (2048, 4096, 2048),
        (),
        1,
        None,
        progress.append,
        retrieval=RulerFixture(variant="multiquery", queries=4),
    )
    assert result["status"] == "completed", result
    assert result["workload"] == "retrieval"
    assert all(row["retrieval"]["exact_accuracy"] == 1 for row in result["rows"])
    assert any("retrieval 4/4" in line for line in progress)
    path = Path(result["path"])
    assert "--retrieval" in (path / "command.txt").read_text()
    assert "Exact answers" in (path / "report.md").read_text()
    rows = [json.loads(line) for line in (path / "results.jsonl").read_text().splitlines()]
    measured = next(row for row in rows if row["phase"] == "measured")
    wrong = {
        **measured,
        "observation": {
            **measured["observation"],
            "outcome": "invalid",
            "retrieval": {"correct": 3, "total": 4, "exact_match": False, "format_valid": True},
        },
    }
    failure = {**measured, "observation": {"outcome": "timeout"}}
    summary = report.summarize([measured, wrong, failure], "failed", "test", "test", 3)
    score = summary["rows"][0]["retrieval"]
    assert score["exact_accuracy"] == 1 / 3
    assert score["field_accuracy"] == 7 / 12
    assert score["scored"] == 2
    assert summary["rows"][0]["eligible"] == 2
