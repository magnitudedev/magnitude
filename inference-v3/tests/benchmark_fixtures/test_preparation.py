import hashlib
import json

import httpx
import pytest

from benchmark_fixtures import preparation, prose
from benchmark_fixtures.contexts import Context, History
from benchmark_fixtures.interactions import ExpectedCall, Interaction
from benchmark_fixtures.preparation import Fixture
from benchmark_fixtures.storage import fetch


async def test_source_is_verified_before_caching_and_cached_offline(tmp_path):
    content = b"pinned book"
    checksum = hashlib.sha256(content).hexdigest()
    path = tmp_path / "book"
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, content=content)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        assert await fetch(client, "https://fixture.test/book", checksum, path) == content
        assert await fetch(client, "https://fixture.test/book", checksum, path) == content
        assert len(calls) == 1
        with pytest.raises(ValueError, match="checksum mismatch"):
            await fetch(client, "https://fixture.test/book", "bad hash", path)
        assert path.read_bytes() == content


def test_prose_normalization_retains_body_only():
    source = (
        "\ufefflicense\r\n*** START OF THE PROJECT GUTENBERG EBOOK MOBY DICK ***\r\n"
        "\r\nTitle\r\nCall me Ishmael.\r\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK MOBY DICK ***\r\nlicense"
    )
    assert prose.normalize(source.encode()) == "Title\nCall me Ishmael."
    with pytest.raises(ValueError, match="markers"):
        prose.normalize(b"no markers")


async def test_contiguous_64k_window_cache_and_provenance(monkeypatch, tmp_path):
    class Tokenizer:
        identity = "tokenizer-a"
        calls = 0

        def encode(self, text):
            self.calls += 1
            return tuple(range(70000))

    async def source(**kwargs):
        return "book", {"sha256": "source"}

    monkeypatch.setattr(prose, "prepare", source)
    tokenizer = Tokenizer()
    fixture = Fixture(
        identity="prose.moby-dick", context_tokens=65536, continuation_tokens=16, offset=10
    )
    first = await preparation.prepare(fixture, tokenizer, cache=tmp_path)
    second = await preparation.prepare(fixture, tokenizer, cache=tmp_path)
    assert first == second and tokenizer.calls == 1
    assert first.prompt == tuple(range(10, 65546))
    assert first.continuation == tuple(range(65546, 65562))
    assert first.provenance["actual_context_tokens"] == 65536
    assert len(list((tmp_path / "prepared").glob("*.json"))) == 1
    token_file = next((tmp_path / "tokens").glob("*.json"))
    record = json.loads(token_file.read_text())
    record["tokens"][0] = -1
    token_file.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="digest"):
        await preparation.prepare(fixture, tokenizer, cache=tmp_path)
    tokenizer.identity = "tokenizer-b"
    await preparation.prepare(fixture, tokenizer, cache=tmp_path)
    assert tokenizer.calls == 2
    with pytest.raises(ValueError, match="window ends"):
        await preparation.prepare(
            fixture.model_copy(update={"offset": 5000}), tokenizer, cache=tmp_path
        )


def interaction():
    return Interaction(
        id="echo",
        category="simple-python",
        messages=[{"role": "user", "content": "Echo 7"}],
        tools=[{"type": "function", "function": {"name": "echo"}}],
        expected=[ExpectedCall(name="echo", arguments={"value": [7]})],
        provenance={},
    )


async def test_history_reaches_target_at_first_complete_round():
    item = interaction()
    history = History([item], "test", item)
    counts = []

    async def counter(context):
        counts.append(len(context.messages))
        return len(context.messages) * 7

    prepared = await history.prepare(65536, counter, "test-tokenizer")
    assert prepared.tokens >= 65536
    assert prepared.tokens - 3 * 7 < 65536
    assert len(counts) < 30
    messages = prepared.content.messages
    calls = [c["id"] for m in messages for c in m.get("tool_calls", [])]
    replies = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
    assert calls == replies and len(set(calls)) == len(calls)
    assert messages[-1] == item.messages[0]
    history.complete()
    grown = await history.prepare(70000, counter, "test-tokenizer")
    assert grown.content.messages[: len(messages)] == messages


async def test_history_rejects_non_growing_renderer():
    item = interaction()

    async def constant(context):
        return 1

    with pytest.raises(ValueError, match="did not grow"):
        await History([item], "test", item).prepare(100, constant, "broken")


async def test_session_plan_uses_shared_history_and_counts():
    from session_bench.suites import compile_plan

    item = interaction()

    async def counter(context):
        return len(context.messages) * 7

    plan = await compile_plan(
        [item], "corpus", ("context",), (1024,), counter=counter, sizing_identity="test-tokenizer"
    )
    request = plan.requests[0]
    shared = await History([item], request.session, item).prepare(1024, counter, "test-tokenizer")
    assert Context(messages=request.messages, tools=request.tools) == shared.content
    assert request.fixture_provenance == shared.provenance
    assert plan.warmup.fixture_provenance["recipe"] == "qualification-decision-v1"
