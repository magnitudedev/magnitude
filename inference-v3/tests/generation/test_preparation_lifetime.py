"""Cancelled image preparation retains ownership until the host worker finishes."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from engine.serving.requests import ChatRequest
from engine.serving.session import ChatService


def request(image=True):
    return ChatRequest.model_validate({
        "model": "test",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        ] if image else "hello"}],
    })


def test_cancelled_image_preparation_closes_its_late_result():
    async def run():
        loop = asyncio.get_running_loop()
        started, closed = asyncio.Event(), asyncio.Event()
        release = threading.Event()
        closes = []

        def close():
            closes.append(True)
            loop.call_soon_threadsafe(closed.set)

        def prepare(body):
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5)
            return SimpleNamespace(close=close)

        service = SimpleNamespace(prepare=prepare)
        task = asyncio.create_task(ChatService.prepare_async(service, request()))
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not closes
        finally:
            release.set()
        await asyncio.wait_for(closed.wait(), 5)
        assert closes == [True]

    asyncio.run(run())


def test_successful_preparation_transfers_ownership_and_text_stays_inline():
    async def run():
        owner = threading.get_ident()
        threads, closes = [], []
        prompt = SimpleNamespace(close=lambda: closes.append(True))

        def prepare(body):
            threads.append(threading.get_ident())
            return prompt

        service = SimpleNamespace(prepare=prepare)
        assert await ChatService.prepare_async(service, request(False)) is prompt
        assert threads[-1] == owner
        assert await ChatService.prepare_async(service, request()) is prompt
        assert threads[-1] != owner
        assert not closes
        prompt.close()
        assert closes == [True]

    asyncio.run(run())


def test_cancelled_handoff_closes_an_already_prepared_prompt(monkeypatch):
    closes = []
    prompt = SimpleNamespace(close=lambda: closes.append(True))

    async def complete_then_cancel(build):
        build()
        raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "to_thread", complete_then_cancel)
    service = SimpleNamespace(prepare=lambda body: prompt)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(ChatService.prepare_async(service, request()))
    assert closes == [True]
