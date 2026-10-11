"""Fitting fixtures to context lengths with one engine's own rendering.

``Fixture.sized`` asks for one rendered length at a time. Rendering is a process call (the
harness, or a comparison engine's own template path), so the sizing of every fixture runs
in lockstep: each sizing waits on its own thread, and the lengths all pending sizings ask
for are rendered together in one batch.
"""

import asyncio
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor

from magnitude_fixtures import Conversation, Fixture
from magnitude_fixtures.canonical import encoded

Message = dict
#: Rendered lengths of chat bodies (``{"messages", "tools"}``), in order.
Lengths = Callable[[list[dict]], Awaitable[list[int]]]


def chat(messages: list[Message], tools: list[Message]) -> dict:
    """The chat body a conversation is rendered as: served with thinking disabled."""
    return {
        "messages": messages,
        **({"tools": tools} if tools else {}),
        "chat_template_kwargs": {"enable_thinking": False},
    }


async def size(targets: list[tuple[Fixture, int]], lengths: Lengths) -> list[Conversation]:
    """Each fixture sized to its target by the contract rule (the most history at or under
    the target), in order; renders are batched and never repeated. A fixture whose whole
    history fits under its target is refused rather than served short."""
    loop = asyncio.get_running_loop()
    known: dict[str, int] = {}
    pending: list[tuple[str, dict, asyncio.Future[int]]] = []
    live = len(targets)

    async def flush() -> None:
        """Render once every live sizing waits; each waiter receives its length or the error."""
        if not pending or len(pending) != live:
            return
        batch = pending[:]
        pending.clear()
        unique = {key: body for key, body, _ in batch}
        try:
            rendered = await lengths(list(unique.values()))
            if len(rendered) != len(unique):
                raise ValueError("renderer returned a different number of lengths")
        except Exception as error:
            for _, _, future in batch:
                future.set_exception(error)
            return
        known.update(zip(unique, rendered, strict=True))
        for key, _, future in batch:
            future.set_result(known[key])

    async def submit(body: dict) -> int:
        key = encoded(body)
        if key in known:
            return known[key]
        future: asyncio.Future[int] = loop.create_future()
        pending.append((key, body, future))
        await flush()
        return await future

    def rendered_length(messages: list[Message], tools: list[Message]) -> int:
        return asyncio.run_coroutine_threadsafe(submit(chat(messages, tools)), loop).result()

    # One thread per sizing: a sizing that could not start would stall the lockstep.
    threads = ThreadPoolExecutor(max_workers=max(1, len(targets)), thread_name_prefix="sizing")

    async def one(fixture: Fixture, target: int) -> Conversation:
        nonlocal live
        try:
            conversation = await loop.run_in_executor(
                threads, fixture.sized, target, rendered_length
            )
        finally:
            live -= 1
            await flush()
        if len(conversation.messages) == len(fixture.messages):
            # Every unit fits: the context would be measured shorter than it is named.
            raise ValueError(f"{fixture.id} holds too little history to fill {target} tokens")
        return conversation

    try:
        return list(await asyncio.gather(*(one(fixture, target) for fixture, target in targets)))
    finally:
        threads.shutdown(wait=False, cancel_futures=True)
