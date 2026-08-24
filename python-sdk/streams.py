"""Typed sync/async SSE helpers copied into the generated OpenAPI package."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable, Iterator

from .client import AuthenticatedClient, Client
from .models.chat_completion_chunk import ChatCompletionChunk
from .models.chat_completion_request import ChatCompletionRequest
from .models.inference_resource_invalidation import InferenceResourceInvalidation
from .models.inference_resource_topic import InferenceResourceTopic
from .models.response_create_request import ResponseCreateRequest
from .models.response_stream_event import ResponseStreamEvent

ClientType = AuthenticatedClient | Client


def _events(lines: Iterable[str]) -> Iterator[str]:
    data: list[str] = []
    for line in lines:
        if line == "":
            if data:
                yield "\n".join(data)
                data = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        yield "\n".join(data)


async def _async_events(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    data: list[str] = []
    async for line in lines:
        if line == "":
            if data:
                yield "\n".join(data)
                data = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        yield "\n".join(data)


def stream_chat_completions(
    client: ClientType,
    body: ChatCompletionRequest,
    *,
    include_progress: bool = False,
) -> Iterator[ChatCompletionChunk]:
    headers = {"Magnitude-Include-Progress": "true"} if include_progress else {}
    with client.get_httpx_client().stream(
        "POST", "/v1/chat/completions", json=body.to_dict(), headers=headers
    ) as response:
        response.raise_for_status()
        for data in _events(response.iter_lines()):
            if data == "[DONE]":
                return
            yield ChatCompletionChunk.from_dict(json.loads(data))


async def stream_chat_completions_async(
    client: ClientType,
    body: ChatCompletionRequest,
    *,
    include_progress: bool = False,
) -> AsyncIterator[ChatCompletionChunk]:
    headers = {"Magnitude-Include-Progress": "true"} if include_progress else {}
    async with client.get_async_httpx_client().stream(
        "POST", "/v1/chat/completions", json=body.to_dict(), headers=headers
    ) as response:
        response.raise_for_status()
        async for data in _async_events(response.aiter_lines()):
            if data == "[DONE]":
                return
            yield ChatCompletionChunk.from_dict(json.loads(data))


def stream_responses(
    client: ClientType,
    body: ResponseCreateRequest,
    *,
    include_progress: bool = False,
) -> Iterator[ResponseStreamEvent]:
    headers = {"Magnitude-Include-Progress": "true"} if include_progress else {}
    with client.get_httpx_client().stream(
        "POST", "/v1/responses", json=body.to_dict(), headers=headers
    ) as response:
        response.raise_for_status()
        for data in _events(response.iter_lines()):
            yield ResponseStreamEvent.from_dict(json.loads(data))


async def stream_responses_async(
    client: ClientType,
    body: ResponseCreateRequest,
    *,
    include_progress: bool = False,
) -> AsyncIterator[ResponseStreamEvent]:
    headers = {"Magnitude-Include-Progress": "true"} if include_progress else {}
    async with client.get_async_httpx_client().stream(
        "POST", "/v1/responses", json=body.to_dict(), headers=headers
    ) as response:
        response.raise_for_status()
        async for data in _async_events(response.aiter_lines()):
            yield ResponseStreamEvent.from_dict(json.loads(data))


def watch_inference_events(
    client: ClientType,
    *,
    topics: Iterable[InferenceResourceTopic] = (),
) -> Iterator[InferenceResourceInvalidation]:
    params = {"topics": ",".join(topics)} if topics else None
    with client.get_httpx_client().stream(
        "GET", "/api/v1/events", params=params
    ) as response:
        response.raise_for_status()
        for data in _events(response.iter_lines()):
            yield InferenceResourceInvalidation.from_dict(json.loads(data))


async def watch_inference_events_async(
    client: ClientType,
    *,
    topics: Iterable[InferenceResourceTopic] = (),
) -> AsyncIterator[InferenceResourceInvalidation]:
    params = {"topics": ",".join(topics)} if topics else None
    async with client.get_async_httpx_client().stream(
        "GET", "/api/v1/events", params=params
    ) as response:
        response.raise_for_status()
        async for data in _async_events(response.aiter_lines()):
            yield InferenceResourceInvalidation.from_dict(json.loads(data))
