import httpx

from magnitude_inference.client import Client
from magnitude_inference.models.chat_completion_request import ChatCompletionRequest
from magnitude_inference.models.inference_resource_topic import InferenceResourceTopic
from magnitude_inference.streams import stream_chat_completions, watch_inference_events


def test_chat_stream_decodes_chunks_and_progress_header() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["Magnitude-Include-Progress"] == "true"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=(
                'data: {"id":"chatcmpl-1","object":"chat.completion.chunk",'
                '"created":1,"model":"model:q4","choices":[]}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    http = httpx.Client(
        base_url="http://127.0.0.1:10100/inference",
        transport=httpx.MockTransport(respond),
    )
    client = Client(base_url="http://127.0.0.1:10100/inference").set_httpx_client(http)
    body = ChatCompletionRequest.from_dict({
        "model": "model:q4",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    })
    chunks = list(stream_chat_completions(client, body, include_progress=True))
    assert len(chunks) == 1
    assert chunks[0].model == "model:q4"


def test_native_event_stream_decodes_invalidations_and_topics() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.params["topics"] == "models,instances"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text='event: invalidation\ndata: {"topic":"models","revision":4}\n\n',
        )

    http = httpx.Client(
        base_url="http://127.0.0.1:10100/inference",
        transport=httpx.MockTransport(respond),
    )
    client = Client(base_url="http://127.0.0.1:10100/inference").set_httpx_client(http)
    events = list(watch_inference_events(client, topics=[
        InferenceResourceTopic.MODELS,
        InferenceResourceTopic.INSTANCES,
    ]))
    assert [(event.topic, event.revision) for event in events] == [("models", 4)]
