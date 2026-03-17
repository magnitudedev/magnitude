from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx_lm
import outlines
from fastapi import FastAPI
from pydantic import BaseModel
from outlines.types import CFG


DEFAULT_MODEL = "NexVeridian/OmniCoder-9B-6bit"

app = FastAPI(title="xml-act-grammar")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = 500


def build_prompt(messages: list[ChatMessage]) -> str:
    system_prompt = """You are an AI coding agent that must respond in xml-act format.

Format:
- Optionally start with <lenses> containing <lens name="...">...</lens>
- Optionally include <comms> containing <message to="...">...</message>
- Optionally include <actions> containing XML tool calls and optional <inspect></inspect>
- End with exactly one turn-control tag: <next/> or <yield/>

Be concise and valid. Do not include markdown fences.
"""
    rendered_messages = "\n".join(
        f"<{message.role}>\n{message.content}\n</{message.role}>"
        for message in messages
    )
    return f"<system>\n{system_prompt}\n</system>\n{rendered_messages}"


def load_model_and_grammar(model_name: str):
    model = outlines.from_mlxlm(*mlx_lm.load(model_name))
    grammar_string = Path(__file__).with_name("grammar.lark").read_text()
    cfg = CFG(grammar_string)
    return model, cfg


@app.on_event("startup")
def startup_event() -> None:
    model, cfg = load_model_and_grammar(DEFAULT_MODEL)
    app.state.model = model
    app.state.cfg = cfg
    app.state.model_name = DEFAULT_MODEL


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
    model_name = request.model or DEFAULT_MODEL

    if app.state.model_name != model_name:
        model, cfg = load_model_and_grammar(model_name)
        app.state.model = model
        app.state.cfg = cfg
        app.state.model_name = model_name

    prompt = build_prompt(request.messages)
    output = app.state.model(prompt, app.state.cfg, max_tokens=request.max_tokens)

    return {
        "id": "chatcmpl-xml-act-grammar",
        "object": "chat.completion",
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output,
                },
                "finish_reason": "stop",
            }
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve:app", host="127.0.0.1", port=8000, reload=False)