"""Offline tokenizer/template/language qualification before benchmark measurement."""

import argparse
import json
from pathlib import Path

import llguidance as llg
import llguidance.hf

from magnitude_engine.artifacts.tokenizer import TokenizerArtifact

from .parsing import OutputParser, ToolCall
from .template import ChatTemplate


def qualify(directory: Path) -> dict:
    artifact = TokenizerArtifact.load(directory)
    description = "Emit the supplied integer unchanged."
    tools = [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    prompt = ChatTemplate(artifact).render(
        [{"role": "user", "content": "Call echo with value 7."}],
        tools=tools,
        tool_choice="required",
        chat_template_kwargs={"enable_thinking": False},
    )
    if prompt.constraint is None or prompt.format is None:
        raise ValueError("benchmark requires an enforceable tool-call wire")
    if description not in prompt.text:
        raise ValueError("checkpoint template did not render the supplied tool definition")
    table = llguidance.hf.from_tokenizer(
        artifact.tokenizer, n_vocab=artifact.vocabulary, eos_token=list(artifact.eos_tokens)
    )
    matcher = llg.LLMatcher(
        table, llg.LLMatcher.grammar_from_lark(prompt.constraint.lark), log_level=0
    )
    if matcher.is_error():
        raise ValueError(matcher.get_error())
    body = {
        "json": '{"name":"echo","arguments":{"value":7}}',
        "xml": "<function=echo><parameter=value>7</parameter></function>",
        "gemma": "call:echo{value:7}",
    }[prompt.format.arguments]
    sample = prompt.format.call_open + body + prompt.format.call_close
    if (
        not matcher.consume_tokens(table.tokenize_str(sample, parse_special=True))
        or not matcher.is_accepting()
    ):
        raise ValueError("tool grammar rejected its declared wire")
    parser = OutputParser(prompt.format, tools)
    if parser.feed(sample, final=True) != [ToolCall(0, "echo", {"value": 7})]:
        raise ValueError("tool parser disagrees with the declared wire")
    return {
        "tokenizerIdentity": artifact.identity,
        "family": artifact.family,
        "vocabulary": artifact.vocabulary,
        "promptTokens": len(prompt.tokens),
        "toolsRendered": True,
        "toolGrammarValidated": True,
        "toolCallParsed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(qualify(args.target)))


if __name__ == "__main__":
    main()
