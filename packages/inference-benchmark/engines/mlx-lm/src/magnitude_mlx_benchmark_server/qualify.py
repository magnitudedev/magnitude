from __future__ import annotations

import argparse
import json

from mlx_lm.utils import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualify an MLX-LM tokenizer for the benchmark"
    )
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.model)
    if not tokenizer.has_chat_template:
        raise RuntimeError("tokenizer has no chat template")
    if not tokenizer.has_tool_calling:
        raise RuntimeError("tokenizer has no MLX-LM tool-call parser")
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Call the test tool."}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "Qualification tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    print(
        json.dumps(
            {
                "chat_template": True,
                "tool_calling": True,
                "tool_call_start": tokenizer.tool_call_start,
                "tool_call_end": tokenizer.tool_call_end,
                "rendered_tokens": len(tokens),
            },
            sort_keys=True,
        )
    )
