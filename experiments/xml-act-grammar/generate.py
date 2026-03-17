from __future__ import annotations

import argparse
from pathlib import Path

import mlx_lm
import outlines
from outlines.types import CFG


DEFAULT_MODEL = "NexVeridian/OmniCoder-9B-6bit"


def load_model_and_grammar(model_name: str):
    model = outlines.from_mlxlm(*mlx_lm.load(model_name))
    grammar_path = Path(__file__).with_name("grammar.lark")
    grammar_string = grammar_path.read_text()
    cfg = CFG(grammar_string)
    return model, cfg


def build_prompt(user_prompt: str) -> str:
    system_prompt = """You are an AI coding agent that must respond in xml-act format.

Format:
- Optionally start with <lenses> containing <lens name="...">...</lens>
- Optionally include <comms> containing <message to="...">...</message>
- Optionally include <actions> containing XML tool calls and optional <inspect></inspect>
- End with exactly one turn-control tag: <next/> or <yield/>

Be concise and valid. Do not include markdown fences.
"""
    return f"<system>\n{system_prompt}\n</system>\n<user>\n{user_prompt}\n</user>"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate grammar-constrained xml-act output.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="MLX model name to load")
    parser.add_argument("--max-tokens", type=int, default=500, help="Maximum tokens to generate")
    parser.add_argument(
        "--prompt",
        default="Hello, what can you help me with?",
        help="User prompt to send to the model",
    )
    args = parser.parse_args()

    model, cfg = load_model_and_grammar(args.model)

    prompt = build_prompt(args.prompt)
    result = model(prompt, cfg, max_tokens=args.max_tokens)

    print(result)


if __name__ == "__main__":
    main()