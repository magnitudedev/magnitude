"""Reference tokenization and chat-template renders for the chat layer.

For each tokenizer profile the release catalog uses, this writes under
``results/tokenizers/``:

- ``<name>.gguf``: a vocabulary-only GGUF, the locked file's metadata with no
  tensors (split keys dropped), read from the header bytes by range request;
- ``<name>.json``: the model's own reference tokenizer ids for a fixed corpus
  (HF ``tokenizers`` from the original repository's ``tokenizer.json``), and
  the reference Jinja renders (``transformers`` ``apply_chat_template``) of the
  GGUF's chat template for representative conversations.

The reference is independent of the engine. Consume it with::

    MAGNITUDE_TEST_GGUF=results/tokenizers/<name>.gguf \
    MAGNITUDE_TEST_REFERENCE=results/tokenizers/<name>.json \
    cargo test -p magnitude-chat --test real_tokenizer -- --ignored
"""

from __future__ import annotations

import argparse
import json
import struct
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download

import audit_catalog_tensor_types as audit

RESULTS = Path(__file__).resolve().parent / "results" / "tokenizers"

# name: (GGUF repository, locked revision, file, reference repository, revision)
MODELS = {
    "qwen3.5-4b": (
        "unsloth/Qwen3.5-4B-MTP-GGUF", "86835bf9949e4d14d6860f7910b1340ad4f271a9",
        "Qwen3.5-4B-Q4_0.gguf", "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    ),
    "gemma-4-e2b": (
        "unsloth/gemma-4-E2B-it-qat-GGUF", "66a399f68ddd113b06dff02fca9523e55465d11d",
        "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf", "google/gemma-4-E2B-it",
        "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    ),
    "gemma-4-12b": (
        "unsloth/gemma-4-12B-it-qat-GGUF", "980b060c40a8539ac159e0501a3e0f66a6365af3",
        "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf", "google/gemma-4-12B-it",
        "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
    ),
    "gemma-4-31b": (
        "unsloth/gemma-4-31B-it-qat-GGUF", "43cc1aeb31adf47ec06a854507ce552cd9862e6f",
        "gemma-4-31B-it-qat-UD-Q4_K_XL.gguf", "google/gemma-4-31B-it",
        "842da3794eaa0b77d5f08bae87a17459d91ff475",
    ),
    "lfm2.5-2.6b": (
        "LiquidAI/LFM2.5-2.6B-GGUF", "b421ad1d549afeda6a0fb2ad3a697cb5a7879adc",
        "LFM2.5-2.6B-Q4_0.gguf", "LiquidAI/LFM2.5-2.6B", "654f9463ce32b05d0429d76fe1f580b27d4c1ac0",
    ),
    "lfm2.5-8b-a1b": (
        "LiquidAI/LFM2.5-8B-A1B-GGUF", "dfd5fdcad7a1c0d31473fb4ca443b8befbacddf0",
        "LFM2.5-8B-A1B-Q4_0.gguf", "LiquidAI/LFM2.5-8B-A1B",
        "5dd22602c2e9f6a097b1de4c4efe0658b605015c",
    ),
    "minicpm5-2b": (
        "openbmb/MiniCPM5-2B-GGUF", "2079a22f3beaa4e306449978533478fe0522f4b3",
        "MiniCPM5-2B-Q4_K_M.gguf", "openbmb/MiniCPM5-2B", "12a3808a956f869c767195e9266b59c4d21d92e2",
    ),
    "muse-glimmer-30b": (
        "unsloth/Muse-Glimmer-30B-GGUF", "1afeb8e879f60116d206cf724425dbe1e1a2f7f5",
        "Muse-Glimmer-30B-UD-Q4_K_XL.gguf", "meta-models/Muse-Glimmer-30B",
        "a4e59da52a7bc87ae7251dd5545c0dd437c44b68",
    ),
    "nemotron-3.5-lightning": (
        "ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF",
        "88d7ce0b0fa385c5108866ce5d33690927531a37",
        "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_K_M.gguf",
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        "a9904d24bcc1d289a1950fa9d2b978c47cf903b9",
    ),
    "nemotron-3-super": (
        "unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF", "036038fb30334a2d56a146c6f0d4871ab5edccbb",
        "UD-IQ4_NL/NVIDIA-Nemotron-3-Super-120B-A12B-UD-IQ4_NL-00001-of-00003.gguf",
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "2dc98e2afe4face0e4ce40972a915c45368bd34a",
    ),
}

CORPUS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog. It's a test; they'll see, we've done, "
    "I'd go, you're right, I'M LOUD, DON'T.",
    "  leading spaces and trailing   ",
    "multiple    internal     spaces\tand\ttabs\t\t\tend",
    "line one\nline two\n\nline four\n\n\n\nafter blank lines\n",
    "windows\r\nline\r\nendings\r\n\r\ndouble crlf and lone \r carriage",
    "Numbers: 1 12 123 1234 12345 123456 3.14159 -42 1,000,000 0x1F 2026-09-27 v1.2.3",
    "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n\n"
    "print([fib(i) for i in range(10)])\n",
    "fn main() {\n\tlet x: Vec<u32> = (0..10).map(|i| i * i).collect();\n\tprintln!(\"{:?}\", x);\n}\n",
    "{\"key\": \"value\", \"list\": [1, 2, 3], \"nested\": {\"a\": null, \"b\": true}}",
    "<div class=\"x\">&amp; &lt;tag&gt;</div> // comment /* block */ #!/usr/bin/env bash",
    "日本語のテキストと中文文本，还有한국어 텍스트。",
    "Emoji: 😀👍🏽👨‍👩‍👧‍👦 🇳🇴 ❤️ ✅",
    "Combining: é ä ñ — café naïve résumé Ångström",
    "Arabic: مرحبا بالعالم. Hebrew: שלום עולם. Hindi: नमस्ते दुनिया। Thai: สวัสดีชาวโลก",
    "Cyrillic: Привет, мир! Greek: Γειά σου Κόσμε. Math: ∑ᵢ xᵢ² ≤ ∞ ∫₀¹ f(x) dx ≠ π",
    "CamelCaseIdentifier snake_case_name SCREAMING_CASE kebab-case iPhone McDonald's",
    "URL: https://example.com/path/to/page?query=1&other=two#frag and email a.b@c.io",
    "Mixed123Numbers456 and a1b2c3 and 99bottles",
    "\n\n\n",
    "   ",
    " ",
    "a",
    "tab\tseparated\tvalues\n1\t2\t3\n",
    "Quotes: “smart” ‘single’ «guillemets» „German“",
    "Zero​width and non breaking and ideographic　space",
    "Repeated punctuation!!! ??? ... --- === *** ### @@@ $$$ %%% ^^^ &&& ((( )))",
    "ALLCAPS WORDS AND lowercase words And Title Case Words",
    "   indented\n      more indented\n\t\ttabbed\n",
    "\u0000\u0001\u007f bytes ÿĀ \U0001f600\U000e0001 end",
    # Special-token text of several vocabularies; each tokenizer recognizes its own.
    "<|im_start|>user\nhi<|im_end|>\n<s> </s> <bos> <eos> <|endoftext|> <think>x</think> "
    "<|turn>model\n<turn|> 〈|EOS|〉 <|start|>assistant<|message|>ok<|eot|>",
]

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "days": {"type": "integer"},
            },
            "required": ["city"],
        },
    },
}

CALL = {
    "type": "function",
    "function": {"name": "get_weather", "arguments": {"city": "Paris", "days": 2}},
}

# (name, messages, tools); every render adds the generation prompt.
CONVERSATIONS = [
    ("single", [{"role": "user", "content": "Hi there! What's 2+2?"}], None),
    (
        "multi_turn",
        [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Name a colour."},
            {"role": "assistant", "content": "Blue."},
            {"role": "user", "content": "Another one, with some 日本語."},
        ],
        None,
    ),
    ("tools", [{"role": "user", "content": "What is the weather in Paris?"}], [WEATHER]),
    (
        "tool_round_trip",
        [
            {"role": "user", "content": "What is the weather in Paris?"},
            {"role": "assistant", "content": "", "tool_calls": [dict(CALL, id="call_1")]},
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather",
             "content": "{\"temperature\": 21, \"sky\": \"clear\"}"},
        ],
        [WEATHER],
    ),
]

FIXED_NOW = datetime(2026, 9, 27, 12, 0, 0, tzinfo=timezone.utc)


def header_bytes(repository: str, revision: str, path: str) -> bytes:
    url = audit.header_url(repository, revision, path)
    size = 1 << 20
    while size <= 256 << 20:
        request = urllib.request.Request(url, headers={"Range": f"bytes=0-{size - 1}"})
        with urllib.request.urlopen(request, timeout=300) as response:
            if response.headers.get("x-repo-commit") not in (None, revision):
                raise ValueError(f"{repository} resolved to another commit than {revision}")
            data = response.read(size + 1)
        try:
            audit.parse_header(data)
            return data
        except audit.NeedMore:
            size *= 2
    raise ValueError("GGUF header exceeds 256 MiB")


SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def vocabulary_only(data: bytes) -> bytes:
    """The container's metadata, without split keys, and no tensors."""

    def skip(kind: int, offset: int) -> int:
        if kind in SCALAR_SIZES:
            return offset + SCALAR_SIZES[kind]
        if kind == 8:
            return offset + 8 + struct.unpack_from("<Q", data, offset)[0]
        if kind == 9:
            element, count = struct.unpack_from("<IQ", data, offset)
            offset += 12
            if element in SCALAR_SIZES:
                return offset + SCALAR_SIZES[element] * count
            for _ in range(count):
                offset = skip(element, offset)
            return offset
        raise ValueError(f"unsupported GGUF metadata type {kind}")

    if data[:4] != b"GGUF":
        raise ValueError("not a little-endian GGUF container")
    _, metadata_count = struct.unpack_from("<QQ", data, 8)
    offset = 24
    kept = []
    for _ in range(metadata_count):
        start = offset
        length = struct.unpack_from("<Q", data, offset)[0]
        name = data[offset + 8 : offset + 8 + length].decode()
        offset = skip(struct.unpack_from("<I", data, offset + 8 + length)[0], offset + 12 + length)
        if not name.startswith("split."):
            kept.append(data[start:offset])
    out = data[:8] + struct.pack("<QQ", 0, len(kept)) + b"".join(kept)
    return out + b"\0" * (-len(out) % 32)


def chat_template(data: bytes) -> str:
    marker = b"tokenizer.chat_template"
    index = data.index(struct.pack("<Q", len(marker)) + marker)
    offset = index + 8 + len(marker)
    assert struct.unpack_from("<I", data, offset)[0] == 8
    length = struct.unpack_from("<Q", data, offset + 4)[0]
    return data[offset + 12 : offset + 12 + length].decode()


def generate(name: str) -> None:
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    from transformers.utils import chat_template_utils

    # Templates read the date through `strftime_now`; the engine renders it
    # from the request's time in UTC.
    class FixedClock:
        @staticmethod
        def now(tz=None):
            return FIXED_NOW

    chat_template_utils.datetime = FixedClock

    repository, revision, path, reference, reference_revision = MODELS[name]
    data = header_bytes(repository, revision, path)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.gguf").write_bytes(vocabulary_only(data))
    tokenizer = Tokenizer.from_file(
        hf_hub_download(reference, "tokenizer.json", revision=reference_revision)
    )
    corpus = []
    for text in CORPUS:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        corpus.append(
            {"text": text, "ids": ids, "decoded": tokenizer.decode(ids, skip_special_tokens=False)}
        )
    template = chat_template(data)
    renderer = AutoTokenizer.from_pretrained(
        reference, revision=reference_revision, trust_remote_code=False
    )
    renders = []
    for conversation, messages, tools in CONVERSATIONS:
        text = renderer.apply_chat_template(
            messages,
            tools=tools,
            chat_template=template,
            add_generation_prompt=True,
            tokenize=False,
        )
        renders.append(
            {
                "name": conversation,
                "messages": messages,
                "tools": tools or [],
                "text": text,
                "ids": tokenizer.encode(text, add_special_tokens=False).ids,
            }
        )
    reference_record = {
        "gguf": {"repository": repository, "revision": revision, "path": path},
        "reference": {"repository": reference, "revision": reference_revision},
        "now": int(FIXED_NOW.timestamp()),
        "corpus": corpus,
        "renders": renders,
    }
    (RESULTS / f"{name}.json").write_text(json.dumps(reference_record, ensure_ascii=False, indent=1))
    print(name, len(corpus), "texts,", len(renders), "renders")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help=f"any of {', '.join(MODELS)}; default all")
    arguments = parser.parse_args()
    unknown = sorted(set(arguments.names) - set(MODELS))
    if unknown:
        parser.error(f"unknown names: {', '.join(unknown)}")
    for name in arguments.names or MODELS:
        generate(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
