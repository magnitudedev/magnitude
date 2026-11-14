"""Reference tokenization and chat-template renders for the chat layer.

For each model this writes under ``--output`` (default ``results/tokenizers/``):

- ``<name>.gguf.gz``: a vocabulary-only GGUF, the tokenizer metadata (and
  architecture) of the model's GGUF with no tensors;
- ``<name>.json``: the model's own reference tokenizer ids for a fixed corpus
  (HF ``tokenizers`` from the original repository's ``tokenizer.json``), and
  the reference Jinja renders (``transformers`` ``apply_chat_template``) of the
  GGUF's chat template for representative conversations, and, for the models
  in ``REPLYING``, assistant replies as the template writes them after the
  generation prompt.

A published model's vocabulary is its locked GGUF's header, read by range
request. A converted model's is what llama.cpp's converter writes from the
checkpoint's tokenizer files (its ``set_vocab``), read from the source tree
``--llama-cpp`` names (v0.9.0, ``cd959bae``, the revision our converter builds
on); no weights are read.

The reference is independent of the engine. Run it in the evaluation workspace
(``uv run --project inference/evaluation --package magnitude-references
--extra check python -I tokenizer_reference.py``). Consume it with::

    MAGNITUDE_TEST_GGUF=results/tokenizers/<name>.gguf.gz \
    MAGNITUDE_TEST_REFERENCE=results/tokenizers/<name>.json \
    cargo test -p magnitude-chat --test real_tokenizer -- --ignored

GLM-5.3-Flash's (``glm53``) is committed in ``tests/assets/`` and always runs.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download
from magnitude_references.gguf import DEFAULT_ALIGNMENT, LocalSource, RemoteSource, Value, encode_header

RESULTS = Path(__file__).resolve().parent / "results" / "tokenizers"

# name: (GGUF repository, locked revision, file, reference repository, revision)
PUBLISHED = {
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

# name: (checkpoint repository, revision, the converter's model class as `module.Class`)
CONVERTED = {
    "glm53": (
        "zai-org/GLM-5.3-Flash", "46a050bec13002ad29ee795c16e61cf4fa3c4a9c", "glm.Glm5NextModel",
    ),
}

# The checkpoint files a converter's `set_vocab` reads.
TOKENIZER_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")

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
    "Combining: é ä ñ — café naïve résumé Ångström",
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
    "Zero​width and non breaking and ideographic　space",
    "Repeated punctuation!!! ??? ... --- === *** ### @@@ $$$ %%% ^^^ &&& ((( )))",
    "ALLCAPS WORDS AND lowercase words And Title Case Words",
    "   indented\n      more indented\n\t\ttabbed\n",
    "\u0000\u0001\u007f bytes ÿĀ \U0001f600\U000e0001 end",
    # Contractions whose letters case-fold beyond ASCII (long s, Kelvin sign).
    "it'ſ, it'S, we'LL, K'd, 'Ve, 'ﬅ",
    # Special-token text of several vocabularies; each tokenizer recognizes its own.
    "<|im_start|>user\nhi<|im_end|>\n<s> </s> <bos> <eos> <|endoftext|> <think>x</think> "
    "<|turn>model\n<turn|> 〈|EOS|〉 <|start|>assistant<|message|>ok<|eot|>",
    "[gMASK]<sop><|system|>s<|user|>u<|assistant|><think>r</think>a<tool_call>f<arg_key>k</arg_key>"
    "<arg_value>v</arg_value></tool_call><|observation|><tool_response>x</tool_response>/nothink"
    "<|begin_of_image|><|image|><|end_of_image|>",
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

SEARCH = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web «verbatim»",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
        },
    },
}


def call(identity: str, name: str, arguments: dict) -> dict:
    return {"id": identity, "type": "function", "function": {"name": name, "arguments": arguments}}


def result(identity: str, name: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": identity, "name": name, "content": content}


ASK = {"role": "user", "content": "What is the weather in Paris?"}
CALLED = {"role": "assistant", "content": "", "reasoning_content": "One call for Paris.",
          "tool_calls": [call("call_1", "get_weather", {"city": "Paris", "days": 2})]}
ANSWERED = result("call_1", "get_weather", "{\"temperature\": 21, \"sky\": \"clear\"}")
REASONED = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Name a colour."},
    {"role": "assistant", "content": "Blue.", "reasoning_content": "Any colour will do."},
    {"role": "user", "content": "Another one."},
    {"role": "assistant", "content": "Red."},
    {"role": "user", "content": "And one more, with some 日本語."},
]
# Two calls answered out of call order.
PARALLEL = [
    {"role": "user", "content": "Weather in Paris, and search for llama.cpp."},
    {"role": "assistant", "content": "Both at once.",
     "tool_calls": [call("call_b", "get_weather", {"city": "Paris"}),
                    call("call_A", "search", {"query": "llama.cpp \"chat\"\nparser", "limit": 3})]},
    result("call_A", "search", "[{\"title\": \"llama.cpp\"}]"),
    result("call_b", "get_weather", "sunny"),
]

# (name, messages, tools, the request's template arguments); every render adds the generation prompt.
# A template reads the arguments it names and ignores the others.
CONVERSATIONS = [
    ("single", [{"role": "user", "content": "Hi there! What's 2+2?"}], None, {}),
    (
        "multi_turn",
        [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Name a colour."},
            {"role": "assistant", "content": "Blue."},
            {"role": "user", "content": "Another one, with some 日本語."},
        ],
        None,
        {},
    ),
    ("tools", [ASK], [WEATHER], {}),
    ("tool_round_trip", [ASK, CALLED, ANSWERED], [WEATHER], {}),
    ("tool_round_trip_cleared", [ASK, CALLED, ANSWERED], [WEATHER], {"clear_thinking": True}),
    ("parallel_results", PARALLEL, [WEATHER, SEARCH], {}),
    ("reasoning_kept", REASONED, None, {}),
    ("reasoning_kept_explicitly", REASONED, None, {"clear_thinking": False}),
    ("reasoning_cleared", REASONED, None, {"clear_thinking": True}),
    (
        "reasoning_in_content",
        [
            {"role": "user", "content": "Name a colour."},
            {"role": "assistant", "content": "<think>Any will do.</think>\n\nBlue.  "},
            {"role": "user", "content": "Another."},
        ],
        None,
        {},
    ),
    ("effort_low", [ASK], [WEATHER], {"reasoning_effort": "low"}),
    ("effort_high", [ASK], None, {"reasoning_effort": "high"}),
    ("effort_unnamed", [ASK], None, {"reasoning_effort": "medium"}),
]

FORECAST = {
    "type": "function",
    "function": {
        "name": "forecast",
        "description": "Forecast",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "options": {
                    "type": "object",
                    "properties": {
                        "units": {"type": "string"},
                        "hours": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["units", "hours"],
                },
                "tags": {"type": "array", "items": {"type": "string"}},
                "verbose": {"type": "boolean"},
                "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            },
            "required": ["city", "options"],
        },
    },
}

NOW = {
    "type": "function",
    "function": {
        "name": "now",
        "description": "The current time",
        "parameters": {"type": "object", "properties": {"zone": {"type": "string"}}},
    },
}


def assistant(reasoning: str, content: str, calls: list[dict]) -> dict:
    message = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    if calls:
        message["tool_calls"] = calls
    return message


# (name, messages, tools, template arguments, the assistant reply). The reply's text is what the
# template writes for it after the generation prompt; the engine must read it back as the reply.
REPLIES = [
    ("single_call", [ASK], [WEATHER], {},
     assistant("One call for Paris.", "", [call("call_1", "get_weather", {"city": "Paris", "days": 2})])),
    ("content_and_calls", PARALLEL[:1], [WEATHER, SEARCH], {},
     assistant("Two tools.", "Both at once.", [
         call("call_b", "get_weather", {"city": "Paris"}),
         call("call_A", "search", {"query": "llama.cpp \"chat\"\nparser </arg_val <line> & more",
                                   "limit": 3}),
     ])),
    ("json_values", [ASK], [FORECAST], {},
     assistant("", "", [call("call_1", "forecast", {
         "city": "Oslo",
         "options": {"units": "metric", "hours": [6, 12]},
         "tags": ["日本", "a\"b"],
         "verbose": True,
         "limit": None,
     })])),
    ("call_without_arguments", [{"role": "user", "content": "What time is it?"}], [NOW], {},
     assistant("Ask the clock.", "", [call("call_1", "now", {})])),
    ("content_without_calls", [ASK], [WEATHER], {"reasoning_effort": "low"},
     assistant("No tool needed.", "It is sunny in Paris.", [])),
]

# Models whose references hold replies.
REPLYING = {"glm53"}

FIXED_NOW = datetime(2026, 9, 27, 12, 0, 0, tzinfo=timezone.utc)


def vocabulary(metadata: dict[str, Value]) -> bytes:
    """The tokenizer metadata (and architecture) of a GGUF's metadata, as a GGUF without tensors."""
    kept = {key: value for key, value in metadata.items()
            if key.startswith("tokenizer.") or key == "general.architecture"}
    return encode_header(kept, [], DEFAULT_ALIGNMENT)


def converted_metadata(files: Path, model_class: str, llama_cpp: Path) -> dict[str, Value]:
    """The metadata llama.cpp's converter writes for the checkpoint's vocabulary: its model class's
    `set_vocab` over the checkpoint's hyperparameters as the converter merges them, no tensors."""
    sys.path.insert(0, str(llama_cpp))
    import importlib

    import gguf
    from conversion.base import ModelBase

    module, name = model_class.rsplit(".", 1)
    cls = getattr(importlib.import_module(f"conversion.{module}"), name)
    model = cls.__new__(cls)  # only the vocabulary: no tensors are indexed
    model.dir_model = files
    hparams = ModelBase.load_hparams(files, False)
    model.hparams = {**hparams, **hparams.get("text_config", {})}  # as TextModel.__init__ merges them
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "vocabulary.gguf"
        model.gguf_writer = gguf.GGUFWriter(path, gguf.MODEL_ARCH_NAMES[cls.model_arch])
        model.set_vocab()
        model.gguf_writer.write_header_to_file()
        model.gguf_writer.write_kv_data_to_file()
        model.gguf_writer.write_tensors_to_file()
        model.gguf_writer.close()
        return LocalSource(path).header.metadata


def generate(name: str, output: Path, llama_cpp: Path | None) -> None:
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

    output.mkdir(parents=True, exist_ok=True)
    if name in PUBLISHED:
        repository, revision, path, reference, reference_revision = PUBLISHED[name]
        metadata = RemoteSource(repository, revision, path).header.metadata
        source = {"gguf": {"repository": repository, "revision": revision, "path": path}}
        tokenizer_json = hf_hub_download(reference, "tokenizer.json", revision=reference_revision)
        renderer = AutoTokenizer.from_pretrained(
            reference, revision=reference_revision, trust_remote_code=False
        )
    else:
        if llama_cpp is None:
            raise SystemExit(f"{name} is converted: --llama-cpp names llama.cpp's source tree")
        reference, reference_revision, model_class = CONVERTED[name]
        files = output / name
        for file in TOKENIZER_FILES:
            hf_hub_download(reference, file, revision=reference_revision, local_dir=files)
        metadata = converted_metadata(files, model_class, llama_cpp)
        source = {"converter": model_class}
        tokenizer_json = str(files / "tokenizer.json")
        renderer = AutoTokenizer.from_pretrained(files, trust_remote_code=False)
    (output / f"{name}.gguf.gz").write_bytes(gzip.compress(vocabulary(metadata), 9, mtime=0))
    tokenizer = Tokenizer.from_file(tokenizer_json)

    def ids(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False).ids

    corpus = [{"text": text, "ids": ids(text), "decoded": tokenizer.decode(ids(text), skip_special_tokens=False)}
              for text in CORPUS]
    template = metadata["tokenizer.chat_template"].value
    renders = []
    for conversation, messages, tools, template_arguments in CONVERSATIONS:
        text = renderer.apply_chat_template(
            messages,
            tools=tools,
            chat_template=template,
            add_generation_prompt=True,
            tokenize=False,
            **template_arguments,
        )
        renders.append({"name": conversation, "messages": messages, "tools": tools or [],
                        "template_arguments": template_arguments, "text": text, "ids": ids(text)})
    replies = []
    for reply, messages, tools, template_arguments, message in REPLIES if name in REPLYING else []:
        def render(conversation: list[dict], generation_prompt: bool) -> str:
            return renderer.apply_chat_template(
                conversation,
                tools=tools,
                chat_template=template,
                add_generation_prompt=generation_prompt,
                tokenize=False,
                **template_arguments,
            )

        prompt = render(messages, True)
        written = render([*messages, message], False)
        if not written.startswith(prompt):
            raise SystemExit(f"{name} {reply}: the reply's render does not extend the prompt")
        replies.append({
            "name": reply, "messages": messages, "tools": tools, "template_arguments": template_arguments,
            "text": written[len(prompt):], "content": message["content"],
            "reasoning": message["reasoning_content"],
            "calls": [{"name": tool_call["function"]["name"], "arguments": tool_call["function"]["arguments"]}
                      for tool_call in message.get("tool_calls", [])],
        })
    record = {
        **source,
        "reference": {"repository": reference, "revision": reference_revision},
        "now": int(FIXED_NOW.timestamp()),
        "corpus": corpus,
        "renders": renders,
        **({"replies": replies} if replies else {}),
    }
    (output / f"{name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n")
    print(name, len(corpus), "texts,", len(renders), "renders,", len(replies), "replies")


def main() -> int:
    models = [*PUBLISHED, *CONVERTED]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help=f"any of {', '.join(models)}; default all")
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--llama-cpp", type=Path, help="llama.cpp v0.9.0's source tree, for converted models")
    arguments = parser.parse_args()
    unknown = sorted(set(arguments.names) - set(models))
    if unknown:
        parser.error(f"unknown names: {', '.join(unknown)}")
    for name in arguments.names or models:
        generate(name, arguments.output, arguments.llama_cpp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
