"""DeepSeek V4.1's chat encoding: the reference the engine's render of a V4.1 GGUF is compared against.

Written from reading the checkpoint's own encoding, deepseek-ai/DeepSeek-V4.1-Flash-1017 at
`1bae0e1088993b9b82de08f59de5be5a34e42170`, `encoding/encoding.py` (`encode_messages`). That file is Python
shipped in a checkpoint, so it is read as data and never imported or run. `check_golden` compares this renderer
with the checkpoint's own expected outputs (`encoding/tests`).

It covers the encoding's text features the engine's chat path carries: system messages (first and
mid-conversation), user, assistant and tool turns, latest reminders, text content parts, reasoning, tool
declarations and calls, thinking on or off, the reasoning effort and `drop_thinking`. Images, quick-instruction
tasks, `wo_eos`, response formats, tool namespaces and continued `context` are not: the engine's request model has
no field for them.

`request` maps a chat request (top-level tools, `enable_thinking`) onto the encoding's input as DeepSeek's own port
does (deepseek-ai/DeepSeek-V4.1-Flash, `chat_template.jinja`): the tools join a leading system message, or a new
empty one.

`python -m magnitude_references.deepseek41_chat --tokenizer tokenizer.json --encoding-tests encoding/tests
--output DIR` checks the renderer on the goldens, then writes the engine's fixture: the reference tokenizer's ids
for a corpus, the renders of a set of requests, and assistant replies with tool calls as the text the model writes
after a prompt (`deepseek41.json`), and the published GGUF's vocabulary
(`DeepSeek-V4.1-Flash-1017-vocabulary.gguf.gz`, read with range requests).
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
from pathlib import Path
from typing import Any

from magnitude_references.gguf import DEFAULT_ALIGNMENT, RemoteSource, encode_header

BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
THINK_START = "<think>"
THINK_END = "</think>"
DSML = "｜DSML｜"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
SYSTEM = "<｜System｜>"
LATEST_REMINDER = "<｜latest_reminder｜>"

# encoding.py:443-453
EFFORTS = {"low": 50, "high": 75, "max": 100}
EFFORT_TEMPLATE = "Reasoning Effort: {} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"

# encoding.py:481-506, with its DSML tag names (:428-430) filled in
TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜ calls>" block like the following:

<｜DSML｜ calls>
<｜DSML｜ invoke name="$TOOL_NAME">
<｜DSML｜ parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜ parameter>
...
</｜DSML｜ invoke>
<｜DSML｜ invoke name="$TOOL_NAME2">
...
</｜DSML｜ invoke>
</｜DSML｜ calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

{schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _texts(content: Any) -> list[str]:
    """A message's text blocks: a string is one, a list holds text parts (encoding.py:219, :387-393)."""
    if content is None:
        return [""]
    if isinstance(content, str):
        return [content]
    for part in content:
        if part.get("type") != "text":
            raise ValueError(f"only text content parts are supported: {part.get('type')}")
    return [part.get("text", "") for part in content]


def _text(content: Any) -> str:
    return "\n\n".join(_texts(content))


def _merge(messages: list[dict]) -> list[dict]:
    """Runs of user and tool messages as one user message of blocks (encoding.py:184-230)."""
    merged: list[dict] = []
    for message in messages:
        message = copy.deepcopy(message)
        role = message["role"]
        if role == "tool":
            block = {"type": "tool_result", "id": message.get("tool_call_id", ""), "text": _text(message.get("content"))}
            if merged and merged[-1]["role"] == "user":
                merged[-1]["blocks"].append(block)
            else:
                merged.append({"role": "user", "blocks": [block]})
        elif role == "user":
            blocks = [{"type": "text", "text": text} for text in _texts(message.get("content"))]
            if merged and merged[-1]["role"] == "user":
                merged[-1]["blocks"].extend(blocks)
            else:
                message["blocks"] = blocks
                merged.append(message)
        else:
            merged.append(message)
    return merged


def _sort_tool_results(messages: list[dict]) -> None:
    """Tool results in the order of the latest assistant's calls (encoding.py:233-266)."""
    order: dict[str, int] = {}
    for message in messages:
        if message["role"] == "assistant" and message.get("tool_calls"):
            order = {}
            for index, call in enumerate(message["tool_calls"]):
                if call.get("id"):
                    order[call["id"]] = index
        elif message["role"] == "user":
            results = [block for block in message["blocks"] if block["type"] == "tool_result"]
            if len(results) > 1 and order:
                ranked = iter(sorted(results, key=lambda block: order.get(block["id"], 0)))
                message["blocks"] = [next(ranked) if block["type"] == "tool_result" else block
                                     for block in message["blocks"]]


def _last_user(messages: list[dict]) -> int:
    """A user message with a text block, or a system message after the first (encoding.py:562-582)."""
    for index in range(len(messages) - 1, -1, -1):
        role = messages[index]["role"]
        if role == "user" and any(block["type"] == "text" for block in messages[index]["blocks"]):
            return index
        if role == "system" and index > 0:
            return index
    return -1


def _arguments(call: dict) -> dict:
    """A call's arguments as a dict: a JSON string is decoded, at most twice (encoding.py:532-544)."""
    arguments = call["function"]["arguments"]
    if not isinstance(arguments, dict):
        for _ in range(2):
            if not isinstance(arguments, str):
                break
            try:
                arguments = json.loads(arguments)
            except ValueError:
                break
        if not isinstance(arguments, dict):
            arguments = {"arguments": call["function"]["arguments"]}
    return arguments


def _calls(calls: list[dict]) -> str:
    """DSML tool calls (encoding.py:524-555, :669-683)."""
    invokes = []
    for call in calls:
        parameters = [
            f'<{DSML} parameter name="{key}" string="{"true" if isinstance(value, str) else "false"}">'
            f"{value if isinstance(value, str) else _json(value)}</{DSML} parameter>"
            for key, value in _arguments(call).items()
        ]
        invokes.append(f'<{DSML} invoke name="{call["function"]["name"]}">\n' + "\n".join(parameters)
                       + f"\n</{DSML} invoke>")
    return f"\n\n<{DSML} calls>\n" + "\n".join(invokes) + f"\n</{DSML} calls>"


def _render(index: int, messages: list[dict], thinking: bool, drop: bool, budget: int) -> str:
    """One message (encoding.py:585-742)."""
    message = messages[index]
    role = message["role"]
    last_user = _last_user(messages)
    effort = EFFORT_TEMPLATE.format(budget) if index == 0 and thinking else ""
    prompt = SYSTEM if index == 0 and (effort or role == "system") else ""
    prompt += effort
    if role == "system":
        if index > 0:
            prompt += SYSTEM
        prompt += _text(message.get("content"))
        if message.get("tools"):
            prompt += "\n\n" + TOOLS_TEMPLATE.format(
                schemas="\n".join(_json(tool["function"]) for tool in message["tools"]))
    elif role == "user":
        prompt += USER + "\n\n".join(
            block["text"] if block["type"] == "text" else f"<tool_result>{block['text']}</tool_result>"
            for block in message["blocks"])
    elif role == "latest_reminder":
        prompt += LATEST_REMINDER + _text(message.get("content"))
    elif role == "assistant":
        if thinking and (not drop or index > last_user):
            prompt += (message.get("reasoning_content") or "") + THINK_END
        prompt += _text(message.get("content"))
        if message.get("tool_calls"):
            prompt += _calls(message["tool_calls"])
        prompt += EOS
    else:
        raise ValueError(f"unknown role: {role}")
    following = messages[index + 1]["role"] if index + 1 < len(messages) else None
    if following in (None, "assistant", "latest_reminder") and (role == "user" or (role == "system" and index > 0)):
        prompt += ASSISTANT + (THINK_START if thinking and (not drop or index >= last_user) else THINK_END)
    return prompt


def encode(messages: list[dict], *, thinking: bool, drop_thinking: bool = True, reasoning_effort: Any = None) -> str:
    """The prompt of a conversation in the encoding's input form (encoding.py:770-844)."""
    effort = "high" if reasoning_effort is None else reasoning_effort
    if not ((type(effort) is int and 1 <= effort <= 100) or effort in EFFORTS):
        raise ValueError(f"invalid reasoning effort: {effort!r}")
    budget = EFFORTS[effort] if isinstance(effort, str) else effort
    merged = _merge(messages)
    _sort_tool_results(merged)
    drop = drop_thinking and not any(message.get("tools") for message in merged)
    return BOS + "".join(_render(index, merged, thinking, drop, budget) for index in range(len(merged)))


def request(messages: list[dict], tools: list[dict], *, enable_thinking: bool = True, drop_thinking: bool = True,
            reasoning_effort: Any = None) -> str:
    """The prompt of a chat request with its generation prompt: top-level tools join the leading system message,
    or a new empty one."""
    messages = copy.deepcopy(messages)
    if tools:
        if messages and messages[0]["role"] == "system":
            messages[0]["tools"] = tools
        else:
            messages.insert(0, {"role": "system", "content": "", "tools": tools})
    return encode(messages, thinking=enable_thinking, drop_thinking=drop_thinking, reasoning_effort=reasoning_effort)


def reply(messages: list[dict], tools: list[dict], assistant: dict, **arguments: Any) -> str:
    """The text the model writes after a request's prompt to give `assistant`: the conversation's encoding with
    that reply, past the request's prompt, without its closing EOS (generation stops on EOS and drops it). This is
    the completion `parse_message_from_completion_text` reads (encoding.py:946-989)."""
    prompt = request(messages, tools, **arguments)
    answered = request(messages + [assistant], tools, **arguments)
    if not answered.startswith(prompt) or not answered.endswith(EOS):
        raise AssertionError(f"the reply does not extend the prompt:\n{prompt!r}\n{answered!r}")
    return answered[len(prompt):-len(EOS)]


def check_golden(tests: Path) -> list[int]:
    """The checkpoint's expected outputs this renderer covers, each rendered equal (as `test_encoding.py` loads
    them: top-level tools on the first message, "chat" mode unless the case names one)."""
    covered = []
    for path in sorted(tests.glob("test_input_*.json")):
        number = int(path.stem.rsplit("_", 1)[1])
        data = json.loads(path.read_text())
        case = data if isinstance(data, dict) else {"messages": data}
        messages = copy.deepcopy(case["messages"])
        if "tools" in case:
            messages[0]["tools"] = case["tools"]
        if any(isinstance(m.get("content"), list) or "task" in m or "wo_eos" in m for m in messages):
            continue
        rendered = encode(messages, thinking=case.get("thinking_mode", "chat") == "thinking",
                          reasoning_effort=case.get("reasoning_effort"))
        expected = (tests / f"test_output_{number}.txt").read_text()
        if rendered != expected:
            raise AssertionError(f"golden {number} renders differently:\n{rendered!r}\n{expected!r}")
        covered.append(number)
    return covered


# The engine's fixture. The texts cover V4.1's three splits and its added tokens: digit runs, the CJK ideograph and
# kana run (U+4E00-U+9FA5, U+3040-U+30FF) against ideographs and Hangul outside it, a letter run after one ASCII
# punctuation mark, whitespace runs, emoji, contractions, and the special-token text the vocabulary adds.
CORPUS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog. It's a test; they'll see, we've done, I'd go, you're right, "
    "I'M LOUD, DON'T.",
    "Numbers: 1 12 123 1234 12345 123456 1234567 3.14159 -42 1,000,000 0x1F 2026-09-27 v1.2.3 第1234章",
    "  leading spaces and trailing   ",
    "multiple    internal     spaces\tand\ttabs\t\t\tend",
    "line one\nline two\n\nline four\n\n\n\nafter blank lines\n",
    "windows\r\nline\r\nendings\r\n\r\ndouble crlf and lone \r carriage",
    "\n\n\n",
    "   ",
    " ",
    "a",
    "日本語のテキストと中文文本，还有한국어 텍스트。ひらがなカタカナｶﾀｶﾅ",
    "Edge ideographs: 一龥龦鿿 㐀 𠀀 々〆ヶ ゠ヿ ㇰ",
    "Emoji: 😀👍🏽👨‍👩‍👧‍👦 🇳🇴 ❤️ ✅",
    "Combining: é ä ñ — café naïve résumé Ångström",
    "Cyrillic: Привет, мир! Greek: Γειά σου Κόσμε. Math: ∑ᵢ xᵢ² ≤ ∞ ∫₀¹ f(x) dx ≠ π",
    "Punctuation then letters: .NET #include @user $HOME 'quoted' \"double\" (paren) [x] {y} <tag> ~home",
    "Repeated punctuation!!! ??? ... --- === *** ### @@@ $$$ %%% ^^^ &&& ((( ))) 。。。！？",
    "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n",
    "{\"key\": \"value\", \"list\": [1, 2, 3], \"nested\": {\"a\": null, \"b\": true}}",
    "URL: https://example.com/path/to/page?query=1&other=two#frag and email a.b@c.io",
    "CamelCaseIdentifier snake_case_name SCREAMING_CASE kebab-case iPhone McDonald's",
    "Zero​width and non breaking and ideographic　space",
    "\u0000\u0001\u007f bytes ÿĀ \U0001f600\U000e0001 end",
    "<｜begin▁of▁sentence｜><｜System｜>sys<｜User｜>hi<｜Assistant｜><think>r</think>ok<｜end▁of▁sentence｜>",
    "<｜latest_reminder｜>date<tool_result>x</tool_result> <｜DSML｜ calls> ｜DSML｜ <dsml:invoke></dsml:invoke>",
    "<｜tool▁calls▁begin｜><｜fim▁hole｜><|EOT|><｜▁pad▁｜><｜place▁holder▁no▁0｜><｜deepseek_image｜><｜unknown｜>",
    "<|im_start|>user\nhi<|im_end|>\n<s> </s> <bos> <eos> <|endoftext|> <think>x</think>",
]

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}, "days": {"type": "integer"}},
            "required": ["city"],
        },
    },
}
SEARCH = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
}


def _call(id: str, name: str, arguments: dict) -> dict:
    return {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}


WEATHER_TRIP = [
    {"role": "user", "content": "What is the weather in Paris and Tokyo?"},
    {"role": "assistant", "content": "", "reasoning_content": "Two cities: two calls.",
     "tool_calls": [_call("call_1", "get_weather", {"city": "Paris", "days": 2}),
                    _call("call_2", "get_weather", {"city": "Tokyo"})]},
    # Results out of call order: the encoding puts them back in it.
    {"role": "tool", "tool_call_id": "call_2", "content": "{\"temperature\": 25}"},
    {"role": "tool", "tool_call_id": "call_1", "content": "{\"temperature\": 21, \"sky\": \"clear\"}"},
]
LOOKUP = [  # test_encoding.py:508-546, drop_thinking without a tools declaration
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
    {"role": "user", "content": "q2"},
    {"role": "assistant", "content": "", "reasoning_content": "r2", "tool_calls": [_call("c1", "lookup", {})]},
    {"role": "tool", "tool_call_id": "c1", "content": "value"},
]
HISTORY = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Name a colour."},
    {"role": "assistant", "content": "Blue.", "reasoning_content": "Any colour will do."},
    {"role": "user", "content": "Another one, with some 日本語."},
]

# (name, messages, tools, the request's template arguments). Every render adds the generation prompt. The first
# is the plain user turn.
CONVERSATIONS = [
    ("user", [{"role": "user", "content": "Hi there! What's 2+2?"}], [], {}),
    ("user_chat", [{"role": "user", "content": "Hi there! What's 2+2?"}], [], {"enable_thinking": False}),
    ("system_multi_turn", HISTORY, [], {}),
    ("system_multi_turn_chat", HISTORY, [], {"enable_thinking": False}),
    ("keep_thinking", HISTORY, [], {"drop_thinking": False}),
    ("effort_low", HISTORY[1:2], [], {"reasoning_effort": "low"}),
    ("effort_max", HISTORY[1:2], [], {"reasoning_effort": "max"}),
    ("effort_30", HISTORY[1:2], [], {"reasoning_effort": 30}),
    ("text_parts", [{"role": "user", "content": [{"type": "text", "text": "First part."},
                                                  {"type": "text", "text": "Second part."}]}], [], {}),
    ("mid_system", [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"},
        {"role": "system", "content": "From now on, answer in French."},
    ], [], {}),
    ("latest_reminder", [
        {"role": "system", "content": "You are DeepSeek."},
        {"role": "latest_reminder", "content": "2026-11-07, Oslo, App, English"},
        {"role": "user", "content": "What day is it?"},
    ], [], {}),
    ("tools", [{"role": "user", "content": "What is the weather in Paris?"}], [WEATHER, SEARCH], {}),
    ("tools_system_chat", [{"role": "system", "content": "Use tools when they help."},
                           {"role": "user", "content": "Search for llama.cpp."}], [SEARCH],
     {"enable_thinking": False}),
    ("tool_round_trip", WEATHER_TRIP, [WEATHER], {}),
    ("tool_round_trip_chat", WEATHER_TRIP, [WEATHER], {"enable_thinking": False}),
    ("tool_calls_without_tools", LOOKUP, [], {}),
    ("tool_calls_next_turn", LOOKUP + [{"role": "user", "content": "q3"}], [], {}),
]

LOOKUP_TOOL = {
    "type": "function",
    "function": {"name": "lookup", "description": "Look a value up", "parameters": {"type": "object", "properties": {}}},
}
PLAN = {
    "type": "function",
    "function": {
        "name": "plan_trip",
        "description": "Plan a trip",
        "parameters": {
            "type": "object",
            "properties": {
                "cities": {"type": "array", "items": {"type": "string"}},
                "options": {"type": "object", "properties": {"metric": {"type": "boolean"}, "budget": {"type": "number"}}},
                "note": {"type": "string"},
            },
            "required": ["cities"],
        },
    },
}

# (name, messages, tools, the request's template arguments, the assistant reply). The engine reads each reply's
# text back to the reply's content, reasoning and calls. The calls cover one call, several, none of a call's
# arguments, string values with quotes and newlines, and JSON values of every kind, in both modes.
REPLIES = [
    ("single_call", [{"role": "user", "content": "What is the weather in Paris?"}], [WEATHER, SEARCH], {},
     {"role": "assistant", "content": "", "reasoning_content": "Paris only: one call.",
      "tool_calls": [_call("call_1", "get_weather", {"city": "Paris", "days": 2})]}),
    ("single_call_without_arguments", LOOKUP[:3], [LOOKUP_TOOL], {}, LOOKUP[3]),
    ("several_calls", WEATHER_TRIP[:1], [WEATHER], {}, WEATHER_TRIP[1]),
    ("several_calls_chat", WEATHER_TRIP[:1], [WEATHER], {"enable_thinking": False}, WEATHER_TRIP[1]),
    ("content_and_call_chat", [{"role": "system", "content": "Use tools when they help."},
                               {"role": "user", "content": "Search for llama.cpp."}], [SEARCH],
     {"enable_thinking": False},
     {"role": "assistant", "content": "Searching.",
      "tool_calls": [_call("call_1", "search", {"query": "llama.cpp \"chat\" parser\nsecond line"})]}),
    ("json_arguments", [{"role": "user", "content": "Plan Paris then Tokyo, metric, under 2000."}],
     [PLAN, WEATHER], {},
     {"role": "assistant", "content": "Planning both.", "reasoning_content": "One plan, then the weather.",
      "tool_calls": [_call("call_1", "plan_trip", {"cities": ["Paris", "東京"],
                                                   "options": {"metric": True, "budget": 1999.5},
                                                   "note": "<b>no</b> rush"}),
                     _call("call_2", "get_weather", {"city": "Tokyo", "days": 0})]}),
]

# The published GGUF whose vocabulary the fixture carries: our converter's output, its tokenizer keys exactly those
# a converted slice carries.
GGUF = ("magnitudedev/DeepSeek-V4.1-Flash-1017-GGUF", "3769ebe41a97ba3619216ebc1fefb53f2c432495",
        "DeepSeek-V4.1-Flash-1017-MXFP4_MOE-00001-of-00008.gguf")
REFERENCE = ("deepseek-ai/DeepSeek-V4.1-Flash-1017", "1bae0e1088993b9b82de08f59de5be5a34e42170")
NOW = 1_767_225_600


def vocabulary(repository: str, revision: str, path: str) -> bytes:
    """The file's tokenizer metadata (and architecture) as a GGUF without tensors, read by range requests."""
    metadata = RemoteSource(repository, revision, path).header.metadata
    kept = {key: value for key, value in metadata.items()
            if key.startswith("tokenizer.") or key == "general.architecture"}
    return encode_header(kept, [], DEFAULT_ALIGNMENT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizer", type=Path, required=True,
                        help=f"tokenizer.json of {REFERENCE[0]} at {REFERENCE[1]}")
    parser.add_argument("--encoding-tests", type=Path, required=True, help="the same revision's encoding/tests")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    from tokenizers import Tokenizer

    covered = check_golden(arguments.encoding_tests)
    tokenizer = Tokenizer.from_file(str(arguments.tokenizer))

    def ids(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False).ids

    corpus = [{"text": text, "ids": ids(text), "decoded": tokenizer.decode(ids(text), skip_special_tokens=False)}
              for text in CORPUS]
    renders = []
    for name, messages, tools, template_arguments in CONVERSATIONS:
        text = request(messages, tools, **template_arguments)
        renders.append({"name": name, "messages": messages, "tools": tools,
                        "template_arguments": template_arguments, "text": text, "ids": ids(text)})
    replies = []
    for name, messages, tools, template_arguments, assistant in REPLIES:
        thinking = template_arguments.get("enable_thinking", True)
        replies.append({"name": name, "messages": messages, "tools": tools, "template_arguments": template_arguments,
                        "text": reply(messages, tools, assistant, **template_arguments),
                        "content": assistant["content"],
                        "reasoning": assistant.get("reasoning_content", "") if thinking else "",
                        "calls": [{"name": call["function"]["name"], "arguments": call["function"]["arguments"]}
                                  for call in assistant["tool_calls"]]})
    record = {
        "gguf": dict(zip(("repository", "revision", "path"), GGUF)),
        "reference": dict(zip(("repository", "revision"), REFERENCE)),
        "goldens": covered,
        "now": NOW,
        "corpus": corpus,
        "renders": renders,
        "replies": replies,
    }
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "deepseek41.json").write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n")
    (arguments.output / "DeepSeek-V4.1-Flash-1017-vocabulary.gguf.gz").write_bytes(
        gzip.compress(vocabulary(*GGUF), 9, mtime=0))
    print(f"goldens {covered}; {len(corpus)} texts, {len(renders)} renders, {len(replies)} replies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
