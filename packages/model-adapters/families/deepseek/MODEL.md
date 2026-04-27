# DeepSeek Model Family

## Overview

The DeepSeek V4 family includes open-weight Mixture-of-Experts models from DeepSeek AI. Both support a 1M token context window.

| Model | Params (Total/Active) | Context | HuggingFace |
|---|---|---|---|
| DeepSeek V4 Pro | 1.6T / 49B | 1M | [deepseek-ai/DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) |
| DeepSeek V4 Flash | 284B / 13B | 1M | [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) |
| DeepSeek V4 Pro Base | 1.6T / 49B | 1M | [deepseek-ai/DeepSeek-V4-Pro-Base](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-Base) |
| DeepSeek V4 Flash Base | 284B / 13B | 1M | [deepseek-ai/DeepSeek-V4-Flash-Base](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base) |

The Base models are pre-trained only; the Pro and Flash (without "Base") are instruct/fine-tuned models.

Both instruct models share the same chat format, defined by a Python encoding script (not a Jinja template).

**Official resources:**
- HuggingFace org: https://huggingface.co/deepseek-ai
- Encoding script: `encoding/encoding_dsv4.py` in the HuggingFace repo
- Encoding tests: `encoding/tests/` in the HuggingFace repo
- Technical report: `DeepSeek_V4.pdf` in the HuggingFace repo
- License: MIT

**⚠️ No Jinja template**: DeepSeek V4 does NOT provide a `chat_template` in `tokenizer_config.json` or a `chat_template.jinja` file. Instead, the repo ships a Python script (`encoding_dsv4.py`) that handles message encoding. This is the authoritative reference for the format.

## Trained Tokens

| Token | Value | Purpose | Type |
|---|---|---|---|
| BOS | `<｜begin▁of▁sentence｜>` | Marks the beginning of the entire input | Added token (special) |
| EOS / PAD | `<｜end▁of▁sentence｜>` | End of sequence; also used as PAD | Added token (special) |
| User role | `<｜User｜>` | Marks the start of a user message | Likely base vocab token* |
| Assistant role | `<｜Assistant｜>` | Marks the start of an assistant message | Likely base vocab token* |
| Latest reminder | `<｜latest_reminder｜>` | Special role for injecting a reminder message | Likely base vocab token* |
| Thinking start | `	intent` | Begins a reasoning/thinking block | Likely base vocab token* |
| Thinking end | `	comment` | Ends a reasoning/thinking block | Likely base vocab token* |
| DSML token | `｜DSML｜` | Namespace token for tool call markup | Likely base vocab token* |
| Tool call open | `<｜DSML｜tool_calls>` | Opens a tool calls section | Text marker |
| Tool call close | `</｜DSML｜tool_calls>` | Closes a tool calls section | Text marker |
| Invoke open | `<｜DSML｜invoke name="...">` | Opens a tool invocation | Text marker |
| Invoke close | `</｜DSML｜invoke>` | Closes a tool invocation | Text marker |
| Parameter open | `<｜DSML｜parameter name="..." string="true\|false">` | Opens a parameter (string flag indicates if value is string or JSON) | Text marker |
| Parameter close | `</｜DSML｜parameter>` | Closes a parameter | Text marker |
| Tool result | `<tool_result>...</tool_result>` | Wraps a tool result (embedded in user messages) | Text marker |
| Action task | `<｜action｜>` | Internal task classification token | Likely base vocab token* |
| Query task | `<｜query｜>` | Internal task classification token | Likely base vocab token* |
| Authority task | `<｜authority｜>` | Internal task classification token | Likely base vocab token* |
| Domain task | `<｜domain｜>` | Internal task classification token | Likely base vocab token* |
| Title task | `<｜title｜>` | Internal task classification token | Likely base vocab token* |
| Read URL task | `<｜read_url｜>` | Internal task classification token | Likely base vocab token* |

*\*These tokens are NOT in the `added_tokens_decoder` of the tokenizer config. They are likely single tokens in the base SentencePiece/BPE vocabulary (the `tokenizer.json` merge table), but this cannot be confirmed without inspecting the full vocabulary. The use of special Unicode characters (`▁` = U+2581, `｜` = U+FF5C) strongly suggests they were included as atomic tokens during tokenizer training.*

**Token type definitions:**
- **Added token (special)**: Explicitly added to the vocabulary as a special token, suppressed during generation
- **Likely base vocab token**: Not in `added_tokens_decoder` but likely a single token in the base vocabulary based on the encoding script's usage and special Unicode characters
- **Text marker**: A string rendered by the template/encoding script but tokenized as regular text (potentially multiple tokens)

**Key observation**: Unlike other model families, DeepSeek V4's tokenizer config only registers BOS and EOS as added tokens. The critical interaction tokens (`<｜User｜>`, `<｜Assistant｜>`, `	intent`, `	comment`, `｜DSML｜`) are handled by the encoding script and are likely part of the base vocabulary. The `｜DSML｜` namespace and all tool call XML tags are constructed as text strings in the encoding script, not as registered tokens.

Note the unicode characters in DeepSeek tokens: `▁` (unicode underscore) in BOS, and `｜` (fullwidth vertical bar) in DSML tokens. These are intentional — the tokenizer maps these specific character sequences to single tokens.

## Chat Format

DeepSeek V4 uses a unique format with role markers `<｜User｜>` and `<｜Assistant｜>` rather than the `<|im_start|>`/`<|im_end|>` pattern. There is no separate system role marker — system messages are rendered as plain text at the beginning.

### System Message

System content is rendered as plain text at the start, before any role markers:

```
<｜begin▁of▁sentence｜>{system_content}<｜User｜>
```

If tools are provided, they are appended after the system content (see Tool Call Format below).

### User Message

```
<｜User｜>{content}
```

### Assistant Message

```
<｜Assistant｜>{content}<｜end▁of▁sentence｜>
```

EOS (`<｜end▁of▁sentence｜>`) is appended after every assistant message.

### Full Simple Conversation

```
<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>What is 2+2?<｜Assistant｜>4<｜end▁of▁sentence｜>
```

## Think Format

DeepSeek V4 supports three reasoning effort modes:

| Mode | Description | Response Format |
|---|---|---|
| **Non-think** (chat) | Fast, intuitive responses | `	comment` + summary (no thinking block) |
| **Think High** (thinking) | Conscious logical analysis | `	intent` + thinking + `	comment` + summary |
| **Think Max** | Push reasoning to fullest extent | Special system prompt + `	intent` + thinking + `	comment` + summary |

### Chat mode (non-thinking)

The assistant message starts with `	comment` (the thinking-end token, acting as a role/content separator):

```
<｜Assistant｜>	comment{content}<｜end▁of▁sentence｜>
```

### Thinking mode

```
<｜Assistant｜>	intent{reasoning_content}	comment{content}<｜end▁of▁sentence｜>
```

### Thinking Max mode

A reasoning effort prefix is prepended before the system message:

```
Reasoning Effort: Absolute maximum with no shortcuts permitted.
You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.
Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.

<｜begin▁of▁sentence｜>{system_content}<｜User｜>{user_content}<｜Assistant｜>	intent{reasoning}	comment{content}<｜end▁of▁sentence｜>
```

### Interleaved Thinking

The encoding script supports dropping reasoning content from earlier turns. When `drop_thinking=True`:
- Messages with roles `user`, `system`, `tool`, `latest_reminder` are always kept
- Assistant messages before the last user index have their `reasoning_content` removed
- Developer messages before the last user index are dropped entirely

If any message in the conversation defines tools, thinking is NOT dropped (overridden to preserve all reasoning).

### Generation Prompt

After the last user message, the transition token depends on mode:

**Chat mode:**
```
{last_user_content}<｜Assistant｜>	comment
```

**Thinking mode:**
```
{last_user_content}<｜Assistant｜>	intent
```

### Multi-turn Example with Thinking

Messages:
```
system: "You are a helpful assistant."
user: "Hi"
assistant: (thinking: "Just a greeting", content: "Hello!")
user: "What's 2+2?"
assistant: (thinking: "Simple math", content: "4")
```

Rendered (with drop_thinking=True):
```
<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>Hi<｜Assistant｜>Hello!<｜end▁of▁sentence｜><｜User｜>What's 2+2?<｜Assistant｜>	intentSimple math.	comment4<｜end▁of▁sentence｜>
```

The first assistant's thinking is dropped because it's before the last user message.

## Tool Call Format

### Tool Definitions

Tools are embedded in the system message using a dedicated template. The system content comes first, then the tools section:

```
{system_content}

## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜tool_calls>" block like the following:

<｜DSML｜tool_calls>
<｜DSML｜invoke name="$TOOL_NAME">
<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>
...
</｜DSML｜invoke>
<｜DSML｜invoke name="$TOOL_NAME2">
...
</｜DSML｜invoke>
</｜DSML｜tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by 	intent), you MUST output your complete reasoning inside 	intent...	comment BEFORE any tool calls or final response.

Otherwise, output directly after 	comment with tool calls or final response.

### Available Tool Schemas

{tool_schemas_json}
```

Each tool schema is serialized as JSON and listed under "Available Tool Schemas".

### Tool Calls

Tool calls use the `｜DSML｜` namespace with XML-like markup:

```
<｜Assistant｜>	intent{reasoning}	comment

<｜DSML｜tool_calls>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls><｜end▁of▁sentence｜>
```

Multiple tool calls in one message:
```
<｜Assistant｜>	intent{reasoning}	comment

<｜DSML｜tool_calls>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
</｜DSML｜invoke>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="city" string="true">New York</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls><｜end▁of▁sentence｜>
```

The `string` attribute on parameters is important:
- `string="true"` — value is a literal string, rendered as-is
- `string="false"` — value is JSON (numbers, booleans, arrays, objects)

### Tool Results

DeepSeek V4 does NOT have a separate "tool" role. Tool results are **merged into user messages** using the `<tool_result>` tag.

The encoding script pre-processes conversations to merge tool messages into the preceding user message using a `content_blocks` format. Each tool result becomes a block within the user message:

```
<｜User｜>
<tool_result>{tool_result_content}</tool_result>
```

When there are both text and tool results in a user message:
```
<｜User｜>{user_text}

<tool_result>{tool_result_content}</tool_result>
```

Multiple tool results:
```
<｜User｜>
<tool_result>{result_1}</tool_result>

<tool_result>{result_2}</tool_result>
```

### Full Example with Tools and Thinking

```
<｜begin▁of▁sentence｜>You are a helpful assistant.

## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜tool_calls>" block like the following:

<｜DSML｜tool_calls>
<｜DSML｜invoke name="$TOOL_NAME">
<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>
...
</｜DSML｜invoke>
</｜DSML｜tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by 	intent), you MUST output your complete reasoning inside 	intent...	comment BEFORE any tool calls or final response.

Otherwise, output directly after 	comment with tool calls or final response.

### Available Tool Schemas

{"name": "get_weather", "description": "Get weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}<｜User｜>What's the weather in SF and NYC?<｜Assistant｜>	intentI need to check the weather for both cities.	comment

<｜DSML｜tool_calls>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
</｜DSML｜invoke>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="city" string="true">New York</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls><｜end▁of▁sentence｜><｜User｜>
<tool_result>65°F, sunny</tool_result>

<tool_result>45°F, rainy</tool_result><｜Assistant｜>	intentNow I have both results. SF is warm, NYC is cold.	commentSF is 65°F and sunny, NYC is 45°F and rainy.<｜end▁of▁sentence｜>
```

## Response Format

The model supports a `response_format` parameter that gets appended to the system message:

```
## Response Format:

You MUST strictly adhere to the following schema to reply:
{json_schema}
```

## Parsing Model Output

The encoding script provides `parse_message_from_completion_text()` for decoding model output:

1. In thinking mode, extract everything before `	comment` as `reasoning_content`
2. Extract content between `	comment` and `<｜end▁of▁sentence｜>` as the response
3. If `<｜DSML｜tool_calls>` is present, parse the invoke/parameter blocks into structured tool calls
4. Validate that no special tokens appear in content

The script raises `ValueError` on malformed output.
