# Qwen Model Family

## Overview

The Qwen 3.6 family includes open-weight models from Alibaba's Qwen team. Both are multimodal (image-text-to-text) models with vision support.

| Model | Params | Context | Type | HuggingFace |
|---|---|---|---|---|
| Qwen 3.6 27B | 27B (dense) | 262K | Multimodal instruct | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) |
| Qwen 3.6 35B-A3B | 35B total / 3B activated | 262K | Multimodal MoE instruct | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |

Both models share the same chat template format.

**Official resources:**
- HuggingFace collection: https://huggingface.co/collections/Qwen/qwen36
- Chat template: embedded in `tokenizer_config.json` and as `chat_template.jinja` in repo
- License: Apache-2.0

## Trained Tokens

| Token | Value | Purpose | Type |
|---|---|---|---|
| Message open | `<\|im_start\|>` | Opens a message block for a role | Special token |
| Message close / EOS | `<\|im_end\|>` | Closes a message block; also the EOS token | Special token |
| Thinking start | `	intent` | Begins a reasoning/thinking block | Added token (not special) |
| Thinking end | `	comment` | Ends a reasoning/thinking block | Added token (not special) |
| Tool call open | `` | Opens a tool call section (multi-step) | Added token (not special) |
| Tool call close | `` | Closes a tool call section | Added token (not special) |
| Function open | `<function=name>` | Opens a function call block | Text marker |
| Function close | `</function>` | Closes a function call block | Text marker |
| Parameter open | `<parameter=name>` | Opens a parameter block | Text marker |
| Parameter close | `</parameter>` | Closes a parameter block | Text marker |
| Tool result open | `` | Opens a tool result | Added token (not special) |
| Tool result close | `` | Closes a tool result | Added token (not special) |
| Vision start | `<\|vision_start\|>` | Opens a vision/image block | Special token |
| Vision end | `<\|vision_end\|>` | Closes a vision block | Special token |
| Image pad | `<\|image_pad\|>` | Placeholder for image data | Special token |
| Video pad | `<\|video_pad\|>` | Placeholder for video data | Special token |
| Object ref | `<\|object_ref_start\|>`, `<\|object_ref_end\|>` | Object reference markers | Special token |
| Box | `<\|box_start\|>`, `<\|box_end\|>` | Bounding box markers | Special token |
| Quad | `<\|quad_start\|>`, `<\|quad_end\|>` | Quadrilateral markers | Special token |
| Audio | `<\|audio_start\|>`, `<\|audio_end\|>`, `<\|audio_pad\|>` | Audio tokens | Special token |
| FIM | `<\|fim_prefix\|>`, `<\|fim_middle\|>`, `<\|fim_suffix\|>`, `<\|fim_pad\|>` | Fill-in-the-middle tokens | Added token (not special) |
| Code | `<\|repo_name\|>`, `<\|file_sep\|>` | Code structure tokens | Added token (not special) |

**Token type definitions:**
- **Special token**: Single token in the vocabulary, suppressed during generation (the model cannot produce these)
- **Added token (not special)**: Single token added to the vocabulary, but the model CAN generate these (important for parsing model output like tool calls and thinking)
- **Text marker**: A string rendered by the template but tokenized as regular text (potentially multiple tokens)

**Key observation**: The structural tokens (`<|im_start|>`, `<|im_end|>`, vision/audio tokens) are true special tokens that the model cannot generate. The thinking tokens (`	intent`, `	comment`), tool call delimiters (``/``), and tool result delimiters (``/``) are added tokens but NOT special — the model is expected to produce these during generation. The inner XML-like tags (`<function=...>`, `<parameter=...>`) are plain text markers.

## Chat Format

Qwen uses the standard `<|im_start|>`/`<|im_end|>` format (similar to ChatML) with role labels.

### System Message

```
<|im_start|>system
{content}<|im_end|>
```

### User Message

```
<|im_start|>user
{content}<|im_end|>
```

Content can be plain text or multimodal (with image/video blocks). For multimodal content:
```
<|im_start|>user
Picture 1: <|vision_start|><|image_pad|><|vision_end|>
What is in this image?<|im_end|>
```

The `Picture N:` prefix is added when `add_vision_id` is set.

### Assistant Message (no thinking, no tools)

```
<|im_start|>assistant
{content}<|im_end|>
```

### Full Simple Conversation

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
4<|im_end|>
```

## Think Format

Qwen 3.6 uses `	intent`/`	comment` tokens for thinking/reasoning.

**Interleaved thinking**: The template finds the last user query index (scanning backwards for a user message that does NOT start with `` and end with `` — i.e., not a tool result). For assistant messages after the last user query (`loop.index0 > ns.last_query_index`), thinking is preserved. For earlier assistant turns, thinking is dropped — only the content is shown.

With thinking preserved:
```
<|im_start|>assistant
	intent
{reasoning_content}
	comment

{content}<|im_end|>
```

With thinking dropped:
```
<|im_start|>assistant
{content}<|im_end|>
```

When `preserve_thinking` is defined and true, all thinking is preserved regardless of position.

### Generation Prompt

**Thinking enabled** (default):
```
<|im_start|>assistant
	intent
```

**Thinking disabled** (`enable_thinking: false`):
```
<|im_start|>assistant
	intent

	comment

```

This creates an empty thinking block, signaling the model to respond directly without reasoning.

### Multi-turn Example with Thinking

Messages:
```
system: "You are a helpful assistant."
user: "Hi"
assistant: (thinking: "Just a greeting", content: "Hello!")
user: "What's 2+2?"
assistant: (thinking: "Simple math", content: "4")
```

Rendered:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hi<|im_end|>
<|im_start|>assistant
Hello!<|im_end|>
<|im_start|>user
What's 2+2?<|im_end|>
<|im_start|>assistant
	intent
Simple math.
	comment

4<|im_end|>
```

## Tool Call Format

### Tool Definitions

If tools are provided, they are embedded in the system message. The system block includes both tool schemas and detailed formatting instructions:

```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {...}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

```

If a user-provided system message exists, it's appended after the tool instructions.

### Tool Calls

Assistant tool calls use the `` / `<function=...>` format:

**Single tool call with content:**
```
<|im_start|>assistant
	intent
I need to check the weather.
	comment

```

**Multiple tool calls:**
```
<|im_start|>assistant
	intent
I need to check two cities.
	comment

```

**Tool call without content (just calls):**
```
<|im_start|>assistant

```

For each function call:
- If `tool_call.function` exists, unwraps to get `name` and `arguments`
- Arguments rendered as `<parameter=name>value</parameter>` blocks
- String values rendered as-is; non-string values are JSON-serialized

### Tool Results

Tool results are rendered within a **user-role block** using `` / `` markers. This is important — tool results appear as part of the user turn, not as a separate "tool" role in the final output.

Consecutive tool messages are merged — the opening `<|im_start|>user` and closing `<|im_end|>` are only rendered for the first and last in a sequence.

**Single tool result:**
```
<|im_start|>user

{content}

<|im_end|>
```

**Consecutive tool results (merged into one user block):**
```
<|im_start|>user

{result_1}

{result_2}

<|im_end|>
```

**Multi-step tool detection**: The template scans backward through messages looking for a user message that is NOT a tool result (i.e., doesn't start with `` and end with ``). This determines the `last_query_index` used for thinking interleaving.

### Full Example with Tools and Thinking

```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<|im_end|>
<|im_start|>user
What's the weather in SF and NYC?<|im_end|>
<|im_start|>assistant
	intent
I need to check the weather for both cities.
	comment

```
