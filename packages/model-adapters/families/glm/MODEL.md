# GLM Model Family

## Overview

The GLM family includes open-weight models from Zhipu AI. GLM-4.7 and GLM-5 are dense models; GLM-5.1 is a Mixture-of-Experts model.

| Model | Params | Context | Architecture | HuggingFace |
|---|---|---|---|---|
| GLM-4.7 | — | — | Dense | [zai-org/GLM-4.7](https://huggingface.co/zai-org/GLM-4.7) |
| GLM-5 | — | — | Dense | [zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5) |
| GLM-5.1 | 754B | 202K | MoE (glm_moe_dsa) | [zai-org/GLM-5.1](https://huggingface.co/zai-org/GLM-5.1) |

All three models share the same chat template format.

**Official resources:**
- HuggingFace org: https://huggingface.co/zai-org
- Chat template: `chat_template.jinja` in HuggingFace repo
- License: MIT
- Languages: English, Chinese

## Trained Tokens

The tokenizer config registers these as special tokens (in `extra_special_tokens`):

| Token | Purpose | Type |
|---|---|---|
| `<|endoftext|>` | EOS and PAD token | Special token |
| `[MASK]` | Masking token | Special token |
| `[gMASK]` | Generation mask (always emitted at the start of every prompt) | Special token |
| `[sMASK]` | Span mask token | Special token |
| `<sop>` | Start of prompt (always emitted right after `[gMASK]`) | Special token |
| `<eop>` | End of prompt | Special token |
| `<|system|>` | System role marker | Special token |
| `<|user|>` | User role marker | Special token |
| `<|assistant|>` | Assistant role marker | Special token |
| `<|observation|>` | Tool result / observation role marker | Special token |
| `<|begin_of_image|>` / `<|end_of_image|>` | Image block boundaries | Special token |
| `<|begin_of_video|>` / `<|end_of_video|>` | Video block boundaries | Special token |
| `<|begin_of_audio|>` / `<|end_of_audio|>` | Audio block boundaries | Special token |
| `<|begin_of_transcription|>` / `<|end_of_transcription|>` | Transcription block boundaries | Special token |

These tokens appear in the chat template but are NOT in `extra_special_tokens`. They are likely single tokens in the base BPE vocabulary (need to inspect `tokenizer.json` to confirm):

| Token | Purpose | Type |
|---|---|---|
| `<think>` | Begins a reasoning block | Likely base vocab token |
| `</think>` | Ends a reasoning block (also serves as a "no thinking" marker) | Likely base vocab token |
| `<tool_call>` / `</tool_call>` | Wraps a tool call | Likely base vocab token |
| `<arg_key>` / `</arg_key>` | Wraps a tool call argument key | Likely base vocab token |
| `<arg_value>` / `</arg_value>` | Wraps a tool call argument value | Likely base vocab token |
| `<tool_response>` / `</tool_response>` | Wraps a tool result | Likely base vocab token |
| `<tools>` / `</tools>` | Wraps a tool schema list (in system prompt and tool reference results) | Likely base vocab token |

**Token type definitions:**
- **Special token**: Single token in the vocabulary, suppressed during generation (the model cannot produce these)
- **Likely base vocab token**: Not in `extra_special_tokens` but used as if atomic in the template; likely a single token in the base BPE vocabulary
- **Text marker**: A string rendered by the template but tokenized as regular text (potentially multiple tokens)

## Chat Format

GLM uses standard angle-bracket role markers. Every prompt starts with `[gMASK]<sop>`.

### Prefix (always present)

```
[gMASK]<sop>
```

### System Message

```
<|system|>{system_content}
```

A trailing newline follows the content.

### User Message

```
<|user|>{user_content}
```

No trailing newline. The next message's role token immediately follows.

### Assistant Message (no thinking, no tools)

```
<|assistant|>
</think>{content}
```

Note: even when there is no thinking, a closing `</think>` is emitted with no opening `<think>`. This is the model's signal for "no reasoning content for this turn."

### Assistant Message (with thinking)

```
<|assistant|>
<think>{reasoning_content}</think>{content}
```

### Assistant Message (with tool calls)

```
<|assistant|>
</think>{content}
<tool_call>{name}<arg_key>{k1}</arg_key><arg_value>{v1}</arg_value><arg_key>{k2}</arg_key><arg_value>{v2}</arg_value></tool_call>
```

If there is no leading content, the content line is empty. Multiple tool calls each get their own `<tool_call>...</tool_call>` block.

### Tool Result Messages

A run of consecutive `tool` role messages opens with `<|observation|>` once (only on the first tool message in the run), followed by one `<tool_response>...</tool_response>` per result:

```
<|observation|><tool_response>{result_1}</tool_response><tool_response>{result_2}</tool_response>
```

Special case: if a tool message contains a list of `tool_reference` items (used for dynamically loading tools that were marked `defer_loading`), the result renders as a tool schema list:

```
<|observation|><tool_response><tools>
{"name":"foo","description":"...","parameters":{...}}
{"name":"bar","description":"...","parameters":{...}}
</tools></tool_response>
```

### Generation Prompt

Appended when `add_generation_prompt=true`:

```
<|assistant|><think>
```

Or when `enable_thinking=false`:

```
<|assistant|></think>
```

## Think Format

GLM uses `<think>...</think>` to wrap reasoning content. Two unusual behaviors:

### "No thinking" is rendered as a bare close tag

When an assistant message has no reasoning content but thinking mode is active, the template emits a bare `</think>` with no preceding `<think>`. This serves as the model's signal that this turn skipped the reasoning step.

### History thinking is dropped

The template tracks which user turns were followed by an assistant message with `reasoning_content` (via the `thinking_indices` namespace variable). For older assistant messages whose corresponding user turn had thinking, the reasoning content is dropped and replaced with empty (just `<think></think>`). Only the most recent assistant message (after the last user turn) keeps its full reasoning.

The `clear_thinking=false` option preserves all history reasoning if set.

## Tool Call Format

### Tool Definitions

When `tools` are provided, a system message is prepended to the conversation BEFORE any other system message:

```
<|system|>
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"name":"get_weather","description":"Get weather","parameters":{...}}
{"name":"search","description":"Search","parameters":{...}}
</tools>

For each function call, output the function name and arguments within the following XML format:
<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value></tool_call>
```

Tools with `defer_loading: true` are skipped from this initial declaration (loaded later via tool_reference responses). The `strict` and `defer_loading` keys are filtered out of the JSON.

### Tool Call Format

```
<tool_call>{function_name}<arg_key>{key1}</arg_key><arg_value>{value1}</arg_value><arg_key>{key2}</arg_key><arg_value>{value2}</arg_value></tool_call>
```

- Argument values are rendered as-is if they are strings, otherwise JSON-encoded
- Multiple `<arg_key>`/`<arg_value>` pairs alternate
- A function with no arguments has nothing between the function name and `</tool_call>`

### Tool Result Format

See "Tool Result Messages" in the Chat Format section.

## Full Example

Conversation:
- system: "You are a helpful assistant."
- user: "What's the weather in SF?"
- assistant (thinking="Need to call get_weather", tool_calls=[{name: "get_weather", args: {city: "SF"}}])
- tool: "65F sunny"
- assistant (thinking="Got the weather, now answer", content="It is 65F and sunny in SF.")

Rendered with tools=[get_weather schema]:

```
[gMASK]<sop><|system|>
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>

{"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}

</tools>

For each function call, output the function name and arguments within the following XML format:
<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value></tool_call><|system|>You are a helpful assistant.
<|user|>What's the weather in SF?<|assistant|>
<think>Need to call get_weather</think>
<tool_call>get_weather<arg_key>city</arg_key><arg_value>SF</arg_value></tool_call>
<|observation|><tool_response>65F sunny</tool_response><|assistant|>
<think>Got the weather, now answer</think>It is 65F and sunny in SF.
```

(Generation prompt would be `<|assistant|><think>` if continuing.)
