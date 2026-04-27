# Kimi Model Family

## Overview

The Kimi family includes open-weight models from Moonshot AI. Both are large MoE models with vision capabilities (image-text-to-text).

| Model | Params (Total/Active) | Context | Architecture | HuggingFace |
|---|---|---|---|---|
| Kimi K2.5 | ~1T / — | — | MoE (kimi_k25) | [moonshotai/Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) |
| Kimi K2.6 | — | — | MoE | [moonshotai/Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6) |

Both models share the same chat template format.

**Official resources:**
- HuggingFace org: https://huggingface.co/moonshotai
- Chat template: `chat_template.jinja` in HuggingFace repo
- Paper: [arxiv.org/abs/2602.02276](https://arxiv.org/abs/2602.02276)
- License: Modified MIT
- Tokenizer: Custom TikToken-based (`tokenization_kimi.py`)

## Trained Tokens

| Token | Value | Purpose | Type |
|---|---|---|---|
| BOS | `[BOS]` | Beginning of sequence | Special token |
| EOS | `[EOS]` | End of sequence | Special token |
| PAD | `[PAD]` | Padding token | Special token |
| UNK | `[UNK]` | Unknown token | Special token |
| Message end | `<\|im_end\|>` | Ends a message/turn | Special token |
| User role | `<\|im_user\|>` | Opens a user message | Special token |
| Assistant role | `<\|im_assistant\|>` | Opens an assistant message | Special token |
| System role | `<\|im_system\|>` | Opens a system message | Special token |
| Role-content separator | `<\|im_middle\|>` | Separates role name from content | Special token |
| Tool calls section open | `<\|tool_calls_section_begin\|>` | Opens a tool calls block | Added token (not special) |
| Tool calls section close | `<\|tool_calls_section_end\|>` | Closes a tool calls block | Added token (not special) |
| Tool call open | `<\|tool_call_begin\|>` | Opens a single tool call | Added token (not special) |
| Tool call argument open | `<\|tool_call_argument_begin\|>` | Opens the argument block within a tool call | Added token (not special) |
| Tool call end | `<\|tool_call_end\|>` | Closes a single tool call | Added token (not special) |
| Thinking start | `	intent` | Begins a reasoning/thinking block | Added token (not special) |
| Thinking end | `	comment` | Ends a reasoning/thinking block | Added token (not special) |
| Media begin | `<\|media_begin\|>` | Opens a media block (image/video) | Special token |
| Media content | `<\|media_content\|>` | Marks media content area | Special token |
| Media end | `<\|media_end\|>` | Closes a media block | Special token |
| Media pad | `<\|media_pad\|>` | Placeholder for media data | Special token |
| Video placeholder | `<\|kimi_k25_video_placeholder\|>` | Video placeholder (K2.5 specific) | Text marker |

**Token type definitions:**
- **Special token**: Single token in the vocabulary, suppressed during generation (the model cannot produce these)
- **Added token (not special)**: Single token added to the vocabulary, but the model CAN generate these (important for parsing model output like tool calls and thinking)
- **Text marker**: A string rendered by the template but tokenized as regular text (potentially multiple tokens)

**Note on special vs added tokens**: The role markers (`<|im_user|>`, `<|im_assistant|>`, etc.) and media tokens are true special tokens — the model cannot generate them. The tool call tokens and thinking tokens (`	intent`, `	comment`) are added to the vocabulary but NOT marked as special, meaning the model can and does generate them during output.

## Chat Format

Kimi uses a role-prefix format where each message is structured as:

```
<|im_{role}|>{name}<|im_middle|>
{content}
<|im_end|>
```

The role name can be customized via the `name` field on each message. If not provided, it defaults to the role string itself (e.g., "user", "assistant", "system").

**Simple conversation (no tools, no thinking):**

```
<|im_system|>system<|im_middle|>
You are a helpful assistant.<|im_end|>
<|im_user|>user<|im_middle|>
What is 2+2?<|im_end|>
<|im_assistant|>assistant<|im_middle|>
	intent	comment
4<|im_end|>
```

Note: In history messages (before the last non-tool-call assistant), thinking is always rendered as `	intent	comment` with no reasoning content between them.

## Think Format

Kimi supports thinking mode with `	intent`/`	comment` tokens.

**Key behavior — split rendering of thinking:**

The template finds the **last non-tool-call assistant message** and splits all messages into two groups:
- **History messages** (before and including that last non-tool-call assistant): thinking/reasoning content is **always dropped**. Only `	intent	comment` is rendered before the content.
- **Suffix messages** (after that last non-tool-call assistant): reasoning content is **preserved**. The full `	intent{reasoning}	comment{content}` is rendered.

This means in a typical multi-turn conversation, only the most recent assistant turn keeps its thinking visible.

**When `thinking=false` is passed**, even suffix messages drop the reasoning content — only `	intent	comment` is emitted.

**Assistant with thinking (suffix message):**
```
<|im_assistant|>assistant<|im_middle|>
	intent
I need to think about this step by step...
	comment
Here is my answer.
<|im_end|>
```

**Assistant with thinking (history message — reasoning dropped):**
```
<|im_assistant|>assistant<|im_middle|>
	intent	comment
Here is my answer.
<|im_end|>
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

Rendered:
```
<|im_system|>system<|im_middle|>
You are a helpful assistant.<|im_end|>
<|im_user|>user<|im_middle|>
Hi<|im_end|>
<|im_assistant|>assistant<|im_middle|>
	intent	comment
Hello!<|im_end|>
<|im_user|>user<|im_middle|>
What's 2+2?<|im_end|>
<|im_assistant|>assistant<|im_middle|>
	intent
Simple math.
	comment
4<|im_end|>
```

## Tool Call Format

### Tool Definitions

Tools are declared in a special system message at the start:

```
<|im_system|>tool_declare<|im_middle|>
[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{...}}}]<|im_end|>
```

The tools JSON is serialized compactly. If a `tools_ts_str` variable is provided, it's used instead (pre-serialized TypeScript string — the repo includes `tool_declaration_ts.py` for this).

### Tool Calls

When an assistant makes tool calls, they're rendered after the content:

```
<|im_assistant|>assistant<|im_middle|>
	intent
I need to check the weather.
	comment
<|tool_calls_section_begin|>
<|tool_call_begin|>call_abc123<|tool_call_argument_begin|>{"city": "San Francisco"}<|tool_call_end|>
<|tool_calls_section_end|>
<|im_end|>
```

Multiple tool calls in one message each get their own `<|tool_call_begin|>...<|tool_call_end|>` within the section block. Each includes an `id` and the arguments as a JSON string.

### Tool Results

Tool results use a markdown-style format:

```
## Return of call_abc123
The weather is 65°F and sunny.
```

The tool call ID is referenced in the heading. The content follows.

### Full Example with Tools and Thinking

```
<|im_system|>tool_declare<|im_middle|>
[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}]<|im_end|>
<|im_system|>system<|im_middle|>
You are a helpful assistant.<|im_end|>
<|im_user|>user<|im_middle|>
What's the weather in SF and NYC?<|im_end|>
<|im_assistant|>assistant<|im_middle|>
	intent
I need to check the weather for both cities.
	comment
<|tool_calls_section_begin|>
<|tool_call_begin|>call_001<|tool_call_argument_begin|>{"city": "San Francisco"}<|tool_call_end|>
<|tool_call_begin|>call_002<|tool_call_argument_begin|>{"city": "New York"}<|tool_call_end|>
<|tool_calls_section_end|>
<|im_end|>
<|im_tool|>tool<|im_middle|>
## Return of call_001
65°F, sunny
<|im_end|>
<|im_tool|>tool<|im_middle|>
## Return of call_002
45°F, rainy
<|im_end|>
<|im_assistant|>assistant<|im_middle|>
	intent
Now I have both results. SF is warm, NYC is cold.
	comment
SF is 65°F and sunny, NYC is 45°F and rainy.
<|im_end|>
```

### Generation Prompt

```
<|im_assistant|>assistant<|im_middle|>
	intent
```

When `thinking=false`:
```
<|im_assistant|>assistant<|im_middle|>
	intent	comment
```
