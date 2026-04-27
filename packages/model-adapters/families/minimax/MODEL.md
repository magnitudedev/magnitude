# MiniMax Model Family

## Overview

The MiniMax family includes open-weight Mixture-of-Experts models from MiniMax AI.

| Model | Params (Total/Active) | Context | HuggingFace |
|---|---|---|---|
| MiniMax M2.5 | — | — | [MiniMaxAI/MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5) |
| MiniMax M2.7 | — | — | [MiniMaxAI/MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7) |

Both models share the same chat template format.

**Official resources:**
- HuggingFace collection: https://huggingface.co/MiniMaxAI
- Chat template: `chat_template.jinja` in HuggingFace repo

## Trained Tokens

| Token | Value | Purpose | Type |
|---|---|---|---|
| BOS | `]~!b[` | Marks the beginning of the entire input | Special token |
| Role prefix | `]~b]` | Opens a message block (followed by role name: `system`, `user`, `ai`, `tool`) | Special token |
| Turn close / EOS | `[e~[` | Ends a message/turn (also the EOS token) | Special token |
| UNK | `]!d~[` | Unknown token | Special token |
| Thinking start | `	intent` | Begins a reasoning/thinking block | Added token (not special) |
| Thinking end | `	comment` | Ends a reasoning/thinking block | Added token (not special) |
| Tool call open | `<minimax:tool_call>` | Opens a tool call section | Added token (not special) |
| Tool call close | `</minimax:tool_call>` | Closes a tool call section | Added token (not special) |
| Tool invoke | `<invoke name="...">` | Opens a tool invocation | Text marker |
| Tool invoke close | `</invoke>` | Closes a tool invocation | Text marker |
| Tool parameter | `<parameter name="...">value</parameter>` | A parameter within an invoke block | Text marker |
| Tool result | `<response>content</response>` | Wraps a tool result | Text marker |
| Tool schema | `<tool>{json}</tool>` | Wraps a tool schema definition | Text marker |
| Function call | `<function_call>` | General function call token | Special token |
| Code interpreter | `<code_interpreter>` | Code interpreter token | Special token |
| Image open | `]<]image[>[` | Opens an image block | Special token |
| Image close | `]<]start of image[>[` / `]<]end of image[>[` | Image boundaries | Special token |
| Video open | `]<]video[>[` | Opens a video block | Special token |
| Video close | `]<]start of video[>[` / `]<]end of video[>[` | Video boundaries | Special token |
| Vision pad | `]<]vision pad[>[` | Vision padding token | Special token |
| Speech tokens | `]<]speech[>[`, `]<]start of speech[>[`, `]<]end of speech[>[` | Speech/audio tokens | Special token |
| FIM tokens | `<fim_prefix>`, `<fim_middle>`, `<fim_suffix>`, `<fim_pad>` | Fill-in-the-middle tokens | Special token |
| Code tokens | `<jupyter_start>`, `<jupyter_code>`, `<jupyter_output>`, `<jupyter_error>`, `<jupyter_text>` | Jupyter code tokens | Special token |
| VCS tokens | `<commit_before>`, `<commit_msg>`, `<commit_after>`, `<commit_message>` | Version control tokens | Special token |
| File tokens | `<filename>`, `<reponame>`, `<add_file>`, `<delete_file>`, `<rename_file>`, `<edit_file>` | File operation tokens | Special token |

**Token type definitions:**
- **Special token**: Single token in the vocabulary, suppressed during generation (the model cannot produce these)
- **Added token (not special)**: Single token added to the vocabulary, but the model CAN generate these (important for parsing model output like tool calls and thinking)
- **Text marker**: A string rendered by the template but tokenized as regular text (potentially multiple tokens)

**Key observation**: The core structural tokens (`]~!b[`, `]~b]`, `[e~[`) that define the chat format are true special tokens. However, the thinking and tool call tokens (`	intent`, `	comment`, `<minimax:tool_call>`) are added tokens but NOT special — the model is expected to generate these as part of its output. The inner XML-like tags (`<invoke>`, `<parameter>`, `<response>`) are plain text that gets tokenized as regular subwords.

## Chat Format

The model uses MiniMax-specific control tokens rather than standard `<|im_start|>`/`<|im_end|>` markers. The `]~!b[`, `]~b]`, and `[e~[` tokens are structural delimiters that tell the model when a new speaker begins and ends.

### System Block (always first)

```
]~!b[]~b]system
{system_message_content}[e~[
```

**Default system message** (if none provided):
```
]~!b[]~b]system
You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax.[e~[
```

Can be overridden by passing a `model_identity` variable to the template.

**Optional system fields:**
- `current_date` — appends `Current date: {date}` on a new line
- `current_location` — appends `Current location: {location}` on a new line

### User Message

```
]~b]user
{content}[e~[
```

### Assistant Message (no thinking, no tools)

```
]~b]ai
{content}[e~[
```

### Full Simple Conversation

```
]~!b[]~b]system
You are a helpful assistant.[e~[
]~b]user
What is 2+2?[e~[
]~b]ai
4[e~[
```

## Think Format

MiniMax uses `	intent`/`	comment` tokens for thinking/reasoning.

**Interleaved thinking**: The template finds the last user message index. For assistant messages at or after the last user message (`loop.index0 > ns.last_user_index`), thinking is preserved. For earlier assistant turns, the thinking content is silently dropped — only the summary content is shown.

```
]~b]ai
	intent
{reasoning_content}
	comment

{content}[e~[
```

When thinking is dropped for a historical turn:
```
]~b]ai
{content}[e~[
```

**Generation prompt** always starts with thinking enabled:
```
]~b]ai
	intent
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
]~!b[]~b]system
You are a helpful assistant.[e~[
]~b]user
Hi[e~[
]~b]ai
Hello![e~[
]~b]user
What's 2+2?[e~[
]~b]ai
	intent
Simple math.
	comment

4[e~[
```

## Tool Call Format

### Tool Definitions

If tools are provided, they are embedded INSIDE the system block, before the `[e~[` closer:

```
]~!b[]~b]system
{system_message_content}

# Tools
You may call one or more tools to assist with the user query.
Here are the tools available in JSONSchema format:

<tools>
<tool>{"name": "get_weather", "description": "Get weather", "parameters": {...}}</tool>
<tool>{"name": "search", "description": "Search the web", "parameters": {...}}</tool>
</tools>

When making tool calls, use XML format to invoke tools and pass parameters:

<minimax:tool_call>
<invoke name="tool-name-1">
<parameter name="param-key-1">param-value-1</parameter>
<parameter name="param-key-2">param-value-2</parameter>
...
</invoke>
</minimax:tool_call>[e~[
```

Each tool's `function` object is serialized as JSON and wrapped in `<tool>...</tool>` tags. The template includes an example showing the `<minimax:tool_call>` / `<invoke>` / `<parameter>` format.

### Tool Calls

```
]~b]ai
{content if any}
<minimax:tool_call>
<invoke name="get_weather"><parameter name="city">San Francisco</parameter></invoke>
</minimax:tool_call>[e~[
```

Multiple tool calls in one assistant message:
```
]~b]ai
<minimax:tool_call>
<invoke name="get_weather"><parameter name="city">San Francisco</parameter></invoke>
<invoke name="get_weather"><parameter name="city">New York</parameter></invoke>
</minimax:tool_call>[e~[
```

For each tool call:
- `tool_call.function` is unwrapped to get `name` and `arguments`
- Arguments are iterated as key-value pairs
- String values rendered as-is; non-string values are JSON-serialized

### Tool Results

Consecutive tool messages are **merged** into a single `]~b]tool` block. The opening `]~b]tool` and closing `[e~[` are only rendered for the first and last in a consecutive sequence.

**Single tool result (string content):**
```
]~b]tool
<response>{content}</response>[e~[
```

**Multiple tool results (array content):**
```
]~b]tool
<response>{result_1}</response>
<response>{result_2}</response>[e~[
```

**Consecutive tool messages (merged):**
```
]~b]tool
<response>65°F and sunny</response>
<response>45°F and rainy</response>[e~[
```

The template validates that tool messages are preceded by an assistant message with tool calls — otherwise it raises an exception.

### Full Example with Tools and Thinking

```
]~!b[]~b]system
You are a helpful assistant.

# Tools
You may call one or more tools to assist with the user query.
Here are the tools available in JSONSchema format:

<tools>
<tool>{"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}</tool>
</tools>

When making tool calls, use XML format to invoke tools and pass parameters:

<minimax:tool_call>
<invoke name="tool-name-1">
<parameter name="param-key-1">param-value-1</parameter>
<parameter name="param-key-2">param-value-2</parameter>
...
</invoke>
</minimax:tool_call>[e~[
]~b]user
What's the weather in SF and NYC?[e~[
]~b]ai
	intent
I need to check the weather for both cities.
	comment

<minimax:tool_call>
<invoke name="get_weather"><parameter name="city">San Francisco</parameter></invoke>
<invoke name="get_weather"><parameter name="city">New York</parameter></invoke>
</minimax:tool_call>[e~[
]~b]tool
<response>65°F, sunny</response>
<response>45°F, rainy</response>[e~[
]~b]ai
	intent
Now I have both results. SF is warm, NYC is cold.
	comment

SF is 65°F and sunny, NYC is 45°F and rainy.[e~[
```
