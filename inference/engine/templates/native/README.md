# Native template extraction

This directory owns the C++ template renderer and semantic-output parser linked into
`magnitude-templates`. Cargo compiles the explicit translation-unit list in `build.rs`; building
the crate does not run CMake, Python, `patch`, or consult another Magnitude source tree.

`source/` is the required subset of the patched extraction produced from llama.cpp revision
`930e2fa5995789efbf249a8bf61325bb626e417b`. The original file hashes and upstream repository are
recorded in `provenance/manifest.json`, and `provenance/patches/` records the ordered transformations
applied to those sources. `LICENSE.llama.cpp` is the upstream license. `src/` and `include/` are
Magnitude's owned C ABI boundary. Model-template fixtures used to verify the generic API live under
`tests/assets/`; they are test inputs, not runtime configuration or numerical-model dependencies.

The series has no patch 0009: it once defaulted `additionalProperties` to allowed for objects that
list `properties`, and was withdrawn to keep the upstream converter semantics the previous service
enforced (an object listing properties is closed unless the schema explicitly allows more).

Patch 0019 makes schema lowering total: every valid JSON Schema lowers, and keywords a grammar
cannot enforce are loosened and recorded instead of rejected. It supersedes the rejections patches
0008 and 0010 introduced. It also restricts string grammars to valid JSON string text.

Patch 0020 makes the Gemma 4 tool grammar read one way: the text before the calls is the scan alone
(the parser's `message*` has no GBNF lookahead and overlaps it), or only a thought when a call is
required, and a dictionary key never starts with the whitespace its separator owns. Patch 0021
makes a required tool choice follow reasoning directly in every other format, with no content
before the call, and makes Qwen3-Coder-format grammars read reasoning the generation prompt opens
only as reasoning, so it closes before the turn ends.

Patch 0022 gives Muse Glimmer a way to disable reasoning. Its template has no switch: the system
message carries a reasoning strength and the model chooses its recipient after `<|start|>assistant`.
With `enable_thinking` false the strength is none unless the request names one, and the reply is
opened to the user (` to=user<|message|>`); with callable tools the recipient stays the model's choice.

Patch 0023 makes MiniCPM5 and LFM2 tool-call text read one way when a free-form argument makes
the tool grammar recursive. Text the parser reads another way by ordered choice or negative
lookahead (a MiniCPM5 value opening `<![CDATA[`, text opening `<think>` before content) has an exact
GBNF form instead of being dropped, so content that no reasoning precedes never opens with the
reasoning tag, and reasoning the LFM2.5 prompt opens closes before the turn ends. Whitespace has one
owner: a MiniCPM5 argument owns the whitespace after it, and Python-style values leave whitespace to
their separators.

Patch 0024 makes the key grammar of an object's additional properties valid JSON string text: it
spells keys other than the listed properties as escaped JSON text, never as raw characters a JSON
string cannot hold.

Patch 0025 makes the LFM2 handler report reasoning, and parse emitted `<think>` blocks, exactly
when the template names a reasoning block: it opens `<think>`, or it strips a closing `</think>`
from history as LFM2.5 1.2B does while leaving the opening tag to the model. A template with
neither tag declares no thinking support, no reasoning markers and no reasoning continuation, and
reasoning detection classifies it as non-reasoning rather than fixed reasoning.

Patch 0026 makes every template's tool format yield a grammar the compiler accepts, or no tool
support. LFM2.5 templates are recognized by their `<|tool_call_start|>[...]<|tool_call_end|>` call
markup, so finetunes that write `"List of tools: " + tools | tojson` keep the LFM2.5 handler, or by
the `"List of tools: ["` literal of a template that leaves calls to the model, as LFM2.5 1.2B does. A format the generic analysis cannot
separate from its probe calls (a marker holding a probe name or value, or a tagged argument value
with no closing marker) is no tool format: tools render into the prompt, the reply reads as content,
and no tool grammar is built. A rule matching only the empty string is spelled `""`, since a bare
trailing `|` continues the alternation onto the next rule.

Patch 0027 makes a lenient (streaming) parse decide nothing at the end of its input, so the stream's
published output is monotonic for every handler at once. A parser that reaches the end of the input
before it can decide needs more input: `until` at a partial delimiter (upstream succeeded there, so
the next parser read the delimiter's bytes, and Kimi K2 published `<|` in reasoning as content),
`end`, and `space`, `chars` and repetitions that could continue. The text before the undecided point
is still read; strict parses are unchanged.

Patch 0028 makes a loop item cost its body. Upstream copied the whole enclosing scope for every item
of every loop (a scope only a loop filter reads), copied the table of builtins on every name lookup
and builtin call, and compared string keys by concatenating them. Templates that scan the rest of
the conversation for each message (Gemma 4) run a number of items quadratic in its length, so these
costs multiplied with it. Rendered output is unchanged.

Patch 0029 renders the prompt once. Upstream rendered the conversation three times per
preparation: the prompt, then the conversation without and with the generation prompt to find the
generation prompt. Those two renders differ from the prompt's only in `add_generation_prompt` and
the continuation, and rendering does not read the continuation, so one of them is the prompt
itself. Every family takes both from `common_chat_template_render_prompt`, which renders the other
side alone. Prompts and generation prompts are unchanged. A preparation describes how many renders
it made.

Patch 0030 reads DeepSeek V4.1's tool calls, as upstream's `74bca3950` (llama.cpp v0.9.0) does: a
template that writes `｜DSML｜ calls>` selects the DeepSeek handler, whose call block, invoke and
parameter tags are then V4.1's, each name after a space (`<｜DSML｜ calls>`). Only its parsing is
ported; its other V4.1 parts serve request fields the extraction never receives. For V4.1, as for V4, a
JSON-schema reply is the value in a ```` ```json ```` fence.

Patch 0031 completes the comparison filters as Jinja defines them: `unique` keeps each item whose
key is new, in order; `sort`, `dictsort`, `min`, `max` and `unique` compare a string by its lower
case unless `case_sensitive` (upstream ignored the argument and always compared exactly), take the
key from `attribute` where Jinja does, and sorts are stable in either direction.

Patch 0032 gives each iteration of a loop body its own scope, as Jinja does: an assignment in the
body is undone when its iteration ends, so the next iteration sees the value from before the loop
(upstream carried it over). The undo costs only the iteration's own assignments, never the scope's
size.

Patch 0033 makes a Gemma 4 call opener close the thought it interrupts when tools are callable. The
tool grammar ends the text before the calls at the first `<|tool_call>`, inside a thought or not, and
a reply to a tool result starts inside the thought its generation prompt opens. The parser read
such a thought on to the next `<channel|>`: it streamed the call as reasoning, and when the call's
arguments held `<channel|>`, read the rest of them as content and a later call opener in them as a
call, then retracted that call; a thought that never closed lost its calls instead. The thought now
ends at `<channel|>` or a call opener, as Qwen3-Coder's reasoning does, so the parser reads the
grammar's text before the calls one way.

Patch 0034 opens a DeepSeek reply as its template renders it. The template opens a reply to the
user with `<｜Assistant｜>` and a reply to tool results with `\n\n`, each followed by the reasoning
tag; the handler expected `<｜Assistant｜>` in both, so every reply to tool results failed on its
first byte. The handler now takes the opener from the rendered generation prompt, before its
reasoning tag, and a continued reply opens the same way.

Patch 0035 makes DeepSeek's tool grammar and parser read the same text. With tools, the parser
read reasoning the generation prompt opens as optional, and refused content beginning with
`<think>` by lookahead, which has no GBNF form: the grammar admitted reasoning closed by `</think>`
and content beginning with `<think>`, or reasoning skipped and the prompt's `<think>` read as
content, which the parser then refused or read again as reasoning. Reasoning the generation
prompt opens is now only reasoning, ending at `</think>` or at the calls block, as Qwen3-Coder's
does, and content may begin with anything.

Patch 0036 makes Kimi K2's tool grammar and parser read the same text. Its reasoning ends at
`</think>` or at a call opener, and the closing `</think>` was optional, so the grammar let
reasoning end anywhere and content take the rest: a call opener split between them was admitted
without a call after it, and the parser, reading the whole opener, refused the text. Reasoning now
closes with `</think>`, followed by content and calls, or runs directly into calls that begin with
their opener.

Patch 0037 makes the end of output end reasoning in every tool grammar and parser, unless a call is
required. Most tool grammars refused the end of output inside reasoning, so a model that stopped
there was made to keep reasoning; Kimi K2's grammar admitted it until patch 0036. The stream
publishes reasoning as it arrives, but the parsers read reasoning only once it closed, so a strict
reading of the whole output took it as content (Kimi K2, DeepSeek, Qwen3-Coder, LFM2, MiniCPM5,
MiniMax-M3, Ministral 3, generic formats), as untagged text (Gemma 4, Muse Glimmer, GPT-OSS) or
refused it (Cohere2 MoE), and only the stream's end-of-output fallback kept the published reading.
Reasoning that never closes is now read to the end of output as reasoning, with no content and no
calls, and the grammar admits that end as its last alternative, so no text follows it. Where content
may begin with the reasoning tag, or the turn needs a body after reasoning (DeepSeek reasoning the
model opens, LFM2, MiniCPM5, MiniMax-M3, Ministral 3, Cohere2 MoE, generic formats), that reading is
tried first. Kimi K3 already read and admitted it, and Functionary v3.2 and GigaChat v3 have no
reasoning. Chat turns, which have no grammar, still rely on the stream: it also keeps the published
reading when output ends between sections (after a closed GPT-OSS or Muse Glimmer analysis, after
Cohere2 MoE reasoning with no text) or inside constrained output a stop string cut short.

Patch 0038 completes the Jinja2 and HuggingFace `transformers` chat-template features missing from
upstream. It adds the standard global builtins (`dict`, `zip`, `enumerate`, `len`, `str`, `int`,
`float`, `bool`, `list`, `min`, `max`, `sum`, `abs`, `round`, `sorted`, `reversed`, `any`, `all`),
separates filter dispatch from member lookup on objects so filters such as `default` and `length`
never shadow dict keys like `param.default`, implements `tojson(sort_keys=...)`, `from_json`,
distinct `select`/`reject` vs `selectattr`/`rejectattr`, the `le` test, `is iterable` on dicts,
`sum`/`list`/`replace(..., count)`/`str.join`, dict/string/list methods (`update`, `pop`,
`setdefault`, `copy`, `clear`, `find`, `rfind`, `splitlines`, `removeprefix`, `removesuffix`,
`count`, tuple `startswith`/`endswith`, whitespace `split`/`rsplit`, negative string indexing,
`extend`, `insert`, `remove`, `index`), `{% do ... %}` (`jinja2.ext.do`), and `{% for ch in str %}`.

Patch 0039 closes an open tag-based reasoning block in the differential autoparser when
`enable_thinking` is explicitly false. Templates that unconditionally open a reasoning tag in
`add_generation_prompt` without inspecting `enable_thinking` (such as GLM-5.3-Flash, which always
ends with `<|assistant|><think>` while rendering non-thinking history turns as `<think></think>`)
now append the closing reasoning tag (`</think>`) to `prompt` and `generation_prompt` when
`enable_thinking` is false. When `enable_thinking` is false and the reasoning block is already
closed in `generation_prompt`, the parser matches the entire closed prefix literally with reasoning
extraction disabled so no whitespace inside the empty block leaks into `reasoning_content` and
constrained grammars begin directly after the closed block. The generic autoparser declares
`enable_thinking` in `handler_context_variables` so pruned and unpruned reasoning discovery agree.

Patch 0040 brings the same thinking-disabled behavior to the Qwen3-Coder (used by LFM3, Qwen3.5,
Qwen3.6, and Nemotron) and LFM2 specialized handlers. In `qwen3-coder.cpp`, an unclosed `<think>`
left at the end of `generation_prompt` when `enable_thinking` is false is closed with `</think>`,
`thinking_opened` checks for an unclosed `<think>` rather than any `<think>` substring (which
previously misfired on closed `<think></think>` prefixes), and when `enable_thinking` is false with
a closed `<think>...</think>` prefix, both `qwen3-coder.cpp` and `lfm2.cpp` match the closed
`generation_prompt` literally with reasoning extraction disabled so plain text, tool-call, and
JSON-schema replies start directly after the closed block with zero `<think>` rules in the grammar.
`specialized_handler::qwen3_coder` declares `enable_thinking` in `handler_context_variables` so
pruned and unpruned reasoning discovery agree.

To refresh the extraction, use the provenance record and patch series in a maintainer workflow,
then check in the resulting patched sources. Source preparation is deliberately not part of a
consumer build.
