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

To refresh the extraction, use the provenance record and patch series in a maintainer workflow,
then check in the resulting patched sources. Source preparation is deliberately not part of a
consumer build.
