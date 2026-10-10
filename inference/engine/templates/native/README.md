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

To refresh the extraction, use the provenance record and patch series in a maintainer workflow,
then check in the resulting patched sources. Source preparation is deliberately not part of a
consumer build.
