---
applies_to:
  - inference-v2/src/magnitude_engine/serving/**
  - inference-v2/tests/serving/**
---

# Inference serving

The host owns HTTP request validation, checkpoint-specific prompt rendering,
text/tool parsing, and transport lifetime. The worker builds the engine blueprint
and owns model execution, state, scheduling, sampling and token constraints.
Rendered tokens and a constraint description cross the worker boundary; live
tokenizers, parsers and device objects do not.

Tool selection has one meaning in both the prompt and the output language. An
automatic selection offers the supplied tools. A named selection offers the
selected function; a required selection tells the model to call an available
function. A selection of none omits tools. Rendering communicates required/named
selection without mutating the caller's history; exact instruction wording is
not a performance guarantee. The checkpoint
still owns chat and argument syntax. Prompt instructions communicate intent;
request-local grammar enforces it. Template keyword overrides cannot replace
request-owned tool selection or other rendering inputs.

Stream parsing emits semantic text, reasoning and complete validated tool calls.
Client disconnection cancels its request; one request's cancellation does not
retire shared device work still used by peers. Terminal usage comes from actual
engine execution. Benchmark comparisons distinguish rendered inputs, generated
work, model service, and public latency; tool buffering can make first semantic
output arrive after the first model token.

Qualification covers immutable input history, automatic/required/named/disabled
tools, reasoning boundaries, grammar and schema enforcement, fragmented streams,
backpressure, cancellation, and consistency of terminal timing and token counts.
