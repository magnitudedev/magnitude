# Serving

**The host owns transport, rendering, media preparation and parsing; a worker
thread owns the engine and token-level constraint matchers. Immutable inputs,
constraint plans, options and snapshots cross between them.**

## Process

```text
HTTP ──► app ──► chat service ──► worker (one thread) ──► runtime ──► engine + model + device
             template · parser · stop text            │
             bounded media preparation                └── built from one composition; reported with its digest
             per request, on the host
```

| Crosses the worker boundary | Never crosses |
|---|---|
| Rendered tokens, prepared media tensors, generation options and symbolic constraint plans, in | A live tokenizer, template, parser, matcher or media processor |
| Published tokens and a snapshot, out | A sequence, a tensor, a ticket, a device object |
| Stop and release, in | A chat format or an input plan's meaning |

The worker wakes on native completion or on a control job, never on a timer, and
completion delivery is reserved so a saturated control queue cannot block it.

## A request

```text
validate wire body ── reject unsupported policies before admission
resolve and prepare bounded media with the artifact's processor
render with the artifact's own template ── tools, media spans, choice, parallelism ── must fit the context
admit ──► id
receive ──► future ◄── publication of one token, or a finish, per delivery
decode text incrementally ──► content / reasoning / tool-call events ──► SSE chunks
```

Template selection uses the operator override, then the `tool_use` variant when
tools are effectively offered, then the declared default. Disabling tools removes
them before selection. A missing default or unknown explicit variant fails.
Directory sources override configuration per variant: processor configuration,
tokenizer configuration, named files, then the default file. In GGUF, the
unsuffixed template overrides a named default. Selected provenance is retained.
Special-token spellings come from the artifact, independently of tokenizer-family
support.

One native preparation determines the prompt, generation prefix, parser and
grammar. Omitted reasoning controls preserve the template's authored default;
explicit unsupported effort levels and conflicting raw controls fail. The worker
compiles the symbolic grammar against its exact tokenizer before allocating model
input or queueing generation. Each request owns an independent matcher.

Grammar compilation preserves the complete accepted language. A wholly regular
completion may use one lexer expression; replacing only a recursive region with
a greedy lexer expression must not change delimiter boundaries. Nonregular
grammars retain their parser structure. Compilation and profile caches are bounded
and scoped to the exact selected source, options, tokenizer and implementation.

| Rule | Reason |
|---|---|
| Parsing is chunk-invariant | The same token stream yields the same events however it is split |
| String stops use bounded lookbehind | A stop may straddle chunks; the buffer that catches it is bounded by the longest stop |
| Tool arguments decode separately from framing | A wire format is a marker convention; argument typing is a schema concern |
| Tool starts and argument deltas are append-only | Streaming and complete responses preserve the same call IDs and argument text |
| Natural completion and truncation are distinct parser outcomes | Length limits and user stops cannot fabricate a completed structured value |
| Disconnect releases the receiver and cancels the request | Submitted work completes; accepted terminal output has an explicit discard owner, so a departed client strands no credit |
| Architecture selection is the runtime adapter's | The scheduler and transport never see a Qwen input plan |

## Properties

The server reports what it is running: the composition as data, its digest, the
artifact identity, and the limits it was configured with. Two servers with equal
digests are running the same construction; what they selected at plan time is
reported by measurement, not here.

Preparation reports template creation, effort probes, cache hits, rendering and
tokenization separately from worker service. Parsing reports input bytes and host
elapsed time. Terminal constraint metrics distinguish grammar preparation, masks,
and forced proposals; nested validation time is not added twice. These observations
do not substitute for measured prefill/decode wall-clock overhead.
