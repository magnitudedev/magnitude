# Serving

**The host owns transport, rendering and parsing; a worker thread owns the engine.
Tokens, options and snapshots cross between them, and nothing else does.**

## Process

```text
HTTP ──► app ──► chat service ──► worker (one thread) ──► runtime ──► engine + model + device
             template · parser · stop text            │
             per request, on the host                 └── built from one composition; reported with its digest
```

| Crosses the worker boundary | Never crosses |
|---|---|
| Rendered tokens and generation options, in | A tokenizer, a template, a parser |
| Published tokens and a snapshot, out | A sequence, a tensor, a ticket, a device object |
| Stop and release, in | A chat format or an input plan's meaning |

The worker wakes on native completion or on a control job, never on a timer, and
completion delivery is reserved so a saturated control queue cannot block it.

## A request

```text
validate wire body ── reject unsupported policies before admission
render with the artifact's own template ── tools, choice, parallelism ── must fit the context
admit ──► id
receive ──► future ◄── publication of one token, or a finish, per delivery
decode text incrementally ──► content / reasoning / tool-call events ──► SSE chunks
```

| Rule | Reason |
|---|---|
| Parsing is chunk-invariant | The same token stream yields the same events however it is split |
| String stops use bounded lookbehind | A stop may straddle chunks; the buffer that catches it is bounded by the longest stop |
| Tool arguments decode separately from framing | A wire format is a marker convention; argument typing is a schema concern |
| Disconnect releases the receiver and cancels the request | Submitted work completes; accepted terminal output has an explicit discard owner, so a departed client strands no credit |
| Architecture selection is the runtime adapter's | The scheduler and transport never see a Qwen input plan |

## Properties

The server reports what it is running: the composition as data, its digest, the
artifact identity, and the limits it was configured with. Two servers with equal
digests are running the same construction; what they selected at plan time is
reported by measurement, not here.
