# Decisions

A decision model answers typed questions about a state (text, JSON or images) with a probability
distribution per question, read from one forward pass with no generated text. Use one for routing,
moderation, extraction, judging and visual checks: answers are fast, calibrated, and cost no output
tokens. Decision models are not chat models: chat routes, agents and harness connections can't use
them, and decision routes refuse chat models.

## Get one

A GGUF is a decision model when it carries a decision description: a decision template
(`tokenizer.chat_template.decision`) and the `decision.*` keys that select how the engine
renders, lays out and reads it. Magnitude's own conversions write it. The engine serves these
models:

| Models | Beyond noul/predicate, choice, score | Images |
| --- | --- | --- |
| vLLM Semantic Router d3-ultra | `box` | state, question instructions and options |
| Base Labs mammoth-1, elephant-1 | none | no |
| Perplexity pplx-decider Flash v1.2 | none | before the state |
| Liquid d1-2B | `multiple`, `box`, `boxes`, `text`, `number`, `date`, estimates | before the state (one for `box`/`boxes`) |

The catalog carries Magnitude's conversion of each (`magnitudedev/<name>-GGUF` on Hugging Face) in
Q4, Q5, Q6 and Q8, for example `d1-2b:gguf:q8`:

```sh
magnitude catalog list --serves decisions
magnitude catalog recommendations --serves decisions
magnitude catalog pull <model-id>
magnitude models load <model-id>
```

Any other decision GGUF is added as a custom model (see the `custom-models` topic):

```sh
magnitude catalog pull <owner>/<repo>[:QUANT]   # from Hugging Face
magnitude models folders add <path>             # or a folder holding the GGUF
magnitude models status --all
magnitude models load <model-id>
```

What a model serves comes from its metadata: a GGUF without a decision description is a chat model,
and one whose description the engine doesn't implement or that doesn't read is reported as
unsupported with the reason (the published `LiquidAI/d1-2B-GGUF` names llama.cpp's decision type and
carries no description, for example).

## Call it

Base URL: `http://127.0.0.1:10100/inference/v1` (any API key locally; the Network access key from
another device). `GET /inference/v1/models` reports decision models with
`architecture.output_modalities: ["decisions"]`.

TypeSafe System One, questions keyed by ID:

```sh
curl http://127.0.0.1:10100/inference/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "MODEL_ID",
  "state": "I was charged twice. Please refund one payment.",
  "questions": {
    "billing": {"type": "noul", "instructions": "Is this about billing?"},
    "route": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "shipping": "Delivery"}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Not urgent", "Urgent"]}
  }
}'
```

OpenAI Decisions, questions in order:

```sh
curl http://127.0.0.1:10100/inference/v1/decisions -H 'Content-Type: application/json' -d '{
  "model": "MODEL_ID",
  "input": "I was charged twice. Please refund one payment.",
  "questions": [
    {"type": "predicate", "name": "billing", "instructions": "Is this about billing?"},
    {"type": "choice", "name": "route", "instructions": "Which team?", "choices": [{"value": "billing"}, {"value": "shipping"}]},
    {"type": "score", "name": "urgency", "instructions": "How urgent?", "levels": [{"label": "Not urgent"}, {"label": "Urgent"}]}
  ]
}'
```

Both formats give identical probabilities. Neither streams. SDKs: TypeSafe's `TypeSafeClient`
with `base_url="http://127.0.0.1:10100/inference"` (no `/v1`) and `client.system_one(...)`; OpenAI's
client with the base URL above and `client.decisions.create(...)`.

## Answers

- Probabilities, a score's `score` and `confidence` are the model's own, as its reference computes
  them (d1-2B: the median level and the likeliest option's probability; mammoth, elephant and
  pplx-decider: the expected level and TypeSafe's formulas; d3-ultra: the expected level and one minus
  the normalized entropy). OpenAI fields follow OpenAI's definitions.
- Extended kinds (`multiple`, `box`, `boxes`, `text`, `number`, `date`, estimates) are System One only.
  A kind, option count or length the model doesn't serve is refused with a validation error naming the
  question.

## Images and limits

- Images are base64 data URLs: System One `assets` (or top-level `images`), OpenAI `input_image`
  parts. Hosted URLs are refused. No video or audio.
- System One: up to 255 options, 1 to 10 score levels (d3-ultra, pplx-decider and d1-2B: 2 or more), 32 MiB body. OpenAI: 1 to 400
  questions. The context limit is the model's.
