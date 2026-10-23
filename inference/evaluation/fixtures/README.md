# magnitude-fixtures

The only model inputs of evaluation: deterministic chat conversations in the fixture format of
`specs/26-10-08/evaluation-contracts.md`, written as JSON Lines fixture sets. `mrig` builds the
set an instrument's `--fixture` names on every run (it is deterministic) and stores it by its
digest (SHA-256 of the file bytes); the serving benchmark builds its own.

```bash
cd inference/evaluation
uv run pytest fixtures
```

`count` is the number of independent sessions (lanes, default 8); `queries` is the number of
keys a retrieval probe asks for (default 4).

```python
from magnitude_fixtures import build, read_set, write_set, validation

fixtures = build("prose-repeat", count=4)  # list[Fixture], from pinned sources
digest = write_set(path, fixtures)
fixtures, digest = read_set(path)
sized = fixture.sized(16_384, rendered_length)  # rendered_length(messages, tools) -> int
sized.messages, sized.tools, sized.prefix, sized.scored  # serve messages[:prefix] with tools
fixture.conversation_with(history=4)  # head + the 4 newest history units + probe
fixture.units()  # message ranges of the sizing turns; the last is the probe
validation.tool_calls(fixture.generation.expected, calls)
validation.retrieval(fixture.generation.expected, text)
```

## Shape shared by every family

Each fixture is one session: a kept head, history turns, then one **probe** turn last.
- Only the probe's assistant message is `scored`, so quality positions sit at the deepest
  point of the context and their count does not grow with context length.
- `generation.prefix` ends at the probe's user message; `expected` validates the probe answer.
- `sizing` is `{"unit": "turn", "keep_head": n}` for every family. Turns have a fixed
  construction (whole BFCL decisions, fixed word windows, fixed record blocks, whole files),
  so no family needs construction-time token counting and no family is `fixed`.
- The fixture holds as much history as its source allows; consumers size it down.

**Sizing** (the contract rule, implemented by `Fixture.sized`; the Rust harness implements the
same rule): keep the head and the probe, then add history units newest first, keeping the
most history whose rendered length is at or under the target (an engine's context capacity
is a hard limit), in message order. `generation.prefix` moves down by
the number of dropped messages. Offered tools are the top-level `tools` followed by the
`tools` of kept history questions in conversation order, deduplicated by function name.
`sized` returns messages without the fixture-only `scored` and `tools` fields, the offered
tools, the shifted prefix and the scored message indices. It finds the largest fitting
history by bracketing and bisection, returns the whole fixture when it all fits, and fails
only when the head and probe alone exceed the target.

## Families

| family | source | head | turn | probe answer and `expected` | full length |
|---|---|---|---|---|---|
| `tools` | BFCL v4 at a pinned gorilla commit, hash-locked (`data/bfcl-v4.*`) | system | user question → assistant tool calls → tool acknowledgements | ground-truth calls; `{"calls": [{"name", "arguments": {key: [allowed values]}}]}`; `max_tokens` 32768 | 797 turns, ≈1.35 MB |
| `prose-continue` | Moby Dick (Gutenberg), vendored and hash-locked (`data/moby-dick.*`) | system | user: 256-word passage → assistant: next 128 words | `{}` (output varies by engine; prose must be non-empty); `max_tokens` 256 | ≈535 turns, ≈1.25 MB |
| `prose-repeat` | same | system | user: 512-word passage plus a repeat instruction → assistant: latest paragraph-aligned suffix of ≥256 words | `{"text": suffix}`; `max_tokens` 256 | ≈400 turns, ≈2 MB |
| `prose-shift` | same | system | as `prose-repeat` | the probe asks for the passage's last 128 words verbatim, about 150 new words, then the same 128 words again; `{"text": the 128 words}`; `max_tokens` 768 | ≈400 turns, ≈2 MB |
| `retrieval` | RULER NIAH recipe (pinned commit, no upstream runtime) | system, first record block with the needles, "Noted." | user: 64 distractor records → "Noted." | JSON object of the needle values; `{"values": {key: value}}`; `max_tokens` 1024 | 192 turns, ≈0.47 MB |
| `code` | 16 files of this repository at a pinned commit, read with `git show` and hash-locked (`data/magnitude-code.lock.json`) | system | user asks for a file → assistant returns it in a fenced block | no `generation` | 16 turns, ≈90 KB |

Sizes are per fixture.

- **tools:** lane *k* probes the *k*-th decision of the interleaved BFCL selection; its history is
  every other decision, rotated to start after the probe so lanes end differently. The
  top-level `tools` are the probe's functions; each history question carries the functions
  its calls use, so a sized session offers exactly the tools of the turns it keeps.
- **prose:** the body from "Call me Ishmael." is unwrapped into paragraphs with straight
  quotes. Lanes probe consecutive windows near the end of the book, and each system message
  names its lane so sessions share no prefix. A repeat suffix of ≥256 words covers the 256-token
  output budget for current tokenizers (≈1.2–1.4 tokens per English word).
- **retrieval:** the needles are in the kept head and the question is in the probe, so the
  facts are always the farthest from the question. Lane *k* uses seed `42 + k`. Records are
  independent of position and sizing.
- **code:** building needs a git checkout that contains the pinned commit.

Answer validation (`validation.py`) keeps session-bench's semantics: tool calls must match
distinct expected calls under BFCL possible-answer rules (an argument may be omitted only when
`""` or `null` is allowed, and booleans never equal numbers). Retrieval answers must be strict
JSON string objects with no duplicate keys. Prose answers need only be non-empty, and the bench
judges finish reasons against its output budget.

## Sources and determinism

Every source is immutable and verified against its lock before use:
- **Moby Dick** is vendored as `data/moby-dick.txt.gz`: the public-domain Gutenberg file with
  its license, gzip-compressed. The upstream URL is mutable (its current bytes no longer match),
  so it is recorded only as `origin`. The decompressed bytes must match the locked sha256.
- **BFCL** is fetched from `raw.githubusercontent.com` at a pinned commit, which is
  content-addressed, then cached under `$XDG_CACHE_HOME/magnitude/evaluation/sources` (default
  `~/.cache`) and checked per file.
- **retrieval** has no upstream input; records are derived from seeds.
- **code** is read with `git show` at a pinned commit, so it needs a checkout that contains it
  (not a shallow clone that omits it).

The same builder and locks produce
byte-identical sets: fields keep a fixed order, tool-call ids and lanes are positional, and
nothing depends on hash order or time.
