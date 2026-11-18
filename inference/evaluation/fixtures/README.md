# magnitude-fixtures

The only model inputs of evaluation. A **template** is one benchmark request named
`<key>:<context>` (e.g. `repeat:8k`, `code:8k`):

| key | source | what it asks |
|---|---|---|
| `repeat` | Moby Dick (Gutenberg), vendored and hash-locked (`data/moby-dick.*`) | repeat a supplied passage verbatim |
| `twist` | Moby Dick (Gutenberg), vendored and hash-locked (`data/moby-dick.*`) | repeat passage end, write ~150 new words, repeat again |
| `retrieve` | RULER-derived key/value records (pinned recipe, no upstream runtime) | return JSON object of needle key/value records |
| `code` | BigCodeBench v0.1.4 (Apache-2.0) at a pinned HF commit, hash-locked (`data/bigcodebench.lock.json`) | implement a self-contained Python function in a fenced block |
| `tools` | BFCL v4 decisions at a pinned gorilla commit, hash-locked (`data/bfcl-v4.*`) | emit ground-truth tool call(s) |

The context is the prompt's length in tokens, `<n>k` with `1k` = 1024: `repeat:8k`, `twist:4k`,
`retrieve:8k`, `code:8k`, `tools:4k`. `Template.parse` is the only parser of a name, and fails
naming the valid keys.

Each key yields **sessions** in a fixed order, lane 0, 1, 2, …; each lane's probe is a
different item of the fixture. **A template is lane 0's session, sized to its context**; a run
of C sequences uses lanes 0 … C − 1, each sized to the same context. A lane's session does not
depend on how many lanes are built.

```bash
cd inference/evaluation
uv run pytest fixtures
```

```python
from magnitude_fixtures import Template, TemplateFile, validation

template = Template.parse("repeat:8k")        # key, context=8192
file = template.build(lanes=2)                # TemplateFile: template, context, source, sessions
path.write_bytes(file.encode()); file.digest  # what the harness reads (`--template FILE`)
session = TemplateFile.read(path).sessions[0]
sized = session.sized(8192, rendered_length)  # rendered_length(messages, tools) -> int
sized.messages, sized.tools, sized.prefix, sized.scored  # serve messages[:prefix] with tools
session.conversation_with(history=4)  # head + the 4 newest history units + probe
session.units()  # message ranges of the sizing turns; the last is the probe
validation.tool_calls(session.generation.expected, calls)
validation.retrieve(session.generation.expected, text)
```

`mrig` builds the template file an instrument's `--template` names, holding the lanes the run
needs, and stores it by its digest (SHA-256 of the file bytes).

## The session

Each session is a kept head, history turns, then one **probe** turn last.
- Only the probe's assistant message is `scored`, so quality positions sit at the deepest
  point of the context and their count does not grow with context length.
- `generation.prefix` ends at the probe's user message; `expected` validates the probe answer.
- `keep_head` is the head's length. Turns have a fixed construction (whole BFCL decisions, whole
  BigCodeBench tasks, fixed word windows, fixed record blocks), so no fixture needs
  construction-time token counting.
- The session holds as much history as its source allows; consumers size it down.

**Sizing** (the contract rule, implemented by `Session.sized`; the Rust harness implements the
same rule): keep the head and the probe, then add history units newest first, keeping the
most history whose rendered length is at or under the target (an engine's context capacity
is a hard limit), in message order. `generation.prefix` moves down by the number of dropped
messages. Offered tools are the top-level `tools` followed by the `tools` of kept history
questions in conversation order, deduplicated by function name. `sized` returns messages
without the session-only `scored` and `tools` fields, the offered tools, the shifted prefix
and the scored message indices. It finds the largest fitting history by bracketing and
bisection, returns the whole session when it all fits, and fails only when the head and probe
alone exceed the target.

## Keys

| key | head | turn | probe answer and `expected` | full length |
|---|---|---|---|---|
| `repeat` | system | user: 512-word passage plus a repeat instruction → assistant: latest paragraph-aligned suffix of ≥256 words | `{"text": suffix}`; `max_tokens` 256 | ≈400 turns, ≈2 MB |
| `twist` | system | as `repeat` | the probe asks for the passage's last 128 words verbatim, about 150 new words, then the same 128 words again; `{"text": the 128 words}`; `max_tokens` 768 | ≈400 turns, ≈2 MB |
| `retrieve` | system, first record block with the needles, "Noted." | user: 64 distractor records → "Noted." | JSON object of the 4 needle values; `{"values": {key: value}}`; `max_tokens` 1024 | 192 turns, ≈0.47 MB |
| `code` | system | user: BigCodeBench `instruct_prompt` → assistant: canonical Python solution in a fenced block | self-contained Python implementation; `{}` (output varies by engine; text must be non-empty); `max_tokens` 1024 | 1140 tasks, ≈1.5 MB |
| `tools` | system | user question → assistant tool calls → tool acknowledgements | ground-truth calls; `{"calls": [{"name", "arguments": {key: [allowed values]}}]}`; `max_tokens` 32768 | 797 turns, ≈1.35 MB |

Sizes are per session.

- **`repeat` / `twist` (Moby Dick):** the body from "Call me Ishmael." is unwrapped into
  paragraphs with straight quotes. Lane *k* probes the *k*-th window from the end of the book,
  and each system message names its lane so sessions share no prefix. A repeat suffix of ≥256
  words covers the 256-token output budget for current tokenizers (≈1.2–1.4 tokens per English
  word).
- **`retrieve` (RULER):** the needles are in the kept head and the question is in the probe, so
  the facts are always the farthest from the question. Lane *k* uses seed `42 + k`. Records are
  independent of position and sizing.
- **`code` (BigCodeBench):** lane *k* probes the *k*-th task from the end of BigCodeBench
  v0.1.4 after every preceding task, and each system message names its lane so sessions share no
  prefix.
- **`tools` (BFCL):** lane *k* probes the *k*-th decision of the interleaved BFCL selection; its
  history is every other decision, rotated to start after the probe so lanes end differently. The
  session's top-level `tools` are the probe's functions; each history question carries the
  functions its calls use, so a sized session offers exactly the tools of the turns it keeps.

Answer validation (`validation.py`) is keyed by template key and keeps session-bench's
semantics: tool calls must match distinct expected calls under BFCL possible-answer rules (an
argument may be omitted only when `""` or `null` is allowed, and booleans never equal numbers).
`retrieve` answers must be strict JSON string objects with no duplicate keys. Text answers
(`repeat`, `twist`, `code`) need only be non-empty, and the bench judges finish reasons against
its output budget.

## Sources and determinism

Every source is immutable and verified against its lock before use:
- **Moby Dick** is vendored as `data/moby-dick.txt.gz`: the public-domain Gutenberg file with
  its license, gzip-compressed. The decompressed bytes must match `data/moby-dick.lock.json`.
- **BigCodeBench** is fetched on demand from Hugging Face (`bigcode/bigcodebench`,
  `data/v0.1.4-00000-of-00001.parquet`) at a pinned commit, cached under
  `$XDG_CACHE_HOME/magnitude/evaluation/sources` (default `~/.cache`), and verified against
  `data/bigcodebench.lock.json`.
- **BFCL** is fetched from `raw.githubusercontent.com` at a pinned commit, which is
  content-addressed, then cached under `$XDG_CACHE_HOME/magnitude/evaluation/sources` (default
  `~/.cache`) and checked per file.
- **RULER** has no upstream input; records are derived from seeds.

The same builder and locks produce byte-identical template files: fields keep a fixed order,
tool-call ids and lanes are positional, and nothing depends on hash order or time.
