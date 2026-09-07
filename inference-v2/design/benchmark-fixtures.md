# Benchmark context fixtures

`benchmark_fixtures` owns source acquisition, deterministic content construction and
prepared inputs. Model benchmarks and session-bench consume it; execution, scheduling
and result storage stay with their respective runners.

## Content

| Fixture | Construction | What it represents |
|---|---|---|
| `prose.moby-dick` | A contiguous token window from the pinned Project Gutenberg text; subsequent book tokens form the replay continuation | Sustained prose context and continuation |
| `tools.bfcl` | Pinned BFCL questions, tool schemas and declared calls assembled into complete interactions | Structured tool history and a pending tool decision |

Moby Dick is downloaded from the URL in its checked-in source lock and verified by
SHA-256. Normalize line endings and remove the Gutenberg wrapper; retain the book's
front matter. No book text or prepared token arrays belong in the repository.
Sources, tokenizations and prepared fixtures live under
`~/.cache/magnitude/benchmarks/` (or the corresponding `XDG_CACHE_HOME` directory).
A changed download fails verification until an intentional lock revision.

Prose offsets are token positions in the normalized book. Windows have the exact
requested token length, without repetition, padding or chat wrapping. Generation is
raw autoregressive book continuation. The tokenizer determines the book's token
count; a window extending past its end fails.

Tool offsets select the starting interaction in the pinned corpus. Histories retain
complete question/call/result rounds and compatible tool definitions. Tool results
are synthetic argument acknowledgements, not captured executions or agent traces.
Preparation uses the consumer's actual tokenizer and renderer to find the first
complete round reaching the target; record the resulting length, including schemas
and template overhead. No characters-per-token estimate determines the input.

Session-bench sizes with the first selected target, gives every target those same
messages and tools, then records each target's native rendered count. Target order
therefore participates in fixture preparation. Model comparisons use the same
prepared token sequence when their tokenizer/rendering contract matches. Canonical
JSON key order is shared between sizing, saved requests and execution.

## Execution modes

| Mode | Measured work |
|---|---|
| `prefill` | Consume a declared continuation chunk after a prepared prefix; no vocabulary projection requested |
| `replay` | One-token forwards with logits, supplying fixed continuation tokens instead of feeding predictions back |
| `generate` | One-token forwards with greedy predictions fed back autoregressively; stop at EOS or the declared token limit |

Content and execution mode are independent. Prose replay uses actual subsequent
book tokens. Tool replay uses a rendered declared completion of the pending decision,
requiring that rendering preserve the prompt prefix. It never pads a short completion;
a replay requiring more tokens fails. Record actual generated lengths, including EOS.

The model benchmark times forward execution, state transactions and completion;
generation also includes synchronous greedy selection. Prefix preparation, restoration,
downloads and tokenization are outside timing. Generation prepares all but the last
prompt token so its first measured forward produces the first output token. This
boundary is distinct from pipelined engine service and HTTP serving measurements.
Session-bench continues to own those serving measurements and session schedules.

### Engine workload validation

Engine/generation hierarchy cases use a declared fixed output count and record
actual generated tokens, completed state/delivery, prefix reuse and repeated-run
determinism. These characterize service under the specified workload. Numerical
model equivalence is qualified separately with matched inputs and geometry; fixed
replay controls hold future inputs constant when free generation diverges.

Exact greedy equality across different legal chunk/batch geometries is not a
universal workload oracle: stock upstream Qwen also changes logits and continuations
across those geometries. Existing independent-output checks remain available for
experiments that explicitly require that equivalence. A corrected workload contract
gets a distinct benchmark identity; failed earlier checks remain rejected evidence.
No numerical tolerance changes follow from this distinction.

### Session-bench prose

`--prose` uses the same pinned book with a chat continuation recipe,
`prose-chat-history-v1`. It wraps each passage in a request to return only its prose
continuation. This differs from the model benchmark's raw, unwrapped token window;
the recipe identity keeps those observations distinct.

An independent reading session begins at the book's start. Preparation extends a
contiguous passage at word boundaries until the first target's rendered prompt reaches
the context target, including chat overhead. A passage has at least 256 words; `single`
uses that minimum. The actual token count is recorded, without padding or repetition.

After a planned turn, append the next 128 book words as the canonical assistant
continuation. A subsequent request starts with the following passage and retains all
earlier messages. Model output never enters future prepared inputs. Fork children
share the parent's canonical completion; independent sessions have distinct labels.
Running out of book text fails explicitly. Provenance records the source identity,
passage word boundaries, canonical continuation length, rendered length and content digest.

The serving runner generates up to 256 tokens per prose request, with EOS or the
budget as valid endpoints. It records actual lengths and requires a complete text
response and timing/usage evidence. It does not score the continuation against the book.

Use both prose and tools when qualifying whole-model behavior, including 4K, 16K and
at least 64K context. Start with short controlled runs; longer sampling belongs at
acceptance boundaries. These fixtures characterize performance, not answer quality.
Existing session-bench validation remains a separate runner policy.

Synthetic component inputs are appropriate when the relevant conditions are controlled:
shapes, precision, layout, residency, routes, sharing or acceptance. Declare those
conditions. Repeated token IDs do not establish representative whole-model behavior,
particularly where content changes expert routing or generation length.

## Evidence

Existing benchmark results carry fixture ID, source/recipe identity, tokenizer and
renderer identity, offset, requested/actual lengths, and content/token digests alongside
the implementation fingerprint and measurement mode. Session-bench stores fixture
provenance in `requests.jsonl` and per-target counts in its existing prompt-count files.
These describe the actual inputs without adding another results database or evidence
table. Compare only matched content, execution boundaries and operating points.
