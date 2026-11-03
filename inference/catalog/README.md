# Release model catalog

`models.json` owns the catalog. A model lists its `sources`: each is one repository's
publication of it, with the variants it hosts, the projector they share and the license it grants,
so a model's variants can come from different publishers. A separately packaged speculative draft
names its own repository. `models.lock.json` maps every repository the catalog resolves from to its
immutable Hugging Face commit.

Input modality is inspected from the exact target package. Image-capable targets receive a
projector during generation: a sole repository `mmproj` is selected automatically, while
repositories with multiple candidates require an exact `projector.path` declaration on the source.
The projector is locked as a related component of the target package.

Each variant's primary GGUF is the one file (or first shard) whose path contains its `format`.
When other repository files also contain that format, such as MTP heads, the variant declares its
exact `path` instead.

A catalog entry declares no context length. Its serving profile is the target's supported maximum
context from its GGUF metadata, which is the context the engine resolves, serves and assesses.

Nor does an entry declare what its model serves. A target whose GGUF header carries a decision
description (its decision template `tokenizer.chat_template.decision`) serves decisions only and
every other target serves chat, read
from the locked target header when the bundle is generated and carried in its planner input, so the
catalog knows it before download.

A chat model's entry declares its `artificialAnalysis` score; a decision model's declares none, since
the Intelligence Index measures generated text. Generation joins the score to what the header serves
(`serves: chat` with its intelligence, or `decisions`) and refuses either declared the other way.

```sh
cargo run --release -p magnitude-service-catalog-tool -- update-lock ...   # advance the commit map
cargo run --release -p magnitude-service-catalog-tool -- build-bundle ...  # build the header bundle
cargo run --release -p magnitude-service-catalog-tool -- headers --output DIR --model-store DIR  # headers for tests
cargo run --release -p magnitude-service-catalog-tool -- acquire --model qwen3.5-4b:gguf:q4 --model-store DIR
cargo run --release -p magnitude-service-catalog-tool -- audit --headers DIR --output FILE --model-store DIR  # tensor-type audit
cargo run --release -p magnitude-service-catalog-tool -- header local PATH  # metadata and tensors
cargo run --release -p magnitude-service-catalog-tool -- slice --output FILE hub --repository REPO --revision REV --path PATH --model-store DIR
```

`fixture-header` and `weight-sample` write the assessment fixtures (see
`engine/fixtures/assessment/README.md`); `slice` writes a truncated real-weight GGUF of a model too
large to run whole, with a provenance record beside it. `header`, `weight-sample` and `slice` read
a `local PATH` or a `hub --repository REPO --revision REV --path PATH --model-store DIR` file; a
Hub file is read by range requests only, through the model store's Hub client.

`conversions/decision.py` writes a `magnitudedev` decision GGUF from a decision model's
safetensors checkpoint: the base architecture through llama.cpp's converter at its pinned revision,
plus the decision description it writes for the checkpoint it recognizes (the decision template
from `conversions/decision-templates/`, the `decision.*` keys of
`engine/decision/KEYS.md`, `<arch>.attention.causal = false`) and, for a description read through
a readout head, the head as `decision.readout.weight` in its reference dtype. It takes a llama.cpp
checkout at the pinned revision and a local snapshot of the checkpoint's repository (weights,
configuration, tokenizer), from `inference/`:

```sh
uv run --script catalog/conversions/decision.py convert --llama-cpp LLAMA_CPP --checkpoint CHECKPOINT \
    --output OUT --name NAME
python3 catalog/conversions/decision.py describe --checkpoint CHECKPOINT --output FILE  # the description alone
```

| Checkpoint | `--name` | Template | Writes (`OUT/`) | Readout |
|---|---|---|---|---|
| `vllm-sr/d3-ultra` | `d3-ultra` | `d3-header-v1` | `d3-ultra-BF16.gguf`, `mmproj-d3-ultra-BF16.gguf` | F32 readout head |
| `baselabs/mammoth-1` | `mammoth-1` | `qev-labels-v2` | `mammoth-1-BF16.gguf` | slot labels from the LM head |
| `baselabs/elephant-1` | `elephant-1` | `qev-labels-v2` | `elephant-1-BF16.gguf` | slot labels from the LM head |
| `perplexity-ai/pplx-decider-flash-v1.2` | `pplx-decider-flash-v1.2` | `joint-question-inline-readout-v1` | `pplx-decider-flash-v1.2-BF16.gguf`, `mmproj-pplx-decider-flash-v1.2-BF16.gguf` | BF16 readout head |
| `LiquidAI/d1-2B` | `d1-2B` | `d1-prompt-v1` | `d1-2B-BF16.gguf`, `mmproj-d1-2B-BF16.gguf` | option codes' spellings from the LM head |

Base Labs' reference applies its block layout to text only, so their conversions serve no images
and write no projector; d3-ultra's and pplx-decider's projectors are llama.cpp's own conversion of
their vision towers, and d1-2B's is llama.cpp's LFM3-VL projector, its text model llama.cpp's LFM3
conversion of the same weights (without llama.cpp's own d1 type and template, which the description
replaces).

Each model's variants are the catalog's `gguf:q4` (`Q4_K_M`), `gguf:q5` (`Q5_K_M`), `gguf:q6`
(`Q6_K`) and `gguf:q8` (`Q8_0`), published as `magnitudedev/<NAME>-GGUF` (`NAME` the conversion's
`--name`) and quantized from the BF16 conversion with llama.cpp's `llama-quantize` at the same
revision. A description read through a readout head keeps the head in its reference dtype, which
the engine requires (F32 or BF16), so its quantization names that dtype:

```sh
llama-quantize --tensor-type decision.readout=f32  d3-ultra-BF16.gguf d3-ultra-Q4_K_M.gguf Q4_K_M
llama-quantize --tensor-type decision.readout=bf16 pplx-decider-flash-v1.2-BF16.gguf pplx-decider-flash-v1.2-Q8_0.gguf Q8_0
llama-quantize mammoth-1-BF16.gguf mammoth-1-Q4_K_M.gguf Q4_K_M   # no readout head: no override
```

A GGUF converted while the description's keys were under `<arch>.decision.` is rewritten with each
under `decision.` in its place, every other key and every tensor byte unchanged, so a published
quantization moves without being quantized again:

```sh
uv run --script catalog/conversions/decision.py rename-namespace --llama-cpp LLAMA_CPP \
    --input d3-ultra-Q4_K_M.gguf --output OUT/d3-ultra-Q4_K_M.gguf
```

`acquire` resolves one catalog model at its locked revisions, downloads into a managed model store
only the target, drafter and projector files that are missing, verifies them through the service's
own download path, and prints their absolute paths and SHA-256 digests as JSON (the evaluation
model-reference contract). Evaluation tools obtain models only through it.

Generation resolves the pinned repositories and writes `model-planner-inputs.bundle`: the exact GGUF
header (every byte through the aligned tensor-data offset) of every package component, which is
what the engine's header-only planning and chat-template inspection read before a model is
downloaded. Repeated references to the same immutable package share one package identity and one
set of bundled headers. Each target also carries its weight sample (a few hundred bytes of its early
normalization weights, read by range requests), stored as an input of its own that the variants
of one model share. The bundle is derived release output and is not committed. It is the only
catalog-related file shipped alongside the service; catalog definitions and pins are compiled into
the executable.

`headers` builds that bundle under `--output` (only when it is not current for this lock) and
writes each component's exact header, extended sparsely to its file's size, to `--output/headers/`
at `<repository>/<revision>/<path>`, with an index naming the lock. Family and assessment tests read
that export through `MAGNITUDE_CATALOG_HEADERS` and refuse an export of another lock; the
evaluation flow's `--catalog-headers` runs the export on the host and sets the variable. A header
a test needs outside the lock is a pinned artifact (`magnitude_artifacts::catalog_headers::artifacts`:
its repository, revision and path), exported beside the catalog's the same way.
