# Custom models

Magnitude runs any GGUF whose model family the engine has kernels for, not only the models it recommends:
other quantizations, finetunes, merges, and abliterated or uncensored versions. Models come from
two places:

- **Hugging Face, through Magnitude.** `models pull owner/repo[:QUANT]` downloads into Magnitude's
  own store. `models rm` deletes that copy.
- **Found on this computer.** The Hugging Face cache, LM Studio, the llama.cpp cache, Ollama, and
  folders you add are scanned for GGUFs. These files belong to their apps; `models hide` hides a
  found model without touching its files, and so does `models rm` for a found model. Models added
  from Hugging Face can't be hidden; `models rm` removes them.

Model IDs are `hf:<owner>/<repo>/<file>` for a model added from Hugging Face or found in the Hugging
Face cache, and `local:<name>-<quant>:<hash>` otherwise, such as
`local:gemma-4-12b-instruct-q8_0:41ad09c2e7f3`. Only the hash identifies a `local:` model; the name
part is for reading and may change. Each copy's ID comes from that copy alone.

## Find and add a model

```text
magnitude models search "Qwen3.6 35B-A3B"
magnitude models show HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
magnitude models pull HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q4_K_P
magnitude models show <model-id>
magnitude models load <model-id>
```

`models search` lists repositories with the base model each derives from (fine-tune, merge, adapter
or quant of it), the recommended quant with its size, estimated speed and placement on this
computer, and downloads. Results whose architecture Magnitude doesn't run follow in their own table
with the architecture and the reason. `models show <model-id>` of a recommended model lists its other
quants and prints the search that finds other versions.

`models show` with a huggingface.co link, `owner/repo`, or `owner/repo:QUANT` changes nothing. It
lists every version with its bit width, size, estimated speed, served context, placement, and the
engine's verdict for this computer (`Fits`, `Too large`, or `Unsupported:` with the reason), marks
the recommended version and any version already on this computer, then shows where the chosen
version's bytes go (device memory, host RAM, read from disk) and its speed at each assessed context
depth, the drafter choices, and the exact command to download.

`models pull` with no quant picks the recommended version and says so. It pins the repository's
current commit and prints the model ID. If that version is already on this computer, only the
drafter downloads. A version Magnitude can't run is refused with the reason; one that needs more
memory than this computer can provide is refused unless you pass `--force`. A gated repository
says what it needs: run `magnitude hf login` to sign in, or accept the model's terms on
its Hugging Face page (some models also need the author's approval), then try again. A recommended model's ID downloads or updates that model. A custom model ID updates
that model: an added model to its repository's newer commit, and any model to a newer revision of
its recommended model's drafter.

## Legacy DeepSeek V4.1 GSQ-RCO GGUFs

The original four-shard GSQ-RCO layout with a separate Engram file cannot be loaded directly.
If you don't have a Magnitude source checkout, use the
[upstream-format GSQ-RCO release](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/tree/65d53bc8be65123e255ff02f3ded958b6dbf54fd)
instead. Download shards `00001-of-00006` through `00005-of-00006` for one quantization, plus
`DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` **from the same converted revision**. In the download
directory, name that new Engram copy `<model-prefix>-00006-of-00006.gguf`, using the prefix from the
other five shards. Add the directory with
`magnitude models folders add /path/to/model-directory`.

To reuse an original download, run Magnitude's catalog tool from an up-to-date source checkout
with its pinned Rust toolchain and a native build toolchain. Keep the four original backbone
shards in one folder and supply the original `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`. Download
[`chat_template.jinja`](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/resolve/65d53bc8be65123e255ff02f3ded958b6dbf54fd/chat_template.jinja)
and [`exp_probs_b_vl.f32`](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/resolve/65d53bc8be65123e255ff02f3ded958b6dbf54fd/exp_probs_b_vl.f32)
from that pinned release. The importer checks both hashes and does not fetch anything.

Prepare `SHA256SUMS` with five `sha256sum`-format lines: each 64-digit SHA-256, two spaces, then the
local basename of one original backbone shard or the original Engram file. No paths or duplicate
names. Take the hashes from the publisher's trusted legacy revision, not from the already-downloaded
files. On Hugging Face, open each file's details at its exact commit and use the LFS SHA-256, or run
`curl -fsSI 'https://huggingface.co/OWNER/REPO/resolve/COMMIT/FILE'` and use the 64-digit
`x-linked-etag` (without quotes) on the Hub response. Check `x-repo-commit` against the intended commit.
Do not use a branch name, the Xet hash, or the ETag from the redirected CDN response. For an edited
variant, use the hashes published for that exact variant, not hashes for the stock ISTA weights.
If a trusted source cannot supply them, use the already-converted release instead.

From the checkout's `inference` directory, using absolute paths:

```sh
cargo run --release --locked -p magnitude-service-catalog-tool -- import-deepseek41-legacy \
  /path/to/legacy/MODEL-00001-of-00004.gguf \
  --engram /path/to/legacy/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf \
  --chat-template /path/to/chat_template.jinja \
  --vision-bias /path/to/exp_probs_b_vl.f32 \
  --source-sha256 /path/to/SHA256SUMS \
  --output /path/to/new-model-directory --dry-run
```

The dry run checks the manifest, headers, and small conversion inputs and reports the exact output
size. Allow hundreds of GB for a full bundle. A dry run does not authenticate the full source files
or write any output.
Use a filesystem that supports large files and hard links, and choose a new output directory. Rerun
without `--dry-run` to authenticate all five source files, write and reread the six output shards,
and publish the first shard last. Keep the originals unchanged throughout the run; the importer
refuses unfinished chat-template updates and never edits inputs. It imports only the text model,
not an optional projector or drafter.

Add the converted folder after the importer finishes successfully. Failed runs roll back their
output when possible. Abrupt termination or power loss can leave staging or partially published
files; publication is not crash-atomic. Use a new output directory for a retry.

## Drafters

Every model that runs gets the best drafter known to work with it. Only drafters Magnitude can
stand behind are chosen automatically, and the first one the engine binds, in this order, is
preselected:

1. a separate drafter in the model's own repository, or its folder for a local file;
2. the drafter trained for the matching recommended model;
3. MTP heads built into the model;
4. no drafter.

Within one source, methods go LiLiCorr, DFlash2, DSpark, DFlash, a separate MTP file, then built-in MTP heads,
and ties go to the smaller download. Magnitude never searches Hugging Face for drafters. A drafter
you name is checked like the others but is never chosen automatically.

`--drafter` on `models pull`, and `models drafter <model-id> <drafter>`, take:

- `auto`: the preselected drafter (the default);
- `none`: no drafter;
- `built-in`: MTP heads inside the model;
- a recommended model's ID, which uses that model's drafter;
- a Hugging Face file (a huggingface.co link, `owner/repo/path.gguf`, `hf:owner/repo/path.gguf`, or
  `owner/repo:path.gguf`) or a local GGUF path.

A drafter the engine rejects for the model is refused with the reason. One whose lineage doesn't
match is allowed and marked untested. Recommended models keep the drafter Magnitude ships with them, so
`models drafter` refuses them. `models show <model-id>` shows the drafter in use and every choice
with its label, its verdict, and the argument that selects it. A found or added model whose
recommended drafter isn't on this computer shows "Faster with a drafter · Add"; `models drafter
<model-id> auto` downloads only the drafter. A drafter whose bytes are already in any location, another
app's included, is used where it is and never downloaded. When a recommended model's drafter gets a new revision,
`models list` shows Update for every model using it; `models pull <model-id>` downloads it.

## What's on this computer

`models list` lists recommended models on this computer and every custom model with its quant, size,
estimated speed, served context (marked when below the model's own), source (Magnitude, HF cache,
LM Studio, llama.cpp, Ollama, or folder) and status, then counts the models that can't run in
Magnitude and hidden models, and ends with the assessment progress. `--all` lists both: models that
can't run with a plain reason, such as "No kernels for the deepseek4 architecture" or "Uses IQ1_M
weights, which Magnitude doesn't support yet", and hidden models marked `Hidden`. `models show <model-id>`
of a model that can't run shows where it is and why. `models unhide` shows a hidden model again.

`models folders list` shows the automatic locations and your own folders, each directory with its
status: Found, Off, Not found, or Can't read. A directory Magnitude can't read is followed by what
to do, such as giving your account read access to a system Ollama service's folder or setting
`OLLAMA_MODELS` to a folder you can read. `models folders add <path>` watches a folder: GGUFs added
later appear and deleted ones disappear. `models folders remove <path>` stops watching it.
`models folders disable <location>` stops scanning an automatic location and `models folders enable
<location>` scans it again, where the location is `huggingface-cache`, `lm-studio`, `llama-cpp`, or
`ollama`; this applies to every directory of that location and never touches its files.
