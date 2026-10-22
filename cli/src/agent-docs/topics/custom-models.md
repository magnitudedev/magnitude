# Custom models

Magnitude runs any GGUF whose model family the engine has kernels for, not only catalog models:
other quantizations, finetunes, merges, and abliterated or uncensored versions. Models come from
two places:

- **Hugging Face, through Magnitude.** `catalog pull owner/repo[:QUANT]` downloads into Magnitude's
  own store. `catalog remove` deletes that copy.
- **Found on this computer.** The Hugging Face cache, LM Studio, the llama.cpp cache, Ollama, and
  folders you add are scanned for GGUFs. These files belong to their apps; `models hide` hides a
  found model without touching its files. Models added from Hugging Face can't be hidden;
  `catalog remove` removes them.

Model IDs are `hf:<owner>/<repo>/<file>` for a model added from Hugging Face or found in the Hugging
Face cache, and `local:<name>-<quant>:<hash>` otherwise, such as
`local:gemma-4-12b-instruct-q8_0:41ad09c2e7f3`. Only the hash identifies a `local:` model; the name
part is for reading and may change. Each copy's ID comes from that copy alone.

## Find and add a model

```text
magnitude catalog search "Qwen3.6 35B-A3B"
magnitude catalog inspect HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
magnitude catalog pull HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q4_K_P
magnitude models status <model-id>
magnitude models load <model-id>
```

`catalog search` lists repositories with the uploader, the catalog model each is a variant of, and
downloads. Results whose architecture Magnitude doesn't run follow in their own table with that
reason. `catalog show <model-id>` prints the search that finds other versions
of a catalog model.

`catalog inspect` accepts a huggingface.co link, `owner/repo`, `owner/repo:QUANT`, or a local GGUF
path, and changes nothing. It lists every version with its size and the engine's verdict for this
computer (`Fits`, `Too large`, or `Unsupported:` with the reason), marks the recommended version and
any version already on this computer, then lists drafter choices and the exact command to download.

`catalog pull` with no quant picks the recommended version and says so. It pins the repository's
current commit and prints the model ID. If that version is already on this computer, only the
drafter downloads. A version Magnitude can't run is refused with the reason; one that needs more
memory than this computer can provide is refused unless you pass `--force`. A gated repository
says what it needs: run `magnitude hf login` to sign in, or accept the model's terms on
its Hugging Face page (some models also need the author's approval), then try again. A catalog model ID keeps working exactly as before. A custom model ID updates
that model: an added model to its repository's newer commit, and any model to a newer revision of
its catalog drafter.

## Drafters

Every model that runs gets the best drafter known to work with it. Only drafters Magnitude can
stand behind are chosen automatically, and the first one the engine binds, in this order, is
preselected:

1. a separate drafter in the model's own repository, or its folder for a local file;
2. the catalog drafter trained for the matching catalog model;
3. MTP heads built into the model;
4. no drafter.

Within one source, methods go DFlash2, DSpark, DFlash, a separate MTP file, then built-in MTP heads,
and ties go to the smaller download. Magnitude never searches Hugging Face for drafters. A drafter
you name is checked like the others but is never chosen automatically.

`--drafter` on `catalog pull`, and `models drafter <model-id> <drafter>`, take:

- `auto`: the preselected drafter (the default);
- `none`: no drafter;
- `built-in`: MTP heads inside the model;
- a catalog model ID, which uses that model's drafter;
- a Hugging Face file (a huggingface.co link, `owner/repo/path.gguf`, `hf:owner/repo/path.gguf`, or
  `owner/repo:path.gguf`) or a local GGUF path.

A drafter the engine rejects for the model is refused with the reason. One whose lineage doesn't
match is allowed and marked untested. Catalog models keep the drafter Magnitude ships with them, so
`models drafter` refuses them. `models status <model-id>` shows the drafter in use and every choice
with its label, its verdict, and the argument that selects it. A found or added model whose
recommended drafter isn't on this computer shows "Faster with a drafter · Add"; `models drafter
<model-id> auto` downloads only the drafter. A drafter whose bytes are already in any location, another
app's included, is used where it is and never downloaded. When a catalog drafter gets a new revision,
`models status` shows Update for every model using it; `catalog pull <model-id>` downloads it.

## What's on this computer

`models status` lists catalog models on this computer and every custom model with its source, then
counts the models that can't run in Magnitude and hidden models. `--all` lists both: models that
can't run with a plain reason, such as "No kernels for the deepseek4 architecture" or "Uses IQ1_M
weights, which Magnitude doesn't support yet", and hidden models marked `Hidden`. `models status <model-id>`
of a model that can't run shows where it is and why. `models unhide` shows a hidden model again.

`models folders list` shows the automatic locations and your own folders, each directory with its
status: Found, Off, Not found, or Can't read. A directory Magnitude can't read is followed by what
to do, such as giving your account read access to a system Ollama service's folder or setting
`OLLAMA_MODELS` to a folder you can read. `models folders add <path>` watches a folder: GGUFs added
later appear and deleted ones disappear. `models folders remove <path>` stops watching it.
`models folders disable <location>` stops scanning an automatic location and `models folders enable
<location>` scans it again, where the location is `huggingface-cache`, `lm-studio`, `llama-cpp`, or
`ollama`; this applies to every directory of that location and never touches its files.
