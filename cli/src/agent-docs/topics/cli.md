# Magnitude CLI

Running `magnitude` without a subcommand prints help and exits.
Use Desktop or `magnitude serve` to run inference. The following commands manage models and agent connections:

```text
magnitude update [check|status|download|install|discard]
magnitude app open
magnitude serve
magnitude server setup|remove
magnitude status
magnitude hardware
magnitude catalog status
magnitude catalog list
magnitude catalog show|cancel|remove <model-id>
magnitude catalog recommendations [--preference <value>] [--limit <count>]
magnitude catalog search <query> [--all] [--limit <count>]
magnitude catalog inspect <huggingface-url | owner/repo | owner/repo:QUANT | path.gguf>
magnitude catalog pull <model-id | owner/repo[:QUANT] | huggingface-url> [--drafter auto|none|<drafter>]
magnitude models status [model-id] [--all]
magnitude models load <model-id>
magnitude models stop
magnitude models import <path.gguf> [--drafter auto|none|<drafter>]
magnitude models drafter <model-id> <auto | none | drafter>
magnitude models hide|unhide <model-id>
magnitude models folders list
magnitude models folders add|remove <path>
magnitude connections list
magnitude connections connect <harness> [--set-model <model-id>] [--install-skill]
magnitude connections sync [harness]
magnitude connections remove <harness>
magnitude docs [topic-id]
```

Run `magnitude serve` to host inference in the foreground, or open the desktop app. On Linux and
macOS, `magnitude server setup` runs Magnitude as a service that starts at boot, with Network access
on; `magnitude server remove` removes it and keeps its data. See `magnitude docs remote-access`.
Model, catalog, hardware, and connection setup commands require an existing service; they do not
start one. `magnitude status` reports service and model status without starting anything. Stop a
foreground server with Ctrl+C. Desktop launch-at-login and Quit are available in the desktop app.

`update check` checks for a release; `download` waits until it is prepared; `status` reports the
current update state; `discard` removes a prepared update. None of these commands opens Desktop.
`update install` restarts a running Desktop to install. It refuses to interrupt a running server:
on Linux and macOS a server installs downloaded updates by itself between requests when it can without
a password, and otherwise prints that the update is downloaded. Stop `serve` (or a Windows service
wrapper) first, run `update install`, then start it again. Starting `serve` again does not install a
downloaded update. Failed installation attempts require explicit retry. Linux installation may require
system authorization.

Each command prints only the product information relevant to that operation. Collection commands
use borderless tables when the rows are directly comparable; detail commands use labeled fields.
Exact model and harness IDs are always printed so their output can be used in later commands.

Every `hardware`, `catalog`, and `models` command accepts `--json` and then prints one JSON document
of the same facts instead of text, with exact IDs, structured reasons a model can't run (with the
engine's developer `detail`), and drafter verdicts. Optional fields are omitted when absent. Errors
stay on stderr with a nonzero exit in both modes. No command prompts or reads standard input: where
a choice is needed it takes the desktop's default and prints what it chose.

`catalog` finds and downloads models: catalog assessment progress, reviewed model choices,
recommendation evidence, Hugging Face search and inspection, and downloads. `models` owns what is on
this computer: catalog models, models added from Hugging Face or as a file, models found in other
apps' folders, drafters, hidden models, model locations, and runtime state. Catalog assessment,
downloads, and model loading are background work: observation commands return the current state and
never wait for any of them to settle.

Magnitude scans the Hugging Face cache, LM Studio, the llama.cpp cache, Ollama, and folders you add
for GGUF models, reading headers only and never changing those files. Assessment evaluates every
model for the current hardware, including compatibility, memory fit, serving configuration,
acceleration, and expected speed. See `magnitude docs custom-models` for adding models from Hugging
Face or disk, drafters, and model locations.

`connections connect --install-skill` installs or refreshes the bundled Magnitude skill in the selected
harness's supported user-wide location before applying the connection.
