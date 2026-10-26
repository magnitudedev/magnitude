# Magnitude CLI

Running `magnitude` without a subcommand prints help and exits.
Use Desktop or `magnitude serve` to run inference.

## Installation

Magnitude includes the desktop app and CLI in one installation. No separate npm package is needed.

macOS / Linux:

```sh
curl -fsSL https://magnitude.dev/install.sh | sh
```

Windows PowerShell:

```powershell
irm https://magnitude.dev/install.ps1 | iex
```

On macOS, open a new terminal or SSH session after installation to use `magnitude`. The Windows script
updates PATH in its current PowerShell session. Manual installers are at https://magnitude.dev/download.

The Linux installer asks whether to set up a persistent server service. Choose Yes only for a machine
intended to serve persistently; without an interactive terminal it skips setup. Installation itself does
not require opening Desktop. Use `magnitude serve` for a foreground headless session. On a dedicated
Linux or macOS server, use `magnitude server setup` if the installer has not already set it up;
see `magnitude docs remote-access`.

## Commands

The following commands manage models and agent connections:

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
magnitude catalog search <query> [--limit <count>]
magnitude catalog inspect <huggingface-url | owner/repo | owner/repo:QUANT>
magnitude catalog pull <model-id | owner/repo[:QUANT] | huggingface-url> [--drafter auto|none|<drafter>] [--force]
magnitude models status [model-id] [--all]
magnitude models load <model-id>
magnitude models stop [model-id]
magnitude models drafter <model-id> <auto | none | drafter>
magnitude models hide|unhide <model-id>
magnitude models folders list
magnitude models folders add|remove <path>
magnitude models folders enable|disable <huggingface-cache | lm-studio | llama-cpp | ollama>
magnitude connections list
magnitude connections connect <harness> [--set-model <model-id>] [--install-skill]
magnitude connections sync [harness]
magnitude connections remove <harness>
magnitude hf login|status|logout
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

Every `hardware`, `catalog`, `models`, and `hf` command accepts `--json` and then prints one JSON document
of the same facts instead of text, with exact IDs, structured reasons a model can't run (with the
engine's developer `detail`), and drafter verdicts. Optional fields are omitted when absent. Errors
stay on stderr with a nonzero exit in both modes; with `--json` an error is one JSON document,
`{"error": {"message": "...", "code": "..."}}`, whose `code` is a stable snake_case identifier. No command prompts or reads standard input: where
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

`hf login` signs Magnitude in to Hugging Face for gated models, private repositories, and
higher download limits. It prints a page and a code to enter on any device, waits until the sign-in
finishes, and never opens a browser, so it works over SSH. The sign-in belongs to the Magnitude
service and is shared with the desktop app; Magnitude never asks for a token.
It waits for its own attempt, exits nonzero when that attempt is declined, expires, fails, or is
replaced by a newer sign-in, and Ctrl+C cancels it. With
`--json` it prints the code document (`attemptId`, `verificationUri`, `verificationUriComplete` when Hugging Face returns it, `userCode`, `expiresIn`, `interval`) and
then the account, one JSON line each. `hf status` shows the account and `hf
logout` signs out.

`connections connect --install-skill` installs or refreshes the bundled Magnitude skill in the selected
harness's supported user-wide location before applying the connection.
