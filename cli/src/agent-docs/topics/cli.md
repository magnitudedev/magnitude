# Magnitude CLI

Running `magnitude` without a subcommand prints help and exits.
Use Desktop or `magnitude serve` to run inference.

## Installation

The desktop app includes the CLI. Download it from https://magnitude.dev/download, or install it from
a terminal:

```sh
curl -fsSL https://magnitude.dev/install.sh | sh -s -- --desktop
```

```powershell
& ([scriptblock]::Create((irm https://magnitude.dev/install.ps1))) -Desktop
```

For servers and headless use, install the CLI only, without the app:

```sh
curl -fsSL https://magnitude.dev/install.sh | sh
```

```powershell
irm https://magnitude.dev/install.ps1 | iex
```

No npm package is needed. Open a new terminal or SSH session after installation to use `magnitude`;
the Windows script also updates PATH in its current PowerShell session. See `magnitude docs headless`
for the CLI-only install, running `magnitude serve` in the background at login, updates, and removal.

## Commands

The following commands manage models and agent connections:

```text
magnitude update [check|status|download|install|discard]
magnitude app open
magnitude serve
magnitude status
magnitude hardware
magnitude models list [--all]
magnitude models recommend [--country <code>] [--limit <count>]
magnitude models search <query> [--limit <count>]
magnitude models show <model-id | owner/repo[:QUANT] | huggingface-url>
magnitude models pull <model-id | owner/repo[:QUANT] | huggingface-url> [--drafter auto|none|<drafter>] [--force] [--only projector|drafter]
magnitude models cancel|rm <model-id>
magnitude models load <model-id>
magnitude models stop [model-id]
magnitude models drafter <model-id> <auto | none | drafter>
magnitude models hide|unhide <model-id>
magnitude models folders list
magnitude models folders add|remove <path>
magnitude models folders enable|disable <huggingface-cache | lm-studio | llama-cpp | ollama>
magnitude models families
magnitude connections list
magnitude connections connect <harness> [--set-model <model-id>] [--install-skill]
magnitude connections sync [harness]
magnitude connections remove <harness>
magnitude hf login|status|logout
magnitude docs [topic-id]
```

Run `magnitude serve` to host inference in the foreground, or open the desktop app. To keep `serve`
running in the background, see `magnitude docs headless`; for other devices, see
`magnitude docs remote-access`. While the desktop app is open it serves, and `serve` waits until it
quits. Model, hardware, and connection setup commands require an existing service; they do not
start one. `magnitude status` reports service and model status without starting anything. Stop a
foreground server with Ctrl+C. Desktop launch-at-login and Quit are available in the desktop app.

`update check` checks for a release; `download` waits until it is prepared; `status` reports the
current update state; `discard` removes a prepared update. None of these commands opens Desktop.
`update install` restarts a running Desktop to install. A CLI-only install updates itself while
`serve` runs: it downloads updates, waits until no inference request is running, then installs and
restarts in place. With `autoDownloadUpdates` set to `false`, `serve` only prints that an update is
available; run `update install` to install it. Failed installation attempts require explicit retry.
Desktop updates on Linux may require system authorization.

Each command prints only the product information relevant to that operation. Collection commands
use borderless tables when the rows are directly comparable; detail commands use labeled fields.
Exact model and harness IDs are always printed so their output can be used in later commands.

Every `hardware`, `models`, and `hf` command accepts `--json` and then prints one JSON document
of the same facts instead of text, with exact IDs, structured reasons a model can't run (with the
engine's developer `detail`), and drafter verdicts. Optional fields are omitted when absent. Errors
stay on stderr with a nonzero exit in both modes; with `--json` an error is one JSON document,
`{"error": {"message": "...", "code": "..."}}`, whose `code` is a stable snake_case identifier. No command prompts or reads standard input: where
a choice is needed it takes the desktop's default and prints what it chose.

`models` finds, downloads, and runs models. `models recommend` lists the tested models for this
computer at the quant chosen for each, with every quant of its ladder; `models search` searches
Hugging Face, with each result's base model and how its recommended quant runs here; `models show`
shows every quant of a model with its bit width, size, speed, served context and placement, and for
the chosen quant the bytes it holds on the device, in host RAM and reads from disk, and its decode
speed at each assessed context depth. `models list` shows what is on this computer (catalog models,
models added from Hugging Face or as a file, and models found in other apps' folders) with the
assessment progress, and `models families` the model families Magnitude runs. Assessment,
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
