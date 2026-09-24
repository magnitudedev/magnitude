# Magnitude CLI

The CLI is headless. Running `magnitude` without a subcommand prints help and exits.
Onboarding lives in the Magnitude desktop app: use Discover to choose models and Connections to
configure external harnesses. Agents can use the following commands to operate inference:

```text
magnitude update [check|status|download|install|discard]
magnitude app open
magnitude serve
magnitude status
magnitude hardware
magnitude catalog status
magnitude catalog list
magnitude catalog show|pull|cancel|remove <model-id>
magnitude catalog recommendations [--preference <value>] [--limit <count>]
magnitude models status [model-id]
magnitude models load <model-id>
magnitude models stop
magnitude connections list
magnitude connections add <harness> [--set-model <model-id>] [--install-skill]
magnitude connections sync [harness]
magnitude connections remove <harness>
magnitude docs [topic-id]
```

Run `magnitude serve` to host inference in the foreground, or open the desktop app.
Model, catalog, hardware, and connection setup commands require an existing service; they do not
start one. `magnitude status` observes the owner and runtime without starting anything. Stop a
foreground server with Ctrl+C. Desktop launch-at-login and Quit are available in the desktop app.

`update check` checks for a release; `download` waits until it is prepared; `status` reports the
current update state; `discard` removes a prepared update. None of these commands opens Desktop.
`update install` restarts a running Desktop with explicit user intent, but refuses to interrupt
a headless server. Stop `serve` first, then either run it again to apply an unattempted prepared
update at startup or run `update install` for a finite installation that leaves the server stopped.
Failed installation attempts require explicit retry. Linux installation may require system authorization.

Each command prints only the product information relevant to that operation. Collection commands
use borderless tables when the rows are directly comparable; detail commands use labeled fields.
Exact model and harness IDs are always printed so their output can be used in later commands.

`catalog` owns catalog assessment progress, reviewed model choices, recommendation
evidence, and download operations. `models` owns models on this computer and their current
installation or runtime state. Catalog assessment and model loading are background work:
observation commands return the current state and never wait for either to settle.

Discovery scans existing local Hugging Face caches for usable GGUF models without downloading or
contacting the Hub. Assessment evaluates catalog and discovered models for the current hardware,
including compatibility, memory fit, serving configuration, acceleration, and expected speed.

`connections add --install-skill` installs or refreshes the bundled Magnitude skill in the selected
harness's supported user-wide location before applying the connection.
