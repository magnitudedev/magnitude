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
magnitude catalog show|pull|cancel|remove <model-id>
magnitude catalog recommendations [--preference <value>] [--limit <count>]
magnitude models status [model-id]
magnitude models load <model-id>
magnitude models stop
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

`catalog` owns catalog assessment progress, reviewed model choices, recommendation
evidence, and download operations. `models` owns models on this computer and their current
installation or runtime state. Catalog assessment and model loading are background work:
observation commands return the current state and never wait for either to settle.

Discovery scans existing local Hugging Face caches for usable GGUF models without downloading or
contacting the Hub. Assessment evaluates catalog and discovered models for the current hardware,
including compatibility, memory fit, serving configuration, acceleration, and expected speed.

`connections connect --install-skill` installs or refreshes the bundled Magnitude skill in the selected
harness's supported user-wide location before applying the connection.
