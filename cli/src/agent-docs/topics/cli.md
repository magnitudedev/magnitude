# Magnitude CLI

Run `magnitude` without a subcommand for the interactive experience. Use these commands for a
non-interactive shell workflow:

```text
magnitude update
magnitude service install|uninstall|start|stop|status
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

Each command prints only the product information relevant to that operation. Collection commands
use borderless tables when the rows are directly comparable; detail commands use labeled fields.
Exact model and harness IDs are always printed so their output can be used in later commands.

`catalog` owns model discovery and assessment progress, reviewed model choices, recommendation
evidence, and download operations. `models` owns models on this computer and their current
installation or runtime state. Catalog assessment and model loading are background work:
observation commands return the current state and never wait for either to settle.

Discovery scans existing local Hugging Face caches for usable GGUF models without downloading or
contacting the Hub. Assessment evaluates catalog and discovered models for the current hardware,
including compatibility, memory fit, serving configuration, acceleration, and expected speed.

`connections add --install-skill` installs or refreshes the bundled Magnitude skill in the selected
harness's supported user-wide location before applying the connection.
