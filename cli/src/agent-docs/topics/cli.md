# Magnitude CLI

Run `magnitude` without a subcommand for the interactive experience. The
non-interactive command surface is:

```text
magnitude update
magnitude service install|uninstall|start|stop|status
magnitude catalog list
magnitude catalog pull|remove|cancel <model-id>
magnitude models status
magnitude models load <model-id>
magnitude models stop
magnitude connections list
magnitude connections add <harness> [--set-current <model-id>]
magnitude connections remove <harness>
magnitude connections sync [harness]
magnitude docs [topic-id]
```

`catalog` owns model discovery and on-disk acquisition. `catalog pull` installs
a model or brings an installed one up to date; pulling a model that is already
current succeeds and reports that. `models` owns runtime residency.
`magnitude models stop` takes no model ID and stops the active local model.

Every non-interactive leaf command accepts `--json`, before or after the
subcommand. JSON success is written as one document to stdout. JSON failure is
written as one `{ "error": ... }` document to stderr with a nonzero exit code.
