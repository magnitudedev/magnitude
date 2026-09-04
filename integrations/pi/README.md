# Magnitude for Pi

This Pi package adds Magnitude model management commands and live local-inference progress to Pi's
built-in working row above the editor.

Magnitude installs it automatically when you run `magnitude connections add pi`. You can also install it directly:

```sh
pi install npm:@magnitudedev/pi
```

The extension executes the local `magnitude` CLI for model status, loading, and stopping. During
requests to the `magnitude` provider, it opts into Magnitude progress events and uses Pi's working
row for model loading, prefill, and active-work timing. When the request finishes, a one-line summary
immediately above the editor shows the model's display name, total Pi agent-run time, first-token
latency, and token-weighted generation throughput. Pi extensions execute with your user permissions.

Commands:

- `/load-model [model-id]` — load an installed model
- `/stop-model` — stop the active model

Installed Magnitude models appear in Pi's built-in `/model` selector. To discover, compare, install,
or remove models from the Magnitude catalog, ask the agent; the connection installs Magnitude's
agent skill and the agent uses the `magnitude catalog` and `magnitude connections` commands.

Set `MAGNITUDE_CLI` to an alternate Magnitude executable path when developing or testing the package.

From a Magnitude source checkout, run the complete local connection and TUI flow with:

```sh
bun run dev:pi
```

This installs the package from `integrations/pi`, refreshes Pi's installed Magnitude models and
agent skill through the normal connection flow, builds and runs the checkout's inference runtime,
and keeps the current source CLI available until Pi exits. Exiting Pi stops the development runtime
and restores the service state that existed before launch. The launcher inherits the current
environment; it does not start or configure tracing.
