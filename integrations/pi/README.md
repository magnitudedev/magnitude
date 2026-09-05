# Magnitude for Pi

This Pi package adds Magnitude model management commands and live local-inference progress to Pi's
built-in working row above the editor.

Requires Pi 0.83.0 or newer. Install from your terminal:

```sh
pi install npm:@magnitudedev/pi-extension
```

On the next fresh interactive Pi launch, Magnitude asks **“Set up local models with Magnitude now?”**
Yes sends this prompt to your current Pi agent and starts onboarding:

> Set up local models for me with the Magnitude CLI. Install it with `npm i -g @magnitudedev/cli` (or my package manager), then run `magnitude docs onboarding` and follow the instructions.

No or Escape leaves “You can set up local models anytime with `/magnitude-setup`.” in the conversation.
The offer is remembered per Pi profile across restarts, reloads, and package updates. Existing
Magnitude model configurations, non-interactive modes, conversations, and startup prompts are left
alone. Run `/magnitude-setup` whenever you want to start onboarding yourself.

The package includes the Magnitude usage skill; an already-loaded skill of that name takes
precedence without a collision warning. `--no-skills` disables this fallback too. The CLI is not
an npm dependency and is not required to load the extension. Your Pi agent needs a working model
and credentials to carry out onboarding.

If you already have Magnitude and a local model, connect directly:

```sh
magnitude connections add pi
```

Restart Pi or run `/reload` after connecting. `PI_CODING_AGENT_DIR`, if set, is honored.
Installing the package directly does not configure provider models,
install the Magnitude CLI, or start its service; the connection command
does all connection configuration and can be run safely after a standalone package install.

The extension bundles Magnitude's private SDK. Model status, loading, and stopping use the existing
RPC endpoint at port 10100. When necessary the SDK runs `magnitude service start`; it does not use
CLI output as a model API. If `/load-model` or `/stop-model` encounters a protocol mismatch, the
extension runs `magnitude connections sync pi` once to install the exact plugin version selected by
the installed CLI, then reloads Pi. Retry your model command after reload; it is never replayed
automatically. Failed sync reports the error without reloading or looping. Manually owned incompatible
packages are not replaced. Autocomplete and inference callbacks never trigger an update. During
requests to the `magnitude` provider, it opts into Magnitude progress events and uses Pi's working
row for model loading, prefill, and active-work timing. When Pi successfully completes a run, a one-line summary
immediately above the editor shows the model's display name, total Pi agent-run time, first-token
latency, and token-weighted generation throughput. Pi extensions execute with your user permissions.

Commands:

- `/magnitude-setup` — ask your current agent to set up Magnitude local models
- `/load-model [model-id]` — load an installed model
- `/stop-model` — stop the active model

Installed Magnitude models appear in Pi's built-in `/model` selector. To discover, compare, install,
or remove models from the Magnitude catalog, ask the agent; the connection installs Magnitude's
agent skill and the agent uses the `magnitude catalog` and `magnitude connections` commands.

Set `MAGNITUDE_CLI` to an alternate Magnitude executable path when developing or testing the package.
`magnitude connections add pi` selects an exact compatible package and verifies its bundled contents.
The SDK is a build-time workspace dependency, not a separately published or installed package.

From a Magnitude source checkout, run the complete local connection and TUI flow with:

```sh
bun run dev:pi
```

This builds and installs the package from `integrations/pi` into an isolated Pi configuration,
with temporary connection receipts and agent skill. Pi runs in the caller's working directory.
It uses the normal connection service,
builds and runs the checkout's inference runtime,
and keeps the current source CLI available until Pi exits. Exiting Pi stops the development runtime
and restores the service state that existed before launch. The launcher inherits the current
environment; it does not start or configure tracing.
