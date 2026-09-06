# Magnitude for Hermes

Native local-model controls and request-scoped inference progress for ordinary
`hermes` and Hermes Desktop. No alternate renderer or host patches.

The package includes Magnitude's usage skill. An existing user-installed Magnitude
skill takes precedence. Installation alone does not install the CLI, download
models, select a provider, start Magnitude, or send a message.

## Use

Run `magnitude setup` and choose Hermes to install and connect the release-matched
companion. The native direct-install form is:

```sh
hermes plugins install https://github.com/magnitudedev/magnitude --ref <distribution-commit> --enable
```

Use the Hermes distribution commit from the corresponding Magnitude release plan,
not the source branch commit. Start ordinary `hermes` after installation. Desktop
discovers both components from the same installation; enable its Desktop component
in Settings → Plugins.

- `/magnitude-setup` displays the canonical onboarding prompt to send to your agent.
- `/magnitude` lists local models.
- `/load-model <model-id>` loads an installed model; `/stop-model` stops it.
- Ordinary terminal progress appears above the editor, once per phase, with final
  timings. Hermes retains its own working row and response parser.
- In Desktop, enable the Desktop component under Settings → Plugins and restart
  the gateway. The Magnitude status-bar button opens model controls. Progress
  follows the focused conversation on its owning backend.

Model controls use Magnitude RPC. Only an explicit model command may start the
service using `magnitude service start`. Progress reads never start or repair it.
A protocol mismatch requires matching CLI/companion versions; a lost mutation
acknowledgement is reported as uncertain and is never automatically replayed.

## Development

Run `bun run build` here to generate the Python wire contract, canonical skill,
and readable Desktop bundle. Python runs in Hermes's own environment; the standard
Hermes installation supplies JSON Schema validation through its MCP dependency.
There is no private Magnitude CLI dependency or additional Python install hook.

The consumer distribution contains only the files listed by the release package's
Hermes content manifest. Release preparation transports an immutable consumer Git
commit in a bundle; publication makes that commit available under the versioned
`magnitude-hermes/<version>` tag. Native installation uses that exact commit with
Hermes's security scanner enabled. The source checkout is not itself an installable
consumer package.

`scripts/accept-integrations.ts` exercises the actual native installer and loader.
Set `HERMES_PYTHON` to the interpreter in your Hermes environment when it differs
from `python3`. It verifies CLI-free setup and skill precedence across reload/removal.
