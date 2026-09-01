# Agent-guided Magnitude onboarding

Use this workflow after the Magnitude CLI has been installed. Run each command yourself, explain
what is happening, and keep the user informed during long-running work. Preserve the exact model ID
shown by Magnitude; display names are not command arguments.

## 1. Finish installing Magnitude

```bash
magnitude service install
magnitude service start
```

Tell the user that these commands finish the background installation: Magnitude is registered as a
per-user service that starts automatically when they log in, then it is started now. On first start,
Magnitude downloads and prepares the inference engine before reporting that the service is ready.
This may take a few minutes.

`service install` registers login startup and resolves the Magnitude service executable. The
inference engine is acquired by the running service, so do not omit `service start` or claim that
`service install` alone prepares the engine.

## 2. Wait for model preparation, then choose with the user

Poll the authoritative preparation status first:

```bash
magnitude catalog status
```

The status reports two phases:

- **Discovery** scans the Hugging Face model caches already on this computer for complete, usable
  GGUF models. This is a local, read-only scan; it does not download models or contact Hugging Face.
- **Assessment** evaluates Magnitude's catalog choices and the usable models found during discovery
  for this specific hardware. It determines compatibility, memory fit, an appropriate serving
  configuration, available acceleration, and expected generation speed. These results let
  Magnitude exclude models that will not run well and rank the ones that will.

Briefly tell the user that Magnitude is finding local models and checking which ones will run best
on their hardware, then keep them updated as the status changes.

Rerun the command about every 10 seconds and report useful changes without flooding the
conversation. Do not continue until both `Discovery` and `Assessment` explicitly say `Complete`.

Then run balanced recommendations:

```bash
magnitude catalog recommendations --preference balanced --limit 10
```

When the output is stable, present two or three of the best balanced recommendations. Summarize the
displayed speed, memory, context, intelligence, accuracy, acceleration, and relevant capabilities,
then ask the user to choose. Also tell them: “These are balanced choices. If you want, I can instead
show options that prioritize speed or intelligence.” Use another recommendation preference only if
they ask: `fastest`, `faster`, `smarter`, or `smartest`.

Use `magnitude catalog show <model-id>` when the user wants more detail about a candidate. Do not
download a model until the user has chosen one.

## 3. Install the chosen model

```bash
magnitude catalog pull <model-id>
magnitude models status <model-id>
```

`catalog pull` admits a background installation. Poll `models status` about every 30 seconds until
installation is `Installed`. Keep the user updated with the reported percentage and downloaded
bytes. After two progressing samples, you may give a rough remaining-time estimate based on the
observed byte rate; label it as an estimate and do not invent one when progress is stalled or the
total size is unavailable.

If status reports a failure, give the user its actionable message. Ask before taking remediation
outside this onboarding workflow.

## 4. Load the installed model

```bash
magnitude models load <model-id>
magnitude models status <model-id>
```

Loading is a separate background operation. Poll `models status` about every 10 seconds, report
meaningful state or percentage changes, and continue only when runtime is `Ready`. If enough
progress samples are available, a rough observed-rate estimate is acceptable; do not promise a
precise completion time. Magnitude has one active local-model residency slot, so loading this model
may replace another resident local model.

## 5. Offer to connect the current harness

Once the model is ready, identify the harness in which you are currently running and ask whether
the user wants Magnitude connected to it. Do not run `magnitude connections list` merely to
rediscover your own harness. Ask the user only if the harness is genuinely ambiguous.

Supported harnesses and canonical IDs are:

| Harness | ID | Current-session model control |
| --- | --- | --- |
| Magnitude | `magnitude` | Use Magnitude's model picker. |
| Pi | `pi` | Run `/model` or press Ctrl+L; choose provider `magnitude` and the selected model. |
| OpenCode | `opencode` | Run `/models`; choose `magnitude/<model-id>`. |
| Hermes | `hermes` | Run `/model`; choose provider `custom:magnitude` and `<model-id>`. |
| OpenClaw | `openclaw` | Use a fresh Magnitude-agent session; stale session overrides can replace a changed default. |
| Codex | `codex` | Run `/model`; choose `magnitude-local/<model-id>`. Use `/status` to confirm. |
| Claude Code | `claude-code` | Run `/model`; choose `anthropic-local/<model-id>`. |
| Oh My Pi | `oh-my-pi` | Run `/switch` or press Alt+P; choose `magnitude/<model-id>`. |
| Cline | `cline` | Run `/model`; choose provider `openai-compatible` and `<model-id>`. |

Magnitude itself is built in and does not need an external connection command. In Magnitude, tell
the user to select the loaded model in the model picker for the current session.

For any other supported harness, after the user agrees, run:

```bash
magnitude connections add <harness-id> --set-model <model-id> --install-skill
```

This publishes all installed Magnitude models to that harness, persists the chosen model as the
default for ordinary new sessions, and installs or refreshes the bundled Magnitude skill. It does
not change the model used by the already-running onboarding session.

Tell the user both of their choices: they can use the harness-specific control above to switch the
current session if the newly connected model is visible there, or start a new session, which will
select the local model by default. If the current process does not reload its catalog or provider
configuration, use a new session. Show the exact handoff command printed by Magnitude as the safest
way to start one, but do not execute it unless the user asks.

Use `magnitude --help`, `magnitude <group> --help`, or `magnitude docs cli` only when command
discovery is needed. Do not substitute the interactive `magnitude setup` flow while operating as
the onboarding agent.
