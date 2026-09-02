# Agent-guided Magnitude onboarding

You are guiding the user through onboarding in an interactive conversation, not silently executing
a batch of commands. Run each command yourself. Before each major action, briefly explain what you
are about to do and why. While commands or background work are running, narrate meaningful progress
so the user knows what is happening. When the user needs to make a decision, explain the relevant
options and tradeoffs in clear, concise language, then ask a focused question. Welcome questions
throughout the process and answer them as they arise. Preserve the exact model ID shown by
Magnitude; display names are not command arguments. Use `magnitude docs` to answer
Magnitude-specific questions.

## 1. Finish installing Magnitude

```bash
magnitude service install
magnitude service start
```

Briefly explain that Magnitude is being registered as a per-user login service and started now. Its
first start downloads and prepares the inference engine, which may take a few minutes.

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

Briefly explain that Magnitude is finding local models and checking which ones fit this hardware,
then report meaningful status changes.

Rerun the command about every 10 seconds and report useful changes without flooding the
conversation. Continue only when both `Discovery` and `Assessment` report `Complete`.

Then run balanced recommendations:

```bash
magnitude catalog recommendations --preference balanced --limit 10
```

Present two or three of the best balanced recommendations. Summarize the
displayed speed, memory, context, intelligence, accuracy, acceleration, and relevant capabilities,
then ask the user to choose. Frame them as balanced choices and briefly offer a speed- or
intelligence-leaning comparison if useful.

Use `faster` when the user ordinarily asks for faster options and `smarter` when they ordinarily ask
for smarter options. These shift the tradeoff while still giving meaningful consideration to both
speed and intelligence. If the user asks for “fast and smart options,” compare `faster` and
`smarter`; do not interpret that as a request for the extremes.

Treat `fastest` and `smartest` as explicit extremes, not the normal next choices. Use `fastest` only
when the user clearly wants maximum speed and cares little about intelligence; explain that it
prioritizes speed and gives intelligence only limited consideration. Use `smartest` only when the
user clearly wants maximum intelligence and is willing to accept slow generation; explain that it
prioritizes intelligence and gives speed only limited consideration. If that extreme intent is not
clear, use `faster` or `smarter` instead.

Use `magnitude catalog show <model-id>` when the user wants more detail about a candidate. Do not
download a model until the user has chosen one.

## 3. Install the chosen model

```bash
magnitude catalog pull <model-id>
magnitude models status <model-id>
```

`catalog pull` admits a background installation. Poll `models status` until installation is
`Installed`. Adjust the polling interval based on observed progress and the estimated remaining
time so the user stays informed without receiving repetitive updates. Keep them updated with
meaningful changes in the reported percentage and downloaded bytes. After two progressing samples,
you may give a rough remaining-time estimate based on the observed byte rate; label it as an
estimate and do not invent one when progress is stalled or the total size is unavailable.

If status reports a failure, give the user its actionable message. Ask before taking remediation
outside this onboarding workflow.

## 4. Load the installed model

```bash
magnitude models load <model-id>
magnitude models status <model-id>
```

Loading is a separate background operation. Poll `models status` until runtime is `Ready`, adjusting
the polling interval based on observed progress and the estimated remaining time. Report meaningful
state or percentage changes often enough to keep the user informed without flooding them with
repetitive updates. If enough progress samples are available, a rough observed-rate estimate is
acceptable; do not promise a precise completion time.

## 5. Offer to connect the current harness

Once the model is ready, offer to connect Magnitude to the current harness. Other supported
harnesses can be connected instead or in addition.

Supported harnesses and canonical IDs are:

| Harness | ID | Handoff after the model is ready |
| --- | --- | --- |
| Magnitude | `magnitude` | The loaded model is selected automatically. |
| Pi | `pi` | Switch this session with `/model` or Ctrl+L, then choose provider `magnitude` and the selected model. |
| OpenCode | `opencode` | Exit the running process with Ctrl+C, then run the printed launch command. |
| Hermes | `hermes` | Exit the running process with Ctrl+C, then run the printed launch command. |
| OpenClaw | `openclaw` | Exit the running process with Ctrl+C, then run the printed launch command; its dedicated Magnitude-agent session avoids stale model overrides. |
| Codex | `codex` | Exit the Codex process with Ctrl+C, then run the printed launch command. |
| Claude Code | `claude-code` | Exit the running process with Ctrl+C, then run the printed launch command. |
| Oh My Pi | `oh-my-pi` | Exit the running process with Ctrl+C, then run the printed launch command. |
| Cline | `cline` | Exit the running process with Ctrl+C, then run the printed launch command. |

Magnitude itself is built in and does not need an external connection command. In Magnitude, the
loaded model is already selected; no picker or relaunch is needed.

For any other supported harness, offer to connect it and select the chosen model. After the user
agrees, run:

```bash
magnitude connections add <harness-id> --set-model <model-id> --install-skill
```

This publishes all installed Magnitude models to that harness, selects the chosen model in its
configuration, and installs or refreshes the bundled Magnitude skill.
The skill gives the harness agent instructions for using the Magnitude CLI to manage local models
later.

Give one clear primary action for the current harness:

- **Pi:** switch in place with `/model` or Ctrl+L. The printed `pi` command in another terminal is a
  secondary option.
- **Every other external harness:** restart the harness process by exiting with Ctrl+C and reopening
  it with the printed launch command.

The printed launch command uses the harness's ordinary command name, not an absolute executable
path. Show it exactly, but do not execute it unless the user asks.

Use `magnitude --help`, `magnitude <group> --help`, or `magnitude docs cli` only when command
discovery is needed. Do not substitute the interactive `magnitude setup` flow while operating as
the onboarding agent.
