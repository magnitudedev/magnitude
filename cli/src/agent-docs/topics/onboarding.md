# Agent-guided Magnitude onboarding

Use this workflow to set up Magnitude entirely through its non-interactive CLI. Keep the user in the
model-selection decision: Magnitude supplies machine-specific evidence, while the user supplies
their speed, capability, context, and storage preferences.

## 1. Start Magnitude

```bash
magnitude --version
magnitude service start
magnitude service status
```

`service start` installs and enables the per-user login service, starts it, and waits only for
service readiness. Hardware discovery and model assessment continue automatically in the
background and do not block service health.

## 2. Choose a model with the user

```bash
magnitude hardware
magnitude catalog recommendations --preference balanced --limit 10
```

Ask whether the user prefers faster, balanced, or smarter output. Use one of `fastest`, `faster`,
`balanced`, `smarter`, or `smartest`. Discuss the displayed speed, memory, context, intelligence,
accuracy, acceleration, and capabilities. If assessment is still running, explain that the current
recommendations may change; do not block or busy-poll.

Use `magnitude catalog list` for a direct list of compatible choices and
`magnitude catalog show <model-id>` for release, architecture, performance, distribution, license,
and source details. Preserve the exact model ID. Recommend a short set of candidates and let the
user choose before downloading one.

## 3. Install and load the chosen model

```bash
magnitude catalog pull <model-id>
magnitude models status <model-id>
magnitude models load <model-id>
magnitude models status <model-id>
```

Installation and loading are admitted background operations. Report meaningful progress from the
focused status command. Continue when installation is `Installed` and runtime is `Ready`. If a
failure is shown, report its actionable message and ask before taking remediation outside the
requested setup scope.

## 4. Connect the current harness

```bash
magnitude connections list
magnitude connections add <harness> --set-current <model-id> --install-skill
```

Identify the harness in which you are currently running and use its canonical ID. Ask only when the
harness is genuinely ambiguous. `--install-skill` atomically replaces that harness's Magnitude skill
with the bundled current version. The connection includes all installed Magnitude models and leaves
unrelated harness settings alone.

When the command prints a handoff command, show it to the user. Do not launch or switch harnesses
unless the user asks. A newly installed skill may require a new harness session before discovery.

Use `magnitude --help`, `magnitude <group> --help`, or `magnitude docs cli` when command discovery is
needed. Do not substitute the interactive `magnitude setup` flow while operating as the onboarding
agent.
