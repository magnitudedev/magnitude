# Magnitude Telemetry

Magnitude collects anonymous usage telemetry to help us understand how the tool is being used and improve the experience. This data is **completely anonymous** and contains **no personal information**.

## What We Collect

We collect high-level, aggregate usage data:

- **Session counts** — how many times Magnitude is started, session duration
- **Message counts** — number of user messages sent (not the content)
- **Tool usage** — which tools are used and how often (e.g. file edit, file write, shell, search), success/failure counts
- **Lines of code** — number of lines written, added, and removed by tools (not the actual code)
- **Model and provider usage** — which LLM providers and models are selected, token consumption (input/output counts)
- **Agent usage** — number and types of sub-agents spawned (explorer, builder, browser, etc.)

## What We Do NOT Collect

- No message content, prompts, or conversation text
- No file contents, file paths, or directory names
- No shell commands or their output
- No API keys, tokens, or credentials
- No personally identifiable information (names, emails, IPs)
- No code, diffs, or repository information

All telemetry data is anonymous aggregate counts and metadata. GeoIP tracking is disabled — we do not track your location.

## How It Works

On first launch, Magnitude generates a random anonymous ID (CUID2) stored locally in `~/.magnitude/config.json`. This ID is used solely to distinguish unique installations in aggregate counts. It is not linked to any personal identity.

Telemetry events are sent to [PostHog](https://posthog.com), an open-source analytics platform.

## How to Opt Out

You can disable telemetry at any time using either method:

**Environment variable** (recommended for CI/automation):
```bash
export MAGNITUDE_TELEMETRY=0
```

**Config file** (`~/.magnitude/config.json`):
```json
{
  "telemetry": false
}
```

When telemetry is disabled, no data is collected or transmitted.
