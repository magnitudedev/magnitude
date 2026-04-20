# Magnitude Telemetry

Magnitude collects minimal anonymous usage telemetry to help us understand how the tool is being used and improve the experience. This data is **completely anonymous** and contains **no personal information**.

## What We Collect (3 Events)

We collect only three high-level events per session:

### `session_start`
- Platform (macOS/Linux/Windows)
- Shell type (bash/zsh/fish/etc.)
- Whether this is a resumed session

### `session_end` (summary of entire session)
- Session duration
- Total number of turns (LLM calls)
- Total number of user messages
- Total input/output tokens across all turns
- **Models used** — per-model breakdown of provider, model ID, and token counts
- Number of memory compaction events

### `provider_connected`
- Provider ID (e.g., anthropic, openai, google)
- Authentication type (OAuth, API key, free tier)

## What We Do NOT Collect

- No message content, prompts, or conversation text
- No file contents, file paths, or directory names
- No shell commands or their output
- No API keys, tokens, or credentials
- No personally identifiable information (names, emails, IPs)
- No code, diffs, or repository information
- No per-tool-call or per-agent tracking
- No lines of code metrics

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
