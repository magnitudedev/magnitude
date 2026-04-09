# Web Search Capture Script

Standalone raw-capture harness for Magnitude web-search providers.

## What it does

- Runs provider web search through Magnitude's existing adapters (`src/tools/web-search*.ts`)
- Captures outbound request details, raw responses, stream chunks, and normalized results
- Writes one artifact directory per provider/auth run

## Providers covered

- OpenAI (`api`, `oauth`)
- OpenRouter
- Vercel AI Gateway
- GitHub Copilot
- Google Gemini
- Anthropic (API key path, plus optional stream mode)

## Run

From `packages/agent`:

```bash
bun run scripts/web-search-capture.ts --provider all
```

Or via package script:

```bash
bun run web-search:capture --provider all
```

## Options

```text
--provider <openai|openrouter|vercel|github-copilot|gemini|anthropic|all>
--query "<text>"
--out <dir>
--openai-auth <api|oauth|both>
--stream-anthropic
--direct-adapter
--system "<text>"
--allowed-domain <domain>      # repeatable
--blocked-domain <domain>      # repeatable
```

## Credentials

By default, the script reads provider credentials from:

- `~/.magnitude/auth.json`

Environment variables still override those values when present:

- OpenAI API: `OPENAI_API_KEY`
- OpenAI OAuth: `OPENAI_OAUTH_TOKEN` (or `OPENAI_ACCESS_TOKEN`)
- OpenAI account id (optional for OAuth): `OPENAI_ACCOUNT_ID`
- OpenRouter: `OPENROUTER_API_KEY`
- Vercel: `AI_GATEWAY_API_KEY` (or `VERCEL_API_KEY`)
- GitHub Copilot OAuth: `GITHUB_COPILOT_TOKEN` (or `COPILOT_OAUTH_TOKEN`)
- Google Gemini: `GOOGLE_API_KEY` (or `GEMINI_API_KEY`)
- Anthropic API: `ANTHROPIC_API_KEY`

Missing credentials are recorded as `auth-missing` artifacts (no network attempt).

## Output layout

Default root:

```text
packages/agent/tmp/web-search-captures/<timestamp>/
```

Per run:

- `manifest.json`
- `request.json`
- `response.json`
- `response.raw.txt`
- `stream.ndjson`
- `normalized-result.json`
- `error.json`

Run root:

- `index.json` summary across attempted runs

## Notes

- Default mode runs through `webSearch()` routing with temporary one-provider `ProviderState`/`ProviderAuth` layers.
- Copilot captures now record and validate direct `/responses` web-search signals:
  - request includes `tools: [{ type: "web_search" }]`
  - request includes `include: ["web_search_call.action.sources"]`
  - response contains `web_search_call.action.sources` and/or `url_citation`
  - manifest notes include `copilot:auth-source=<env|stored>`, plus returned source/citation booleans
- Vercel default capture uses a dedicated AI SDK path (`ai` + `@ai-sdk/openai`) against AI Gateway to avoid OpenAI SDK interception incompatibilities on this provider.
- Vercel runs are flagged as `capture-error` (not green) when diagnostics show requested web-search tools were unsupported or dropped by downstream execution.
- `--direct-adapter` bypasses router selection and calls provider-specific direct adapter runners in `scripts/web-search-capture/runners/` for targeted debugging.
- Secrets are redacted in persisted artifacts (`Authorization`, API keys, OAuth tokens, ChatGPT account id).
