import { describe, expect, test } from 'vitest'
import { Layer, Effect } from 'effect'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { makeProviderRuntimeLive, makeTestResolver } from '@magnitudedev/providers'

import { CodingAgent } from '../../src/coding-agent'
import { ExecutionManagerLive } from '../../src/execution/execution-manager'
import { BrowserServiceLive } from '../../src/services/browser-service'
import { FsLive } from '../../src/services/fs'
import { EphemeralSessionContextTag } from '../../src/agents/types'
import { makeInMemoryChatPersistenceLayer } from '../../src/test-harness/in-memory-persistence'
import { registerApprovalBridge } from '../../src/execution/approval-bridge'
import type { AppEvent, SessionContext } from '../../src/events'
import { CanonicalTurnProjection } from '../../src/projections/canonical-turn'

function waitForEvent(
  transcript: AppEvent[],
  pred: (event: AppEvent) => boolean,
  timeoutMs: number,
): Promise<AppEvent> {
  return new Promise((resolve, reject) => {
    const start = Date.now()
    const poll = () => {
      const found = transcript.find(pred)
      if (found) return resolve(found)
      if (Date.now() - start > timeoutMs) return reject(new Error('Timed out waiting for event'))
      setTimeout(poll, 20)
    }
    poll()
  })
}

describe('exact session-shaped two-turn repro', () => {
  test('turn 2 plain prose should still terminate after malformed turn 1', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'magnitude-exact-two-turn-'))
    const providerRuntime = makeProviderRuntimeLive()
    const persistence = makeInMemoryChatPersistenceLayer()
    const ephemeralSessionContextLayer = Layer.succeed(EphemeralSessionContextTag, {
      disableShellSafeguards: false,
      disableCwdSafeguards: false,
    })

    let callCount = 0
    const seenArgs: unknown[] = []

    const firstResponse = `

<update-task id="explore-web-search" status="completed" />
<message to="user">Great question — here's how it works:

### How Web Search Works

Magnitude has per-provider implementations in [\`packages/agent/src/tools/web-search-*.ts\`](packages/agent/src/tools/web-search-anthropic.ts):
- **Anthropic** — streaming-capable
- **OpenAI** — synchronous
- **Google Gemini** — synchronous (but silently ignores domain filters)

The router at [\`packages/agent/src/tools/web-search.ts\`](packages/agent/src/tools/web-search.ts) determines which one to call based on your active provider.</message>
<yield-tool/>`

    const secondResponse = `

Here's how web search works in Magnitude:

### TL;DR: No automatic fallback — you configure it with one env var.

---

### How it works

Magnitude has separate web search implementations for **Anthropic**, **OpenAI**, and **Google Gemini** ([\`packages/agent/src/tools/web-search*.ts\`](packages/agent/src/tools/web-search.ts)). Each uses the respective provider's native search API (Anthropic's \`computer_search\`, OpenAI's \`web_search\`, Gemini's \`FunctionDeclaration\`).

When you call \`web-search\`, a provider is selected in this order:

1. **\`MAGNITUDE_SEARCH_PROVIDER\` env var** — if set, that's used verbatim (must be \`anthropic\`, \`openai\`, or \`gemini\`)
2. **Auto-detect from your primary model:**
   - \`anthropic\` → Anthropic search
   - \`openai\` → OpenAI search
   - \`google\` / \`google-vertex\` → Gemini search
   - \`google-vertex-anthropic\` / \`amazon-bedrock\` / \`null\` → falls back to **Anthropic**
   - **Everything else** → throws an error

### No automatic fallback for unsupported providers

If you're on Ollama, a local model, or any other provider not in that list — it throws. There's no chain that tries Anthropic automatically. You'd need to set:

\`\`\`bash
export MAGNITUDE_SEARCH_PROVIDER=anthropic
\`\`\`

This makes all web searches route to Anthropic, using your Anthropic API key (or \`ANTHROPIC_API_KEY\` env var). Your primary model is unaffected — it's just the search call that goes elsewhere.

### What you can configure

**\`MAGNITUDE_SEARCH_PROVIDER\`** — the only config option. No persistent config, no CLI flag, no per-session setting — just the env var.

**Per-search options** (passed at call time, not configured):
- \`allowed_domains\` / \`blocked_domains\` — restrict search scope (Anthropic + OpenAI only; **Gemini silently ignores these**)
- \`model\` — override the search model
- \`max_tokens\` — limit output (default 4096)
- \`schema\` — request structured JSON output

Auth for the search provider resolves in order: **OAuth → stored API key → env var** (e.g. \`OPENAI_API_KEY\` for OpenAI, \`GOOGLE_API_KEY\` for Gemini).

---

### Bottom line

If your provider has no web search support, you just point \`MAGNITUDE_SEARCH_PROVIDER\` at one that does. That's it — no magic, no fallback chains. If you want that behavior on all sessions, put it in your shell profile.`

    const runtimeLayer = Layer.mergeAll(
      Layer.provide(ExecutionManagerLive, ephemeralSessionContextLayer),
      Layer.provide(BrowserServiceLive, providerRuntime),
      providerRuntime,
      FsLive,
      makeTestResolver({
        streamResponse: (functionName, args) => {
          seenArgs.push({ functionName, args })
          callCount += 1
          return callCount === 1 ? firstResponse : secondResponse
        },
      }),
      persistence,
    )

    const client = await CodingAgent.createClient(runtimeLayer)
    const transcript: AppEvent[] = []
    const unsubscribe = client.onEvent((event) => {
      transcript.push(event)
    })

    const context: SessionContext = {
      cwd,
      platform: 'macos',
      shell: '/bin/zsh',
      timezone: 'UTC',
      username: 'tester',
      workspacePath: join(cwd, '.workspace'),
      fullName: null,
      git: null,
      folderStructure: '.',
      agentsFile: null,
      skills: null,
    }

    try {
      await client.runEffect(registerApprovalBridge)

      await client.send({
        type: 'session_initialized',
        forkId: null,
        context,
      })

      await new Promise((resolve) => setTimeout(resolve, 100))

      await client.send({
        type: 'user_message',
        messageId: 'user-msg-1',
        forkId: null,
        timestamp: Date.now(),
        content: [{ type: 'text', text: 'trigger exact two-turn repro' }],
        attachments: [],
        mode: 'text',
        synthetic: false,
        taskMode: false,
      })

      const firstCompleted = await waitForEvent(
        transcript,
        (e) => e.type === 'turn_completed' && e.forkId === null,
        3000,
      )
      const firstTurnId = (firstCompleted as Extract<AppEvent, { type: 'turn_completed' }>).turnId

      const canonical = await client.runEffect(
        Effect.flatMap(CanonicalTurnProjection.Tag, (projection) => projection.getFork(null)),
      )
      expect(canonical.lastCompleted?.canonicalMact).toContain('<message to="user"></message>')

      const secondStarted = await waitForEvent(
        transcript,
        (e) => e.type === 'turn_started' && e.forkId === null && e.turnId !== firstTurnId,
        3000,
      )
      const secondTurnId = (secondStarted as Extract<AppEvent, { type: 'turn_started' }>).turnId

      let secondTerminal: AppEvent | null = null
      try {
        secondTerminal = await waitForEvent(
          transcript,
          (e) =>
            (e.type === 'turn_completed' || e.type === 'turn_unexpected_error')
            && 'turnId' in e
            && e.turnId === secondTurnId,
          3000,
        )
      } catch {
        secondTerminal = null
      }

      const secondRawChunks = transcript.filter(
        (e): e is Extract<AppEvent, { type: 'raw_response_chunk' }> =>
          e.type === 'raw_response_chunk' && e.turnId === secondTurnId,
      )
      const secondMessageEnds = transcript.filter(
        (e): e is Extract<AppEvent, { type: 'message_end' }> =>
          e.type === 'message_end' && e.turnId === secondTurnId,
      )

      expect(secondRawChunks.length).toBeGreaterThan(0)
      expect(secondMessageEnds.length).toBeGreaterThan(0)
      expect(seenArgs.length).toBeGreaterThanOrEqual(2)
      expect(secondTerminal).not.toBeNull()
    } finally {
      unsubscribe()
      await client.dispose()
    }
  })
})