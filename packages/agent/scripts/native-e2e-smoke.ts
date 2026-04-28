/**
 * native-e2e-smoke.ts
 *
 * End-to-end smoke test: Fireworks Kimi K2.6 through the full native pipeline.
 * Sends a single user message and streams all TurnEngineEvents to stdout.
 *
 * Usage:
 *   FIREWORKS_API_KEY=xxx bunx --bun native-e2e-smoke.ts
 */

import { Effect, Stream } from 'effect'
import { FetchHttpClient } from '@effect/platform'
import { NativeChatCompletionsCodec } from '/Users/anerli/magnitude/packages/codecs/src/impls/native-chat-completions/codec'
import { OpenAIChatCompletionsDriver } from '/Users/anerli/magnitude/packages/drivers/src/openai-chat-completions/driver'
import { makeNativeTransport, extractAuthToken } from '/Users/anerli/magnitude/packages/agent/src/engine/native-bound-model'
import { createTurnEngine } from '/Users/anerli/magnitude/packages/turn-engine/src/turn-engine'

const apiKey = process.env.FIREWORKS_API_KEY
if (!apiKey) {
  console.log('skipped: no API key')
  process.exit(0)
}

const WIRE_MODEL = 'accounts/fireworks/models/kimi-k2p6'
const ENDPOINT   = 'https://api.fireworks.ai/inference/v1'

// Build codec + driver + transport
const codec = NativeChatCompletionsCodec({
  wireModelName:    WIRE_MODEL,
  defaultMaxTokens: 8192,
  supportsReasoning: true,
  supportsVision:   true,
})
const driver    = OpenAIChatCompletionsDriver
const transport = makeNativeTransport(codec, driver)

// Build a minimal memory: one user inbox message
const memory = [
  {
    type: 'inbox',
    results: [],
    timeline: [
      {
        kind:        'user_message',
        text:        'Say hello in 5 words.',
        timestamp:   Date.now(),
        attachments: [],
      },
    ],
  },
]

const auth = { type: 'api' as const, key: apiKey }
const call = { endpoint: ENDPOINT, authToken: extractAuthToken(auth) }
const options = { thinkingLevel: 'low' as const }
const toolDefs: never[] = []

const engine = createTurnEngine({
  tools: new Map(),
  messageDestination: 'user',
  thoughtKind: 'reasoning',
})

const program = Effect.gen(function* () {
  const responseStream = yield* transport.run(memory, toolDefs, options, call)
  const engineStream = engine.streamWith(responseStream)
  yield* Stream.runForEach(engineStream, (event) =>
    Effect.sync(() => console.log(JSON.stringify(event)))
  )
})

Effect.runPromise(
  program.pipe(Effect.provide(FetchHttpClient.layer))
).catch((err) => {
  console.error('Error:', err)
  process.exit(1)
})
