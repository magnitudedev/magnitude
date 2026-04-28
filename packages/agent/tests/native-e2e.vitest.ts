/**
 * native-e2e.vitest.ts
 *
 * Gated end-to-end test: Fireworks Kimi K2.6 through the full native pipeline.
 *
 * Requires:
 *   RUN_LIVE_TESTS=1 FIREWORKS_API_KEY=xxx bunx --bun vitest run native-e2e
 */

import { describe, it, expect } from 'vitest'
import { Effect, Stream } from 'effect'
import { FetchHttpClient } from '@effect/platform'
import { NativeChatCompletionsCodec } from '@magnitudedev/codecs'
import { OpenAIChatCompletionsDriver } from '@magnitudedev/drivers'
import { makeNativeTransport, extractAuthToken } from '../src/engine/native-bound-model'
import { createTurnEngine, type TurnEngineEvent } from '@magnitudedev/turn-engine'

const WIRE_MODEL = 'accounts/fireworks/models/kimi-k2p6'
const ENDPOINT   = 'https://api.fireworks.ai/inference/v1'

const shouldRun =
  process.env.RUN_LIVE_TESTS === '1' && !!process.env.FIREWORKS_API_KEY

async function runNativeTurn(): Promise<TurnEngineEvent[]> {
  const apiKey = process.env.FIREWORKS_API_KEY!

  const codec = NativeChatCompletionsCodec({
    wireModelName:    WIRE_MODEL,
    defaultMaxTokens: 8192,
    supportsReasoning: true,
    supportsVision:   true,
  })
  const transport = makeNativeTransport(codec, OpenAIChatCompletionsDriver)

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

  const engine = createTurnEngine({
    tools: new Map(),
    messageDestination: 'user',
    thoughtKind: 'reasoning',
  })

  const program = Effect.gen(function* () {
    const responseStream = yield* transport.run(memory, [], { thinkingLevel: 'low' as const }, call)
    const engineStream = engine.streamWith(responseStream)
    return yield* Stream.runCollect(engineStream)
  })

  const chunk = await Effect.runPromise(
    program.pipe(Effect.provide(FetchHttpClient.layer))
  )
  return Array.from(chunk)
}

describe.skipIf(!shouldRun)('native e2e — Fireworks Kimi K2.6', () => {
  it('emits MessageStart, MessageChunk, and TurnEnd(Completed)', async () => {
    const events = await runNativeTurn()

    const tags = events.map((e) => e._tag)

    expect(tags).toContain('MessageStart')
    expect(tags).toContain('MessageChunk')

    const turnEnd = events.find((e) => e._tag === 'TurnEnd')
    expect(turnEnd).toBeDefined()
    expect((turnEnd as any).outcome._tag).toBe('Completed')
  }, 30_000)
})
