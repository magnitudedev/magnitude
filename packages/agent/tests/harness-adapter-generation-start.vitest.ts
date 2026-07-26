import { Effect } from 'effect'
import { describe, expect, it } from 'vitest'
import type { AppEvent } from '../src/events'
import { createHarnessAdapter } from '../src/execution/harness-adapter'

describe('harness adapter generation boundary', () => {
  it('publishes one generation start before the first semantic output event', async () => {
    const events: AppEvent[] = []
    const adapter = createHarnessAdapter({
      forkId: null,
      turnId: 'turn-1',
      chainId: 'chain-1',
      roleId: 'leader',
      defaultProseDest: { kind: 'user' },
      publish: (event) => Effect.sync(() => {
        events.push(event)
      }),
      identicalResponseTracker: null,
      retryCount: 0,
      maxRetries: 3,
      resolveToolKey: () => undefined,
    })

    await Effect.runPromise(adapter.processEvent({ _tag: 'ThoughtStart', level: 'high' }))
    await Effect.runPromise(adapter.processEvent({ _tag: 'ThoughtEnd' }))
    await Effect.runPromise(adapter.processEvent({ _tag: 'MessageStart' }))

    expect(events.map((event) => event.type)).toEqual([
      'model_generation_started',
      'thinking_start',
      'thinking_end',
      'message_start',
    ])
    expect(events.filter((event) => event.type === 'model_generation_started')).toEqual([{
      type: 'model_generation_started',
      forkId: null,
      turnId: 'turn-1',
      chainId: 'chain-1',
    }])
  })
})
