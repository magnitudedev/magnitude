import { describe, test, expect } from 'bun:test'
import { Effect, Deferred } from 'effect'
import { define as defineAgent } from '../agent'
import { define } from './define'

type TestEvent =
  | { type: 'turn_started'; forkId: string | null }
  | { type: 'interrupt'; forkId: string | null }
  | { type: 'worker_marker'; forkId: string | null; marker: 'entered' | 'completed' | 'interrupted' }

describe('Worker.define interrupt coordinator', () => {
  test('root handlers observe durable interrupt state', async () => {
    const observed: TestEvent[] = []
    const gate = await Effect.runPromise(Deferred.make<void, never>())

    const Worker = define<TestEvent>()({
      name: 'InterruptCoordinatorDefineWorker',
      eventHandlers: {
        turn_started: (event, publish) =>
          Effect.gen(function* () {
            yield* publish({ type: 'worker_marker', forkId: event.forkId, marker: 'entered' })
            yield* Deferred.await(gate)
            yield* publish({ type: 'worker_marker', forkId: event.forkId, marker: 'completed' })
          }).pipe(
            Effect.onInterrupt(() =>
              publish({ type: 'worker_marker', forkId: event.forkId, marker: 'interrupted' }),
            ),
          ),
      },
    })

    const TestAgent = defineAgent<TestEvent>()({
      name: 'InterruptCoordinatorDefineAgent',
      projections: [],
      workers: [Worker],
    })

    const client = await TestAgent.createClient()
    const unsub = client.onEvent((event) => {
      observed.push(event)
    })

    try {
      await client.send({ type: 'turn_started', forkId: null })
      await client.send({ type: 'interrupt', forkId: null })
      await Effect.runPromise(Deferred.succeed(gate, undefined))
      await new Promise((r) => setTimeout(r, 10))
    } finally {
      unsub()
      await client.dispose()
    }

    expect(observed.filter((e) => e.type === 'worker_marker' && e.marker === 'entered')).toHaveLength(1)
    expect(observed.filter((e) => e.type === 'worker_marker' && e.marker === 'completed')).toHaveLength(0)
    expect(observed.filter((e) => e.type === 'worker_marker' && e.marker === 'interrupted')).toHaveLength(1)
  })
})
