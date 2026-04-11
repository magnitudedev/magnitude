import { describe, test, expect } from 'bun:test'
import { Effect, Deferred } from 'effect'
import { define as defineAgent } from '../agent'
import { defineForked } from './defineForked'
import type { ForkableEvent } from '../projection/defineForked'

type TestEvent =
  | { type: 'turn_started'; forkId: string | null; turn: number }
  | { type: 'interrupt'; forkId: string | null }
  | { type: 'marker'; forkId: string | null; marker: string; turn?: number }

describe('Worker.defineForked root interrupt follow-up', () => {
  test('root follow-up handlers run correctly after first root interrupt', async () => {
    const observed: TestEvent[] = []
    const firstEntered = await Effect.runPromise(Deferred.make<void, never>())
    const releaseFirst = await Effect.runPromise(Deferred.make<void, never>())
    const secondEntered = await Effect.runPromise(Deferred.make<void, never>())

    const RootWorker = defineForked<TestEvent & ForkableEvent>()({
      name: 'RootFollowupWorker',
      forkLifecycle: {
        activateOn: 'turn_started',
      },
      eventHandlers: {
        turn_started: (event, publish) =>
          Effect.gen(function* () {
            yield* publish({ type: 'marker', forkId: event.forkId, marker: 'entered', turn: event.turn })
            if (event.turn === 1) {
              yield* Deferred.succeed(firstEntered, undefined)
              yield* Deferred.await(releaseFirst)
            } else if (event.turn === 2) {
              yield* Deferred.succeed(secondEntered, undefined)
            }
            yield* publish({ type: 'marker', forkId: event.forkId, marker: 'completed', turn: event.turn })
          }).pipe(
            Effect.onInterrupt(() =>
              publish({ type: 'marker', forkId: event.forkId, marker: 'interrupted', turn: event.turn }),
            ),
          ),
      },
    })

    const TestAgent = defineAgent<TestEvent & ForkableEvent>()({
      name: 'RootFollowupAgent',
      projections: [],
      workers: [RootWorker],
    })

    const client = await TestAgent.createClient()
    const unsub = client.onEvent((event) => observed.push(event))

    try {
      await client.send({ type: 'turn_started', forkId: null, turn: 1 })
      await Effect.runPromise(Deferred.await(firstEntered))
      await client.send({ type: 'interrupt', forkId: null })
      await new Promise((r) => setTimeout(r, 10))
      await Effect.runPromise(Deferred.succeed(releaseFirst, undefined))
      await new Promise((r) => setTimeout(r, 10))

      await client.send({ type: 'turn_started', forkId: null, turn: 2 })
      await new Promise((r) => setTimeout(r, 20))
    } finally {
      unsub()
      await client.dispose()
    }

    const turn1Interrupted = observed.some(
      (e) => e.type === 'marker' && e.marker === 'interrupted' && e.turn === 1,
    )
    const turn2Entered = observed.some(
      (e) => e.type === 'marker' && e.marker === 'entered' && e.turn === 2,
    )
    const turn2Completed = observed.some(
      (e) => e.type === 'marker' && e.marker === 'completed' && e.turn === 2,
    )
    const turn2Interrupted = observed.some(
      (e) => e.type === 'marker' && e.marker === 'interrupted' && e.turn === 2,
    )

    expect(turn1Interrupted).toBe(true)
    expect(turn2Entered).toBe(true)
    expect(turn2Completed).toBe(true)
    expect(turn2Interrupted).toBe(false)
  })
})
