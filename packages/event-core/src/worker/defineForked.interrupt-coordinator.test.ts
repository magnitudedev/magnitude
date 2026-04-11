import { describe, test, expect } from 'bun:test'
import { Effect, Deferred } from 'effect'
import { define as defineAgent } from '../agent'
import { defineForked } from './defineForked'
import type { ForkableEvent } from '../projection/defineForked'

type TestEvent =
  | { type: 'turn_started'; forkId: string | null }
  | { type: 'fork_completed'; forkId: string | null }
  | { type: 'interrupt'; forkId: string | null }
  | { type: 'worker_marker'; forkId: string | null; marker: 'entered' | 'completed' | 'interrupted' }

describe('Worker.defineForked interrupt coordinator', () => {
  test('interrupt lands before handler waits and still interrupts queued fork work', async () => {
    const observed: TestEvent[] = []
    const handlerGate = await Effect.runPromise(Deferred.make<void, never>())

    const RaceWorker = defineForked<TestEvent & ForkableEvent>()({
      name: 'InterruptCoordinatorRaceWorker',
      forkLifecycle: {
        activateOn: 'turn_started',
      },
      eventHandlers: {
        turn_started: (event, publish) =>
          Effect.gen(function* () {
            yield* publish({ type: 'worker_marker', forkId: event.forkId, marker: 'entered' })
            yield* Deferred.await(handlerGate)
            yield* publish({ type: 'worker_marker', forkId: event.forkId, marker: 'completed' })
          }).pipe(
            Effect.onInterrupt(() =>
              publish({ type: 'worker_marker', forkId: event.forkId, marker: 'interrupted' }),
            ),
          ),
      },
    })

    const TestAgent = defineAgent<TestEvent & ForkableEvent>()({
      name: 'InterruptCoordinatorRaceAgent',
      projections: [],
      workers: [RaceWorker],
    })

    const client = await TestAgent.createClient()
    const unsub = client.onEvent((event) => {
      observed.push(event)
    })

    try {
      await client.send({ type: 'turn_started', forkId: 'abc' })
      await client.send({ type: 'interrupt', forkId: 'abc' })
      await Effect.runPromise(Deferred.succeed(handlerGate, undefined))
      await new Promise((r) => setTimeout(r, 10))
    } finally {
      unsub()
      await client.dispose()
    }

    expect(observed.filter((e) => e.type === 'worker_marker' && e.marker === 'entered')).toHaveLength(1)
    expect(observed.filter((e) => e.type === 'worker_marker' && e.marker === 'completed')).toHaveLength(0)
    expect(observed.filter((e) => e.type === 'worker_marker' && e.marker === 'interrupted')).toHaveLength(1)
  })

  test('old interrupt state does not poison later execution on same fork id', async () => {
    const observed: TestEvent[] = []
    const firstGate = await Effect.runPromise(Deferred.make<void, never>())
    const secondGate = await Effect.runPromise(Deferred.make<void, never>())

    let execution = 0

    const Worker = defineForked<TestEvent & ForkableEvent>()({
      name: 'InterruptCoordinatorReuseWorker',
      forkLifecycle: {
        activateOn: 'turn_started',
        completeOn: 'fork_completed',
      },
      eventHandlers: {
        turn_started: (event, publish) =>
          Effect.gen(function* () {
            execution += 1
            const gate = execution === 1 ? firstGate : secondGate
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

    const TestAgent = defineAgent<TestEvent & ForkableEvent>()({
      name: 'InterruptCoordinatorReuseAgent',
      projections: [],
      workers: [Worker],
    })

    const client = await TestAgent.createClient()
    const unsub = client.onEvent((event) => {
      observed.push(event)
    })

    try {
      await client.send({ type: 'turn_started', forkId: 'abc' })
      await client.send({ type: 'interrupt', forkId: 'abc' })
      await client.send({ type: 'fork_completed', forkId: 'abc' })
      await Effect.runPromise(Deferred.succeed(firstGate, undefined))

      await client.send({ type: 'turn_started', forkId: 'abc' })
      await Effect.runPromise(Deferred.succeed(secondGate, undefined))
      await new Promise((r) => setTimeout(r, 10))
    } finally {
      unsub()
      await client.dispose()
    }

    expect(observed.filter((e) => e.type === 'worker_marker' && e.marker === 'interrupted')).toHaveLength(1)
    expect(observed.filter((e) => e.type === 'worker_marker' && e.marker === 'completed')).toHaveLength(1)
  })
})
