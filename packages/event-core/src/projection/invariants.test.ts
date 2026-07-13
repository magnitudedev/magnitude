import { describe, expect, test } from 'bun:test'
import { Effect, Schema } from 'effect'
import { AmbientServiceTag } from '../core/ambient-service'
import {
  ProjectionSnapshotProjectionInvalid,
  ProjectionSnapshotProjectionSetMismatch,
  Service as EventEngineService
} from '../event-engine'
import { make as makeEventEngine } from '../event-engine'
import { ProjectionBusTag } from '../core/projection-bus'
import { define as defineAmbient } from '../ambient'
import { create as createSignal } from '../signal'
import { define } from './define'

type IncrementEvent = { type: 'increment' }
type EitherResult = { readonly _tag: 'Left'; readonly left: unknown } | { readonly _tag: 'Right'; readonly right: unknown }

const RawCountStateSchema = Schema.Struct({ count: Schema.Number })

function makeCountingCountStateSchema(counters: { encode: number; decode: number }, delay = false) {
  return Schema.transformOrFail(
    RawCountStateSchema,
    RawCountStateSchema,
    {
      decode: (raw) =>
        Effect.gen(function* () {
          counters.decode += 1
          if (delay) yield* Effect.sleep('10 millis')
          return raw
        }),
      encode: (state) =>
        Effect.gen(function* () {
          counters.encode += 1
          if (delay) yield* Effect.sleep('10 millis')
          return state
        }),
    }
  )
}

describe('projection framework invariants', () => {
  test('effectful projection handlers commit on success and leave state unchanged on failure', async () => {
    type EffectEvent =
      | { type: 'succeed' }
      | { type: 'fail' }

    const CounterProjection = define<EffectEvent>()({
      name: 'EffectfulCounter',
      state: RawCountStateSchema,
      initial: { count: 0 },
      eventHandlers: {
        succeed: ({ state }) =>
          Effect.succeed({ count: state.count + 1 }),
        fail: () =>
          Effect.fail('no commit')
      },
    })

    const TestEngine = makeEventEngine<EffectEvent>()({
      name: 'EffectfulHandlerAgent',
      schemaVersion: 'test',
      projections: [CounterProjection],
      workers: [],
      expose: { state: { counter: CounterProjection } },
    })

    const client = await TestEngine.createClient()
    try {
      await client.send({ type: 'succeed' })
      await client.send({ type: 'fail' })

      expect(await client.state.counter.get()).toEqual({ count: 1 })
    } finally {
      await client.dispose()
    }
  })

  test('serializes concurrent ambient and event mutations without hot-path schema traversal', async () => {
    const counters = { encode: 0, decode: 0 }
    const CountStateSchema = makeCountingCountStateSchema(counters, true)
    const IncrementAmbient = defineAmbient<number>({ name: 'ConcurrentIncrement', initial: 0 })

    const CounterProjection = define<IncrementEvent>()({
      name: 'ConcurrentCounter',
      state: CountStateSchema,
      initial: { count: 0 },
      ambients: [IncrementAmbient],
      eventHandlers: {
        increment: ({ state }) => ({ count: state.count + 1 }),
      },
      ambientHandlers: (on) => [
        on(IncrementAmbient, ({ state }) => ({ count: state.count + 1 })),
      ],
    })

    const TestEngine = makeEventEngine<IncrementEvent>()({
      name: 'ConcurrentMutationAgent',
      schemaVersion: 'test',
      projections: [CounterProjection],
      workers: [],
      expose: { state: { counter: CounterProjection } },
    })

    const client = await TestEngine.createClient()
    try {
      const operations: Promise<unknown>[] = []
      for (let i = 0; i < 25; i++) {
        operations.push(client.send({ type: 'increment' }))
        operations.push(client.runEffect(
          Effect.flatMap(AmbientServiceTag, (ambients) =>
            ambients.update(IncrementAmbient, i)
          ) as never
        ))
      }

      await Promise.all(operations)

      expect(await client.state.counter.get()).toEqual({ count: 50 })
      expect(counters).toEqual({ encode: 0, decode: 0 })
    } finally {
      await client.dispose()
    }
  })

  test('uses projection schemas at snapshot boundaries only', async () => {
    const counters = { encode: 0, decode: 0 }
    const CountStateSchema = makeCountingCountStateSchema(counters)

    const CounterProjection = define<IncrementEvent>()({
      name: 'SnapshotCounter',
      state: CountStateSchema,
      initial: { count: 0 },
      eventHandlers: {
        increment: ({ state }) => ({ count: state.count + 1 }),
      },
    })

    const TestEngine = makeEventEngine<IncrementEvent>()({
      name: 'SnapshotBoundaryAgent',
      schemaVersion: 'test',
      projections: [CounterProjection],
      workers: [],
      expose: { state: { counter: CounterProjection } },
    })

    const client = await TestEngine.createClient()
    try {
      await client.send({ type: 'increment' })
      expect(counters).toEqual({ encode: 0, decode: 0 })

      const snapshot = await client.runEffect(
        Effect.flatMap(EventEngineService, (engine) =>
          engine.captureProjectionSnapshot({ index: 0, timestamp: 1 }, 'session-1')
        ) as never
      )
      expect(counters).toEqual({ encode: 1, decode: 0 })

      await client.runEffect(
        Effect.flatMap(EventEngineService, (engine) =>
          Effect.gen(function* () {
            const plan = yield* engine.prepareProjectionSnapshotRestore(snapshot)
            yield* plan.commit
          })
        ) as never
      )
      expect(counters).toEqual({ encode: 1, decode: 1 })
    } finally {
      await client.dispose()
    }
  })

  test('restore preflights every projection before committing any projection', async () => {
    const GoodProjection = define<IncrementEvent>()({
      name: 'Good',
      state: RawCountStateSchema,
      initial: { count: 0 },
      eventHandlers: {},
    })

    const BadProjection = define<IncrementEvent>()({
      name: 'Bad',
      state: Schema.Struct({ label: Schema.String }),
      initial: { label: 'initial' },
      eventHandlers: {},
    })

    const TestEngine = makeEventEngine<IncrementEvent>()({
      name: 'AllOrNothingRestoreAgent',
      schemaVersion: 'test',
      projections: [GoodProjection, BadProjection],
      workers: [],
      expose: { state: { good: GoodProjection } },
    })

    const client = await TestEngine.createClient()
    try {
      const result = await client.runEffect(
        Effect.flatMap(EventEngineService, (engine) =>
          Effect.either(
            engine.prepareProjectionSnapshotRestore({
              sessionId: 'session-1',
              engineName: 'AllOrNothingRestoreAgent',
              schemaVersion: 'test',
              eventCursor: { index: 0, timestamp: 1 },
              projections: {
                Good: { count: 99 },
                Bad: { label: 123 },
              },
            } as never)
          )
        ) as never
      ) as EitherResult

      expect(result._tag).toBe('Left')
      if (result._tag === 'Left') {
        const error = result.left
        expect(error).toBeInstanceOf(ProjectionSnapshotProjectionInvalid)
        if (error instanceof ProjectionSnapshotProjectionInvalid) {
          expect(error.projectionName).toBe('Bad')
        }
      }
      expect(await client.state.good.get()).toEqual({ count: 0 })
    } finally {
      await client.dispose()
    }
  })

  test('snapshot restore ignores legacy schemaVersion metadata', async () => {
    const CounterProjection = define<IncrementEvent>()({
      name: 'VersionIgnoredCounter',
      state: RawCountStateSchema,
      initial: { count: 0 },
      eventHandlers: {},
    })

    const TestEngine = makeEventEngine<IncrementEvent>()({
      name: 'VersionIgnoredAgent',
      schemaVersion: 'current-runtime-version',
      projections: [CounterProjection],
      workers: [],
      expose: { state: { counter: CounterProjection } },
    })

    const client = await TestEngine.createClient()
    try {
      await client.runEffect(
        Effect.flatMap(EventEngineService, (engine) =>
          Effect.gen(function* () {
            const plan = yield* engine.prepareProjectionSnapshotRestore({
              sessionId: 'session-1',
              engineName: 'VersionIgnoredAgent',
              schemaVersion: 'old-runtime-version',
              eventCursor: { index: 0, timestamp: 1 },
              projections: {
                VersionIgnoredCounter: { count: 7 },
              },
            })
            yield* plan.commit
          })
        ) as never
      )

      expect(await client.state.counter.get()).toEqual({ count: 7 })
    } finally {
      await client.dispose()
    }
  })

  test('snapshot restore reports projection set mismatches as typed errors', async () => {
    const CounterProjection = define<IncrementEvent>()({
      name: 'ProjectionSetCounter',
      state: RawCountStateSchema,
      initial: { count: 0 },
      eventHandlers: {},
    })

    const TestEngine = makeEventEngine<IncrementEvent>()({
      name: 'ProjectionSetAgent',
      schemaVersion: 'test',
      projections: [CounterProjection],
      workers: [],
    })

    const client = await TestEngine.createClient()
    try {
      const result = await client.runEffect(
        Effect.flatMap(EventEngineService, (engine) =>
          Effect.either(
            engine.prepareProjectionSnapshotRestore({
              eventCursor: { index: 0, timestamp: 1 },
              projections: {
                ExtraProjection: { count: 1 },
              },
            })
          )
        ) as never
      ) as EitherResult

      expect(result._tag).toBe('Left')
      if (result._tag === 'Left') {
        const error = result.left
        expect(error).toBeInstanceOf(ProjectionSnapshotProjectionSetMismatch)
        if (error instanceof ProjectionSnapshotProjectionSetMismatch) {
          expect(error.missing).toEqual(['ProjectionSetCounter'])
          expect(error.extra).toEqual(['ExtraProjection'])
        }
      }
    } finally {
      await client.dispose()
    }
  })
})
