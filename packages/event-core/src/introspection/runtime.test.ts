import { describe, expect, test } from 'bun:test'
import { Effect, Schema } from 'effect'
import { make as makeEventEngine, Service as EventEngineService } from '../event-engine'
import { define } from '../projection'
import { RuntimeIntrospector } from './runtime'

type CounterEvent = { type: 'increment' }

const CounterStateSchema = Schema.Struct({
  count: Schema.Number,
})

describe('runtime introspection', () => {
  test('reads registered projection state without checkpoint capture', async () => {
    const counters = { encode: 0 }
    const CountStateSchema = Schema.transformOrFail(
      CounterStateSchema,
      CounterStateSchema,
      {
        decode: (raw) => Effect.succeed(raw),
        encode: (state) =>
          Effect.sync(() => {
            counters.encode += 1
            return state
          }),
      }
    )

    const CounterProjection = define<CounterEvent>()({
      name: 'Counter',
      state: CountStateSchema,
      initial: { count: 0 },
      eventHandlers: {
        increment: ({ state }) => ({ count: state.count + 1 }),
      },
    })

    const TestEngine = makeEventEngine<CounterEvent>()({
      name: 'RuntimeIntrospectionAgent',
      schemaVersion: 'test',
      projections: [CounterProjection],
      workers: [],
      expose: { state: { counter: CounterProjection } },
    })

    const introspection = await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const engine = yield* EventEngineService
          yield* engine.send({ type: 'increment' })
          const introspection = yield* RuntimeIntrospector
          return yield* introspection.current(null)
        }).pipe(Effect.provide(TestEngine.EngineLayer))
      )
    )

    expect(introspection.engineName).toBe('RuntimeIntrospectionAgent')
    expect(introspection.schemaVersion).toBe('test')
    expect(introspection.projections).toHaveLength(1)
    expect(introspection.projections[0]).toMatchObject({
      name: 'Counter',
      kind: 'global',
      forkId: null,
      state: { count: 1 },
    })
    expect(counters.encode).toBe(0)
  })
})
