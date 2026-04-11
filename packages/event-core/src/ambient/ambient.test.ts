import { describe, expect, it, vi } from 'vitest'
import { Effect } from 'effect'
import * as Agent from '../agent'
import * as Ambient from './index'
import * as Projection from '../projection'
import * as Signal from '../signal'
import { AmbientServiceTag } from '../core/ambient-service'

type TestEvent =
  | { type: 'set'; value: number }
  | { type: 'bump'; forkId: string | null; amount: number }

describe('Ambient primitive', () => {
  it('defines, registers, and reads an ambient value via AmbientService.getValue()', async () => {
    const NumberAmbient = Ambient.define<number>({ name: 'TestNumber', initial: 0 })

    const TestAgent = Agent.define<TestEvent>()({
      name: 'TestAgent',
      projections: [],
      workers: []
    })

    const client = await TestAgent.createClient()

    try {
      const value = await client.runEffect(
        Effect.gen(function* () {
          const ambients = yield* AmbientServiceTag
          yield* ambients.register(NumberAmbient)
          yield* ambients.update(NumberAmbient, 42)
          return ambients.getValue(NumberAmbient)
        })
      )

      expect(value).toBe(42)
    } finally {
      await client.dispose()
    }
  })

  it('supports sync ambient reads in projection event handlers', async () => {
    const NumberAmbient = Ambient.define<number>({ name: 'EventReadNumber', initial: 10 })

    const ReaderProjection = Projection.define<TestEvent, { total: number }>()({
      name: 'Reader',
      initial: { total: 0 },
      ambients: [NumberAmbient],
      eventHandlers: {
        set: ({ event, state, ambient }) => ({
          total: state.total + event.value + ambient.get(NumberAmbient)
        })
      }
    })

    const TestAgent = Agent.define<TestEvent>()({
      name: 'TestAgent',
      projections: [ReaderProjection],
      workers: [],
      expose: {
        state: {
          reader: ReaderProjection
        }
      }
    })

    const client = await TestAgent.createClient()

    try {
      await client.send({ type: 'set', value: 5 })

      expect(await client.state.reader.get()).toEqual({ total: 15 })
    } finally {
      await client.dispose()
    }
  })

  it('triggers ambientHandlers when AmbientService.update() is called', async () => {
    const NumberAmbient = Ambient.define<number>({ name: 'ReactiveNumber', initial: 1 })

    const ReactiveProjection = Projection.define<TestEvent, { seen: number[] }>()({
      name: 'Reactive',
      initial: { seen: [] },
      ambients: [NumberAmbient],
      ambientHandlers: (on) => [
        on(NumberAmbient, ({ value, state }) => ({
          seen: [...state.seen, value]
        }))
      ]
    })

    const TestAgent = Agent.define<TestEvent>()({
      name: 'TestAgent',
      projections: [ReactiveProjection],
      workers: [],
      expose: {
        state: {
          reactive: ReactiveProjection
        }
      }
    })

    const client = await TestAgent.createClient()

    try {
      await client.runEffect(
        Effect.flatMap(AmbientServiceTag, (ambients) => ambients.update(NumberAmbient, 2))
      )

      expect(await client.state.reactive.get()).toEqual({ seen: [2] })
    } finally {
      await client.dispose()
    }
  })

  it('supports forked projection ambient reads in event handlers', async () => {
    const NumberAmbient = Ambient.define<number>({ name: 'ForkReadNumber', initial: 3 })

    const ForkReaderProjection = Projection.defineForked<Extract<TestEvent, { type: 'bump' }>, { total: number }>()({
      name: 'ForkReader',
      initialFork: { total: 0 },
      ambients: [NumberAmbient],
      eventHandlers: {
        bump: ({ event, fork, ambient }) => ({
          total: fork.total + event.amount + ambient.get(NumberAmbient)
        })
      }
    })

    const TestAgent = Agent.define<TestEvent>()({
      name: 'TestAgent',
      projections: [ForkReaderProjection],
      workers: [],
      expose: {
        state: {
          forkReader: ForkReaderProjection
        }
      }
    })

    const client = await TestAgent.createClient()

    try {
      await client.send({ type: 'bump', forkId: 'fork-a', amount: 4 })

      expect(await client.state.forkReader.getFork('fork-a')).toEqual({ total: 7 })
    } finally {
      await client.dispose()
    }
  })

  it('supports forked projection ambientHandlers over full ForkedState', async () => {
    const NumberAmbient = Ambient.define<number>({ name: 'ForkReactiveNumber', initial: 1 })

    const ForkReactiveProjection = Projection.defineForked<Extract<TestEvent, { type: 'bump' }>, { total: number }>()({
      name: 'ForkReactive',
      initialFork: { total: 0 },
      ambients: [NumberAmbient],
      eventHandlers: {
        bump: ({ event, fork }) => ({
          total: fork.total + event.amount
        })
      },
      ambientHandlers: (on) => [
        on(NumberAmbient, ({ value, state }) => ({
          forks: new Map(
            [...state.forks.entries()].map(([forkId, forkState]) => [
              forkId,
              { total: forkState.total + value }
            ])
          )
        }))
      ]
    })

    const TestAgent = Agent.define<TestEvent>()({
      name: 'TestAgent',
      projections: [ForkReactiveProjection],
      workers: [],
      expose: {
        state: {
          forkReactive: ForkReactiveProjection
        }
      }
    })

    const client = await TestAgent.createClient()

    try {
      await client.send({ type: 'bump', forkId: null, amount: 2 })
      await client.send({ type: 'bump', forkId: 'fork-b', amount: 5 })

      await client.runEffect(
        Effect.flatMap(AmbientServiceTag, (ambients) => ambients.update(NumberAmbient, 10))
      )

      expect(await client.state.forkReactive.getFork(null)).toEqual({ total: 12 })
      expect(await client.state.forkReactive.getFork('fork-b')).toEqual({ total: 15 })
    } finally {
      await client.dispose()
    }
  })

  it('flushes signals emitted from ambientHandlers to other projections', async () => {
    const NumberAmbient = Ambient.define<number>({ name: 'SignalAmbient', initial: 0 })

    const SourceProjection = Projection.define<TestEvent, { latest: number | null }>()({
      name: 'Source',
      initial: { latest: null },
      ambients: [NumberAmbient],
      signals: {
        changed: Signal.create<{ value: number }>('Source/changed')
      },
      ambientHandlers: (on) => [
        on(NumberAmbient, ({ value, state, emit }) => {
          emit.changed({ value })
          return { latest: value ?? state.latest }
        })
      ]
    })

    const ListenerProjection = Projection.define<TestEvent, { values: number[] }>()({
      name: 'Listener',
      initial: { values: [] },
      signalHandlers: (on) => [
        on(SourceProjection.signals.changed, ({ value, state }) => ({
          values: [...state.values, value.value]
        }))
      ]
    })

    const TestAgent = Agent.define<TestEvent>()({
      name: 'TestAgent',
      projections: [SourceProjection, ListenerProjection],
      workers: [],
      expose: {
        state: {
          listener: ListenerProjection
        }
      }
    })

    const client = await TestAgent.createClient()

    try {
      await client.runEffect(
        Effect.flatMap(AmbientServiceTag, (ambients) => ambients.update(NumberAmbient, 9))
      )

      expect(await client.state.listener.get()).toEqual({ values: [9] })
    } finally {
      await client.dispose()
    }
  })

  it('does not create events when ambients update', async () => {
    const NumberAmbient = Ambient.define<number>({ name: 'NoEventAmbient', initial: 0 })
    const onEvent = vi.fn()

    const PassiveProjection = Projection.define<TestEvent, { count: number }>()({
      name: 'Passive',
      initial: { count: 0 },
      ambients: [NumberAmbient],
      ambientHandlers: (on) => [
        on(NumberAmbient, ({ state }) => ({ count: state.count + 1 }))
      ]
    })

    const TestAgent = Agent.define<TestEvent>()({
      name: 'TestAgent',
      projections: [PassiveProjection],
      workers: []
    })

    const client = await TestAgent.createClient()

    try {
      const unsubscribe = client.onEvent(onEvent)

      await client.runEffect(
        Effect.flatMap(AmbientServiceTag, (ambients) => ambients.update(NumberAmbient, 1))
      )

      expect(onEvent).not.toHaveBeenCalled()

      unsubscribe()
    } finally {
      await client.dispose()
    }
  })

  it('reads the declared initial value without manual registration', async () => {
    const InitialAmbient = Ambient.define<number>({ name: 'InitialAmbient', initial: 7 })

    const InitialProjection = Projection.define<TestEvent, { total: number }>()({
      name: 'Initial',
      initial: { total: 0 },
      ambients: [InitialAmbient],
      eventHandlers: {
        set: ({ event, ambient }) => ({
          total: event.value + ambient.get(InitialAmbient)
        })
      }
    })

    const TestAgent = Agent.define<TestEvent>()({
      name: 'TestAgent',
      projections: [InitialProjection],
      workers: [],
      expose: {
        state: {
          initial: InitialProjection
        }
      }
    })

    const client = await TestAgent.createClient()

    try {
      await client.send({ type: 'set', value: 1 })

      expect(await client.state.initial.get()).toEqual({ total: 8 })
    } finally {
      await client.dispose()
    }
  })
})
