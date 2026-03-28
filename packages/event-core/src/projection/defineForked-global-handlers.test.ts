import { describe, test, expect } from 'bun:test'
import { Effect } from 'effect'
import { define as defineAgent } from '../agent'
import { define } from './define'
import { defineForked, type ForkableEvent, type ForkedState } from './defineForked'

type TestEvent =
  | { type: 'seed'; forkId: string | null; value: number }
  | { type: 'hit'; forkId: string | null; value: number }
  | { type: 'xread'; forkId: string | null; otherForkId: string | null }

describe('defineForked global handlers + cross-fork read', () => {
  test('global event handler receives full forks map', async () => {
    const seenSizes: number[] = []

    const P1 = defineForked<TestEvent & ForkableEvent, { value: number }>()({
      name: 'P1',
      initialFork: { value: 0 },
      eventHandlers: {
        seed: ({ event }) => ({ value: event.value }),
      },
      globalEventHandlers: {
        hit: ({ state }) => {
          seenSizes.push(state.forks.size)
          return state
        },
      },
    })

    const Agent = defineAgent<TestEvent & ForkableEvent>()({
      name: 'GlobalFullForkMapAgent',
      projections: [P1],
      workers: [],
    })

    const client = await Agent.createClient()
    try {
      await client.send({ type: 'seed', forkId: null, value: 1 })
      await client.send({ type: 'seed', forkId: 'a', value: 2 })
      await client.send({ type: 'hit', forkId: null, value: 0 })

      expect(seenSizes).toEqual([2])
    } finally {
      await client.dispose()
    }
  })

  test('global event handler can update multiple forks', async () => {
    const P2 = defineForked<TestEvent & ForkableEvent, { value: number }>()({
      name: 'P2',
      initialFork: { value: 0 },
      eventHandlers: {
        seed: ({ event }) => ({ value: event.value }),
      },
      globalEventHandlers: {
        hit: ({ event, state }) => {
          const forks = new Map(state.forks)
          const root = forks.get(null) ?? { value: 0 }
          const b = forks.get('b') ?? { value: 0 }
          forks.set(null, { value: root.value + event.value })
          forks.set('b', { value: b.value + event.value })
          return { forks }
        },
      },
    })

    const Agent = defineAgent<TestEvent & ForkableEvent>()({
      name: 'GlobalUpdateMultipleForksAgent',
      projections: [P2],
      workers: [],
    })

    const client = await Agent.createClient()
    try {
      await client.send({ type: 'seed', forkId: null, value: 1 })
      await client.send({ type: 'seed', forkId: 'b', value: 10 })
      await client.send({ type: 'hit', forkId: null, value: 3 })

      const rootFork = await client.runEffect(Effect.flatMap(P2.Tag, (p) => p.getFork(null)))
      const bFork = await client.runEffect(Effect.flatMap(P2.Tag, (p) => p.getFork('b')))
      expect(rootFork).toEqual({ value: 4 })
      expect(bFork).toEqual({ value: 13 })
    } finally {
      await client.dispose()
    }
  })

  test('per-fork handler runs before global handler', async () => {
    let seenInGlobal = -1

    const P3 = defineForked<TestEvent & ForkableEvent, { value: number }>()({
      name: 'P3',
      initialFork: { value: 0 },
      eventHandlers: {
        hit: ({ event, fork }) => ({ value: fork.value + event.value }),
      },
      globalEventHandlers: {
        hit: ({ event, state }) => {
          seenInGlobal = state.forks.get(event.forkId)?.value ?? -1
          return state
        },
      },
    })

    const Agent = defineAgent<TestEvent & ForkableEvent>()({
      name: 'PerForkBeforeGlobalAgent',
      projections: [P3],
      workers: [],
    })

    const client = await Agent.createClient()
    try {
      await client.send({ type: 'hit', forkId: null, value: 5 })
      expect(seenInGlobal).toBe(5)
    } finally {
      await client.dispose()
    }
  })

  test('global handler read returns full state for forked dependencies', async () => {
    let observed: ForkedState<{ value: number }> | null = null

    const ForkDep = defineForked<TestEvent & ForkableEvent, { value: number }>()({
      name: 'ForkDep',
      initialFork: { value: 0 },
      eventHandlers: {
        seed: ({ event }) => ({ value: event.value }),
      },
    })

    const Reader = defineForked<TestEvent & ForkableEvent, { value: number }>()({
      name: 'Reader',
      initialFork: { value: 0 },
      reads: [ForkDep],
      globalEventHandlers: {
        hit: ({ read, state }) => {
          observed = read(ForkDep)
          return state
        },
      },
    })

    const Agent = defineAgent<TestEvent & ForkableEvent>()({
      name: 'GlobalReadFullForkDepAgent',
      projections: [ForkDep, Reader],
      workers: [],
    })

    const client = await Agent.createClient()
    try {
      await client.send({ type: 'seed', forkId: null, value: 7 })
      await client.send({ type: 'seed', forkId: 'a', value: 9 })
      await client.send({ type: 'hit', forkId: null, value: 0 })

      expect(observed?.forks.get(null)).toEqual({ value: 7 })
      expect(observed?.forks.get('a')).toEqual({ value: 9 })
    } finally {
      await client.dispose()
    }
  })

  test('cross-fork read(projection, forkId)', async () => {
    const ForkDep = defineForked<TestEvent & ForkableEvent, { value: number }>()({
      name: 'ForkDepXRead',
      initialFork: { value: 0 },
      eventHandlers: {
        seed: ({ event }) => ({ value: event.value }),
      },
    })

    const Reader = defineForked<TestEvent & ForkableEvent, { value: number }>()({
      name: 'ReaderXRead',
      initialFork: { value: 0 },
      reads: [ForkDep],
      eventHandlers: {
        xread: ({ event, read }) => {
          const other = read(ForkDep, event.otherForkId)
          return { value: other.value }
        },
      },
    })

    const Agent = defineAgent<TestEvent & ForkableEvent>()({
      name: 'CrossForkReadAgent',
      projections: [ForkDep, Reader],
      workers: [],
    })

    const client = await Agent.createClient()
    try {
      await client.send({ type: 'seed', forkId: 'a', value: 42 })
      await client.send({ type: 'xread', forkId: 'b', otherForkId: 'a' })
      const bFork = await client.runEffect(Effect.flatMap(Reader.Tag, (p) => p.getFork('b')))
      expect(bFork).toEqual({ value: 42 })
    } finally {
      await client.dispose()
    }
  })

  test('signals emitted in global handlers are flushed correctly', async () => {
    const seen: string[] = []

    const Signaller = defineForked<TestEvent & ForkableEvent, { value: number }>()({
      name: 'Signaller',
      initialFork: { value: 0 },
      signals: {
        pong: { name: 'pong', shape: {} as { msg: string } },
      },
      globalEventHandlers: {
        hit: ({ emit, state }) => {
          emit.pong({ msg: 'from-global' })
          return state
        },
      },
    })

    const Listener = define<TestEvent & ForkableEvent, { count: number }>()({
      name: 'Listener',
      initial: { count: 0 },
      signalHandlers: (on) => [
        on(Signaller.signals.pong, ({ value, state }) => {
          seen.push(value.msg)
          return { count: state.count + 1 }
        }),
      ],
    })

    const Agent = defineAgent<TestEvent & ForkableEvent>()({
      name: 'GlobalSignalFlushAgent',
      projections: [Signaller, Listener],
      workers: [],
    })

    const client = await Agent.createClient()
    try {
      await client.send({ type: 'hit', forkId: null, value: 0 })
      expect(seen).toEqual(['from-global'])
      const listenerState = await client.runEffect(Effect.flatMap(Listener.Tag, (p) => p.get))
      expect(listenerState).toEqual({ count: 1 })
    } finally {
      await client.dispose()
    }
  })

  test('broadcastEventHandlers is removed (not in type)', () => {
    const ProjectionFactory = defineForked<TestEvent & ForkableEvent, { value: number }>()
    type Config = Parameters<typeof ProjectionFactory>[0]
    expect('broadcastEventHandlers' in ({} as Config)).toBe(false)
  })
})
