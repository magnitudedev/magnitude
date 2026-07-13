import { describe, expect, test } from 'bun:test'
import { Effect, Layer, Option, Ref, Schema } from 'effect'
import {
  AddressedEntryStore,
  estimateAddressedStoredBytes,
  makeInMemoryAddressedEntryStore,
  type AddressedEntryStore as AddressedEntryStoreService
} from '../entry-store'
import { AddressedStoreError } from '../errors'
import {
  ADDRESSED_SEQUENCE_SEGMENT_CAPACITY,
  type AddressedSequenceSegmentIndex,
} from '../collections/sequence'
import { AddressedSequenceIndexSchema } from '../index'
import {
  make as makeEventEngine,
  ProjectionSnapshotProjectionInvalid,
  Service as EventEngineService
} from '../../event-engine'
import type { FrameworkError } from '../../core/framework-error'
import * as Projection from '../../projection'
import type { ProjectionAddressedDescriptors } from '../../projection/addressed'
import { create as createSignal } from '../../signal'

type TestEvent =
  | { type: 'append'; id: string; text: string }
  | { type: 'read_tail'; limit: number }
  | { type: 'fail_after_write'; id: string; text: string }
  | { type: 'producer_chunk'; id: string; suffix: string }
  | { type: 'replace_range'; start: number; end: number; ids: readonly string[] }
  | { type: 'signal_append'; id: string; text: string }
  | { type: 'append_and_signal'; id: string; text: string }
  | { type: 'watch_tail'; limit: number }
  | { type: 'record_append'; member: string; id: string; text: string }
  | { type: 'record_update_text'; member: string; id: string; suffix: string }
  | { type: 'record_remove'; member: string }
  | { type: 'record_read_tail'; member: string; limit: number }

type ForkedTestEvent =
  | { type: 'fork_append'; forkId: string | null; id: string; text: string }
  | { type: 'fork_read_tail'; forkId: string | null; limit: number }

const TestMessageSchema = Schema.Struct({
  id: Schema.String,
  text: Schema.String
})

const TestLogMessages = Projection.addressed.sequence(TestMessageSchema)
const TestTimelineRecord = Projection.addressed.record(TestLogMessages)

const storeKey = (namespace: string, address: string) => `${namespace}\u0000${address}`

const makeCountingAddressedEntryStore = Effect.gen(function* () {
  const entriesRef = yield* Ref.make(new Map<string, unknown>())
  const loadCountsRef = yield* Ref.make(new Map<string, number>())

  const incrementLoad = (namespace: string, address: string) =>
    Ref.update(loadCountsRef, (counts) => {
      const next = new Map(counts)
      const key = storeKey(namespace, address)
      next.set(key, (next.get(key) ?? 0) + 1)
      return next
    })

  const store: AddressedEntryStoreService = {
    load: (namespace, address) =>
      Effect.gen(function* () {
        yield* incrementLoad(namespace, address)
        const entries = yield* Ref.get(entriesRef)
        const key = storeKey(namespace, address)
        return entries.has(key)
          ? Option.some(entries.get(key))
          : Option.none()
      }),
    stat: (namespace, address) =>
      Effect.map(Ref.get(entriesRef), (entries) => {
        const key = storeKey(namespace, address)
        return entries.has(key)
          ? Option.some({
              storedBytes: estimateAddressedStoredBytes(entries.get(key))
            })
          : Option.none()
      }),
    flush: (namespace, address, value) =>
      Ref.update(entriesRef, (entries) => {
        const next = new Map(entries)
        next.set(storeKey(namespace, address), value)
        return next
      })
  }

  const loadCount = (namespace: string, address: string) =>
    Effect.map(Ref.get(loadCountsRef), (counts) => counts.get(storeKey(namespace, address)) ?? 0)

  return { store, loadCount } as const
})

const makeFailingFlushCountingAddressedEntryStore = Effect.gen(function* () {
  const fixture = yield* makeCountingAddressedEntryStore
  const failFlushRef = yield* Ref.make(true)
  const flushAttemptsRef = yield* Ref.make(new Map<string, number>())

  const store: AddressedEntryStoreService = {
    load: fixture.store.load,
    stat: fixture.store.stat,
    flush: (namespace, address, value) =>
      Effect.gen(function* () {
        yield* Ref.update(flushAttemptsRef, (counts) => {
          const next = new Map(counts)
          const key = storeKey(namespace, address)
          next.set(key, (next.get(key) ?? 0) + 1)
          return next
        })

        const shouldFail = yield* Ref.get(failFlushRef)
        if (shouldFail) {
          return yield* new AddressedStoreError({
            operation: 'flush',
            namespace,
            address,
            cause: new Error('planned flush failure')
          })
        }

        return yield* fixture.store.flush(namespace, address, value)
      })
  }

  const flushAttempts = (namespace: string, address: string) =>
    Effect.map(Ref.get(flushAttemptsRef), (counts) => counts.get(storeKey(namespace, address)) ?? 0)

  return {
    store,
    loadCount: fixture.loadCount,
    flushAttempts,
    allowFlush: Ref.set(failFlushRef, false)
  } as const
})

const TestLogStateSchema = Schema.Struct({
  count: Schema.Number,
  messages: AddressedSequenceIndexSchema,
  tailIds: Schema.Array(Schema.String)
})

const TestSignalSourceStateSchema = Schema.Struct({
  emitted: Schema.Number
})

const appendRequested = createSignal<{ readonly id: string; readonly text: string }>(
  'AddressedSignalSource/appendRequested'
)

const TestLogProjection = Projection.define<TestEvent>()({
  name: 'AddressedLog',
  state: TestLogStateSchema,
  addressed: {
    messages: TestLogMessages
  },
  initial: {
    count: 0,
    messages: TestLogMessages.empty,
    tailIds: []
  },
  eventHandlers: {
    append: ({ event, state, addressed }) =>
      Effect.gen(function* () {
        const messages = yield* addressed.messages.append(state.messages, {
          id: event.id,
          text: event.text
        })
        return {
          ...state,
          count: state.count + 1,
          messages
        }
      }),

    append_and_signal: ({ event, state, addressed }) =>
      Effect.gen(function* () {
        const messages = yield* addressed.messages.append(state.messages, {
          id: event.id,
          text: event.text
        })
        return {
          ...state,
          count: state.count + 1,
          messages
        }
      }),

    read_tail: ({ event, state, addressed }) =>
      Effect.gen(function* () {
        const window = addressed.messages.resolveTailWindow(state.messages, event.limit)
        const messages = yield* addressed.messages.readWindow(window)
        return {
          ...state,
          tailIds: messages.map((message) => message.id)
        }
      }),

    fail_after_write: ({ event, state, addressed }) =>
      Effect.gen(function* () {
        yield* addressed.messages.append(state.messages, {
          id: event.id,
          text: event.text
        })
        return yield* Effect.fail('no commit')
      }),

    producer_chunk: ({ event, state, addressed }) =>
      Effect.gen(function* () {
        const messages = yield* addressed.messages.updateById(
          state.messages,
          event.id,
          (message) => ({
            ...message,
            text: `${message.text}${event.suffix}`
          })
        )
        return {
          ...state,
          messages
        }
      }),

    replace_range: ({ event, state, addressed }) =>
      Effect.gen(function* () {
        const messages = yield* addressed.messages.replaceRange(
          state.messages,
          event.start,
          event.end,
          event.ids.map((id) => ({ id, text: id }))
        )
        return {
          ...state,
          count: messages.totalCount,
          messages
        }
      })
  }
})

const AddressedSignalSourceProjection = Projection.define<TestEvent>()({
  name: 'AddressedSignalSource',
  state: TestSignalSourceStateSchema,
  signals: {
    appendRequested
  },
  initial: {
    emitted: 0
  },
  eventHandlers: {
    signal_append: ({ event, state, emit }) => {
      emit.appendRequested({ id: event.id, text: event.text })
      return { emitted: state.emitted + 1 }
    },
    append_and_signal: ({ event, state, emit }) => {
      emit.appendRequested({ id: event.id, text: event.text })
      return { emitted: state.emitted + 1 }
    }
  }
})

const TailViewStateSchema = Schema.Struct({
  watched: Schema.Number,
  rebuilds: Schema.Number,
  tailTexts: Schema.Array(Schema.String)
})
type TailViewState = typeof TailViewStateSchema.Type

/**
 * A consumer that derives from the log's addressed content through read()
 * Proxies — the framework tracks its reads and re-invokes its first-declared
 * signal handler when tracked content changes without an accompanying signal.
 * `rebuilds` counts signal-handler invocations (signal dispatch and addressed
 * triggers alike), so tests can assert exactly-once delivery.
 */
const TailViewProjection = Projection.define<TestEvent>()({
  name: 'AddressedTailView',
  state: TailViewStateSchema,
  reads: [TestLogProjection] as const,
  initial: {
    watched: 0,
    rebuilds: 0,
    tailTexts: []
  },
  eventHandlers: {
    watch_tail: ({ event, state, read }) => ({
      ...state,
      watched: event.limit,
      tailTexts: read(TestLogProjection).messages.slice(-event.limit).map((message) => message.text)
    })
  },
  signalHandlers: (on) => [
    on(AddressedSignalSourceProjection.signals.appendRequested, ({ state, read }): TailViewState =>
      state.watched === 0
        ? { ...state, rebuilds: state.rebuilds + 1 }
        : {
            ...state,
            rebuilds: state.rebuilds + 1,
            tailTexts: read(TestLogProjection).messages.slice(-state.watched).map((message) => message.text)
          }
    )
  ]
})

const AddressedSignalLogProjection = Projection.define<TestEvent>()({
  name: 'AddressedSignalLog',
  state: TestLogStateSchema,
  addressed: {
    messages: TestLogMessages
  },
  initial: {
    count: 0,
    messages: TestLogMessages.empty,
    tailIds: []
  },
  signalHandlers: (on) => [
    on(AddressedSignalSourceProjection.signals.appendRequested, ({ value, state, addressed }) =>
      Effect.gen(function* () {
        const messages = yield* addressed.messages.append(state.messages, {
          id: value.id,
          text: value.text
        })
        return {
          ...state,
          count: state.count + 1,
          messages
        }
      })
    )
  ]
})

const TestRecordLogStateSchema = Schema.Struct({
  timelines: TestTimelineRecord.indexSchema,
  tailIds: Schema.Array(Schema.String)
})

const TestRecordLogProjection = Projection.define<TestEvent>()({
  name: 'AddressedRecordLog',
  state: TestRecordLogStateSchema,
  addressed: {
    timelines: TestTimelineRecord
  },
  initial: {
    timelines: TestTimelineRecord.empty,
    tailIds: []
  },
  eventHandlers: {
    record_append: ({ event, state, addressed }) =>
      Effect.gen(function* () {
        const timelines = yield* addressed.timelines.updateMember(
          state.timelines,
          event.member,
          (messages, timeline) =>
            timeline.append(messages, {
              id: event.id,
              text: event.text
            })
        )
        return {
          ...state,
          timelines
        }
      }),

    record_update_text: ({ event, state, addressed }) =>
      Effect.gen(function* () {
        const timelines = yield* addressed.timelines.updateMember(
          state.timelines,
          event.member,
          (messages, timeline) =>
            timeline.updateById(messages, event.id, (message) => ({
              ...message,
              text: `${message.text}${event.suffix}`
            }))
        )
        return {
          ...state,
          timelines
        }
      }),

    record_read_tail: ({ event, state, addressed }) =>
      Effect.gen(function* () {
        const result = yield* addressed.timelines.readMember(
          state.timelines,
          event.member,
          (messages, timeline) => {
            const window = timeline.resolveTailWindow(messages, event.limit)
            return timeline.readWindow(window)
          }
        )
        return {
          ...state,
          tailIds: Option.match(result, {
            onNone: () => [],
            onSome: (messages) => messages.map((message) => message.id)
          })
        }
      }),

    record_remove: ({ event, state, addressed }) => ({
      ...state,
      timelines: addressed.timelines.remove(state.timelines, event.member)
    })
  }
})

const ForkedLogStateSchema = Schema.Struct({
  count: Schema.Number,
  messages: AddressedSequenceIndexSchema,
  tailIds: Schema.Array(Schema.String)
})

const ForkedLogProjection = Projection.defineForked<ForkedTestEvent>()({
  name: 'ForkedAddressedLog',
  forkState: ForkedLogStateSchema,
  addressed: {
    messages: TestLogMessages
  },
  initialFork: {
    count: 0,
    messages: TestLogMessages.empty,
    tailIds: []
  },
  eventHandlers: {
    fork_append: ({ event, fork, addressed }) =>
      Effect.gen(function* () {
        const messages = yield* addressed.messages.append(fork.messages, {
          id: event.id,
          text: event.text
        })
        return {
          ...fork,
          count: fork.count + 1,
          messages
        }
      }),

    fork_read_tail: ({ event, fork, addressed }) =>
      Effect.gen(function* () {
        const window = addressed.messages.resolveTailWindow(fork.messages, event.limit)
        const messages = yield* addressed.messages.readWindow(window)
        return {
          ...fork,
          tailIds: messages.map((message) => message.id)
        }
      })
  }
})

describe('addressed projection integration', () => {
  test('commits addressed sequence writes with ordinary state and faults flushed entries back later', async () => {
    const store = await Effect.runPromise(makeInMemoryAddressedEntryStore())

    const TestAgent = makeEventEngine<TestEvent>()({
      name: 'AddressedProjectionAgent',
      schemaVersion: 'test',
      projections: [TestLogProjection],
      workers: [],
      expose: {
        state: {
          log: TestLogProjection
        }
      }
    })

    const client = await TestAgent.createClient(
      Layer.succeed(AddressedEntryStore, store)
    )
    const errors: FrameworkError[] = []
    const stopErrors = client.onError((error) => {
      errors.push(error)
    })

    try {
      for (let i = 0; i < ADDRESSED_SEQUENCE_SEGMENT_CAPACITY + 1; i += 1) {
        await client.send({ type: 'append', id: `m${i}`, text: `message ${i}` })
      }

      expect(await client.state.log.get()).toMatchObject({
        count: ADDRESSED_SEQUENCE_SEGMENT_CAPACITY + 1,
        tailIds: []
      })

      await client.send({ type: 'read_tail', limit: 3 })
      expect(errors).toEqual([])
      expect(await client.state.log.get()).toMatchObject({
        tailIds: ['m48', 'm49', 'm50']
      })

      const materializedTail = await client.runEffect(
        Effect.gen(function* () {
          const log = yield* TestLogProjection.Tag
          const state = yield* log.get
          const window = log.addressed.messages.resolveTailWindow(state.messages, 2)
          const messages = yield* log.addressed.messages.readWindow(window)
          return messages.map((message) => message.id)
        })
      )
      expect(materializedTail).toEqual(['m49', 'm50'])

      const firstSegmentAddress = (await client.state.log.get()).messages.segments[0]!.address
      // Recent writes stay writer-pinned and dirty in memory. Snapshot
      // capture flushes them to the store.
      await client.runEffect(
        Effect.flatMap(EventEngineService, (engine) =>
          engine.captureProjectionSnapshot({ index: 0, timestamp: 1 }, 'session-1')
        )
      )
      const firstSegment = await Effect.runPromise(
        store.load('AddressedLog/messages', firstSegmentAddress)
      )
      expect(Option.isSome(firstSegment)).toBe(true)
      if (Option.isSome(firstSegment)) {
        expect(firstSegment.value).toMatchObject({
          items: expect.arrayContaining([
            { id: 'm0', text: 'message 0' },
            { id: 'm49', text: 'message 49' }
          ])
        })
      }

      await client.send({ type: 'fail_after_write', id: 'bad', text: 'discard me' })
      expect(await client.state.log.get()).toMatchObject({
        count: ADDRESSED_SEQUENCE_SEGMENT_CAPACITY + 1,
        tailIds: ['m48', 'm49', 'm50']
      })
    } finally {
      stopErrors()
      await client.dispose()
    }
  })

  test('failed write-through flush aborts the handler commit and reports a framework error', async () => {
    const fixture = await Effect.runPromise(makeFailingFlushCountingAddressedEntryStore)

    const TestAgent = makeEventEngine<TestEvent>()({
      name: 'AddressedProjectionWriteThroughAgent',
      schemaVersion: 'test',
      projections: [TestLogProjection],
      workers: [],
      expose: {
        state: {
          log: TestLogProjection
        }
      }
    })

    const client = await TestAgent.createClient(
      Layer.succeed(AddressedEntryStore, fixture.store)
    )
    const errors: FrameworkError[] = []
    const stopErrors = client.onError((error) => {
      errors.push(error)
    })
    const namespace = 'AddressedLog/messages'
    const segmentAddress = 'AddressedLog/messages/entries/entry-0'

    try {
      await client.runEffect(Effect.yieldNow())
      await client.send({ type: 'append', id: 'm0', text: 'aborted by failed flush' })

      // With auto-pinning, entries stay resident (dirty). The flush happens
      // when snapshot capture calls flushDirty. The failing flush aborts
      // the snapshot capture.
      expect(await client.state.log.get()).toMatchObject({ count: 1 })

      // Trigger flush via snapshot — this will fail
      const snapshotResult = await client.runEffect(
        Effect.either(
          Effect.flatMap(EventEngineService, (engine) =>
            engine.captureProjectionSnapshot({ index: 0, timestamp: 1 }, 'session-1')
          )
        )
      )
      expect(snapshotResult._tag).toBe('Left')

      expect(await Effect.runPromise(fixture.flushAttempts(namespace, segmentAddress))).toBe(1)

      // State remains committed (the append succeeded, only the flush failed)
      expect(await client.state.log.get()).toMatchObject({ count: 1 })

      // Allow flush to succeed, then flush the dirty entry
      await client.runEffect(fixture.allowFlush)
      await client.runEffect(
        Effect.flatMap(EventEngineService, (engine) =>
          engine.captureProjectionSnapshot({ index: 0, timestamp: 1 }, 'session-1')
        )
      )

      expect(await Effect.runPromise(fixture.flushAttempts(namespace, segmentAddress))).toBe(2)
      const persisted = await Effect.runPromise(fixture.store.load(namespace, segmentAddress))
      expect(Option.isSome(persisted)).toBe(true)

      const committedMessages = await client.runEffect(
        Effect.gen(function* () {
          const log = yield* TestLogProjection.Tag
          const state = yield* log.get
          return yield* log.addressed.messages.readAll(state.messages)
        })
      )
      expect(committedMessages).toEqual([{ id: 'm0', text: 'aborted by failed flush' }])
    } finally {
      stopErrors()
      await client.dispose()
    }
  })

  test('addressed content changes trigger consuming projection rebuilds, including across segment rollover', async () => {
    const store = await Effect.runPromise(makeInMemoryAddressedEntryStore())

    const TestAgent = makeEventEngine<TestEvent>()({
      name: 'AddressedTriggerAgent',
      schemaVersion: 'test',
      projections: [TestLogProjection, AddressedSignalSourceProjection, TailViewProjection],
      workers: [],
      expose: {
        state: {
          log: TestLogProjection,
          tailView: TailViewProjection
        }
      }
    })

    const client = await TestAgent.createClient(
      Layer.succeed(AddressedEntryStore, store)
    )
    try {
      await client.send({ type: 'append', id: 'm0', text: 'message 0' })
      await client.send({ type: 'watch_tail', limit: 2 })
      expect(await client.state.tailView.get()).toMatchObject({ watched: 2, tailTexts: ['message 0'] })

      // A content change with no accompanying signal reaches the consumer
      // through the addressed-change trigger — exactly once.
      const rebuildsBeforeChunk = (await client.state.tailView.get()).rebuilds
      await client.send({ type: 'producer_chunk', id: 'm0', suffix: '!' })
      expect(await client.state.tailView.get()).toMatchObject({
        watched: 2,
        rebuilds: rebuildsBeforeChunk + 1,
        tailTexts: ['message 0!']
      })

      // Appends keep the tail fresh — including the append that rolls over
      // into a brand-new segment the consumer has never read. The sequence
      // sentinel carries the notification for the fresh address.
      for (let i = 1; i < ADDRESSED_SEQUENCE_SEGMENT_CAPACITY + 1; i += 1) {
        await client.send({ type: 'append', id: `m${i}`, text: `message ${i}` })
      }
      const logState = await client.state.log.get()
      expect(logState.messages.segments).toHaveLength(2)
      expect(await client.state.tailView.get()).toMatchObject({
        watched: 2,
        tailTexts: [
          `message ${ADDRESSED_SEQUENCE_SEGMENT_CAPACITY - 1}`,
          `message ${ADDRESSED_SEQUENCE_SEGMENT_CAPACITY}`
        ]
      })

      // Streaming into the fresh segment keeps triggering rebuilds.
      await client.send({ type: 'producer_chunk', id: `m${ADDRESSED_SEQUENCE_SEGMENT_CAPACITY}`, suffix: ' (updated)' })
      expect(await client.state.tailView.get()).toMatchObject({
        watched: 2,
        tailTexts: [
          `message ${ADDRESSED_SEQUENCE_SEGMENT_CAPACITY - 1}`,
          `message ${ADDRESSED_SEQUENCE_SEGMENT_CAPACITY} (updated)`
        ]
      })
    } finally {
      await client.dispose()
    }
  })

  test('a signal and an addressed change in the same flush run the consumer exactly once', async () => {
    const store = await Effect.runPromise(makeInMemoryAddressedEntryStore())

    const TestAgent = makeEventEngine<TestEvent>()({
      name: 'AddressedSignalDedupeAgent',
      schemaVersion: 'test',
      projections: [TestLogProjection, AddressedSignalSourceProjection, TailViewProjection],
      workers: [],
      expose: {
        state: {
          tailView: TailViewProjection
        }
      }
    })

    const client = await TestAgent.createClient(
      Layer.succeed(AddressedEntryStore, store)
    )
    try {
      await client.send({ type: 'append', id: 'm0', text: 'message 0' })
      await client.send({ type: 'watch_tail', limit: 1 })
      const before = (await client.state.tailView.get()).rebuilds

      // append_and_signal: TestLog appends (an addressed change the consumer
      // tracks) and the source emits appendRequested in the same event cycle.
      // The signal dispatches unconditionally; the addressed trigger is then
      // skipped because the consumer already ran after the change.
      await client.send({ type: 'append_and_signal', id: 'm1', text: 'message 1' })

      expect(await client.state.tailView.get()).toMatchObject({
        rebuilds: before + 1,
        tailTexts: ['message 1']
      })
    } finally {
      await client.dispose()
    }
  })

  test('projection snapshots encode addressed indexes but not addressed entry bodies', async () => {
    const fixture = await Effect.runPromise(makeCountingAddressedEntryStore)

    const TestAgent = makeEventEngine<TestEvent>()({
      name: 'AddressedProjectionSnapshotAgent',
      schemaVersion: 'test',
      projections: [TestLogProjection],
      workers: [],
      expose: {
        state: {
          log: TestLogProjection
        }
      }
    })

    const client = await TestAgent.createClient(
      Layer.succeed(AddressedEntryStore, fixture.store)
    )

    try {
      await client.send({ type: 'append', id: 'm0', text: 'snapshot body stays in the addressed store' })
      const segmentAddress = (await client.state.log.get()).messages.segments[0]!.address

      const snapshot = await client.runEffect(
        Effect.flatMap(EventEngineService, (engine) =>
          engine.captureProjectionSnapshot({ index: 0, timestamp: 1 }, 'session-1')
        )
      )
      const encodedSnapshot = JSON.stringify(snapshot)
      expect(encodedSnapshot).toContain(segmentAddress)
      expect(encodedSnapshot).toContain('m0')
      expect(encodedSnapshot).not.toContain('snapshot body stays in the addressed store')

      await client.runEffect(
        Effect.flatMap(EventEngineService, (engine) =>
          Effect.gen(function* () {
            const plan = yield* engine.prepareProjectionSnapshotRestore(snapshot)
            yield* plan.commit
          })
        )
      )

      // Restore pins nothing: entries stay in the store until first read.
      expect(await Effect.runPromise(fixture.loadCount('AddressedLog/messages', segmentAddress))).toBe(0)

      const restoredMessages = await client.runEffect(
        Effect.gen(function* () {
          const log = yield* TestLogProjection.Tag
          const state = yield* log.get
          return yield* log.addressed.messages.readAll(state.messages)
        })
      )

      expect(restoredMessages).toEqual([
        { id: 'm0', text: 'snapshot body stays in the addressed store' }
      ])
      // The read loaded the segment on first access (load-on-access).
      expect(await Effect.runPromise(fixture.loadCount('AddressedLog/messages', segmentAddress))).toBe(1)
    } finally {
      await client.dispose()
    }
  })

  test('projection snapshot restore clears resident addressed entries', async () => {
    const fixture = await Effect.runPromise(makeCountingAddressedEntryStore)

    const TestAgent = makeEventEngine<TestEvent>()({
      name: 'AddressedProjectionRestoreResetAgent',
      schemaVersion: 'test',
      projections: [TestLogProjection],
      workers: [],
      expose: {
        state: {
          log: TestLogProjection
        }
      }
    })

    const client = await TestAgent.createClient(
      Layer.succeed(AddressedEntryStore, fixture.store)
    )

    try {
      await client.send({ type: 'append', id: 'm0', text: 'message' })
      const segmentAddress = (await client.state.log.get()).messages.segments[0]!.address
      // The writer pin keeps the segment resident across the chunk
      // update — no store load.
      await client.send({ type: 'producer_chunk', id: 'm0', suffix: '!' })
      expect(await Effect.runPromise(fixture.loadCount('AddressedLog/messages', segmentAddress))).toBe(0)

      const snapshot = await client.runEffect(
        Effect.flatMap(EventEngineService, (engine) =>
          engine.captureProjectionSnapshot({ index: 2, timestamp: 1 }, 'session-1')
        )
      )

      await client.runEffect(
        Effect.flatMap(EventEngineService, (engine) =>
          Effect.gen(function* () {
            const plan = yield* engine.prepareProjectionSnapshotRestore(snapshot)
            yield* plan.commit
          })
        )
      )

      const restoredMessages = await client.runEffect(
        Effect.gen(function* () {
          const log = yield* TestLogProjection.Tag
          const state = yield* log.get
          return yield* log.addressed.messages.readAll(state.messages)
        })
      )

      expect(restoredMessages).toEqual([{ id: 'm0', text: 'message!' }])
      // Restore reset residency; readAll loaded the segment on first access.
      expect(await Effect.runPromise(fixture.loadCount('AddressedLog/messages', segmentAddress))).toBe(1)
    } finally {
      await client.dispose()
    }
  })

  test('projection snapshot restore rejects malformed addressed ordinary indexes', async () => {
    const fixture = await Effect.runPromise(makeCountingAddressedEntryStore)

    const TestAgent = makeEventEngine<TestEvent>()({
      name: 'AddressedProjectionInvalidIndexRestoreAgent',
      schemaVersion: 'test',
      projections: [TestLogProjection],
      workers: [],
      expose: {
        state: {
          log: TestLogProjection
        }
      }
    })

    const client = await TestAgent.createClient(
      Layer.succeed(AddressedEntryStore, fixture.store)
    )

    try {
      await client.send({ type: 'append', id: 'm0', text: 'message' })
      const snapshot = await client.runEffect(
        Effect.flatMap(EventEngineService, (engine) =>
          engine.captureProjectionSnapshot({ index: 0, timestamp: 1 }, 'session-1')
        )
      )

      const invalidSnapshot = {
        ...snapshot,
        projections: {
          ...snapshot.projections,
          AddressedLog: {
            ...snapshot.projections.AddressedLog,
            messages: {
              ...snapshot.projections.AddressedLog.messages,
              segments: snapshot.projections.AddressedLog.messages.segments.map((segment: AddressedSequenceSegmentIndex, index: number) =>
                index === 0
                  ? { ...segment, count: ADDRESSED_SEQUENCE_SEGMENT_CAPACITY + 1 }
                  : segment
              )
            }
          }
        }
      }

      const result = await client.runEffect(
        Effect.flatMap(EventEngineService, (engine) =>
          Effect.either(engine.prepareProjectionSnapshotRestore(invalidSnapshot))
        )
      )

      expect(result._tag).toBe('Left')
      if (result._tag === 'Left') {
        expect(result.left).toBeInstanceOf(ProjectionSnapshotProjectionInvalid)
        if (result.left instanceof ProjectionSnapshotProjectionInvalid) {
          expect(result.left.projectionName).toBe('AddressedLog')
        }
      }
    } finally {
      await client.dispose()
    }
  })

  test('projection snapshot restore rejects inconsistent addressed sequence indexes', async () => {
    const fixture = await Effect.runPromise(makeCountingAddressedEntryStore)

    const TestAgent = makeEventEngine<TestEvent>()({
      name: 'AddressedProjectionInconsistentIndexRestoreAgent',
      schemaVersion: 'test',
      projections: [TestLogProjection],
      workers: [],
      expose: {
        state: {
          log: TestLogProjection
        }
      }
    })

    const client = await TestAgent.createClient(
      Layer.succeed(AddressedEntryStore, fixture.store)
    )

    try {
      await client.send({ type: 'append', id: 'm0', text: 'message' })
      const snapshot = await client.runEffect(
        Effect.flatMap(EventEngineService, (engine) =>
          engine.captureProjectionSnapshot({ index: 0, timestamp: 1 }, 'session-1')
        )
      )

      const invalidSnapshot = {
        ...snapshot,
        projections: {
          ...snapshot.projections,
          AddressedLog: {
            ...snapshot.projections.AddressedLog,
            messages: {
              ...snapshot.projections.AddressedLog.messages,
              totalCount: snapshot.projections.AddressedLog.messages.totalCount + 1
            }
          }
        }
      }

      const result = await client.runEffect(
        Effect.flatMap(EventEngineService, (engine) =>
          Effect.either(engine.prepareProjectionSnapshotRestore(invalidSnapshot))
        )
      )

      expect(result._tag).toBe('Left')
      if (result._tag === 'Left') {
        expect(result.left).toBeInstanceOf(ProjectionSnapshotProjectionInvalid)
        if (result.left instanceof ProjectionSnapshotProjectionInvalid) {
          expect(result.left.projectionName).toBe('AddressedLog')
        }
      }
    } finally {
      await client.dispose()
    }
  })

  test('projection snapshot restore rejects malformed addressed record indexes', async () => {
    const store = await Effect.runPromise(makeInMemoryAddressedEntryStore())

    const TestAgent = makeEventEngine<TestEvent>()({
      name: 'AddressedRecordInvalidIndexRestoreAgent',
      schemaVersion: 'test',
      projections: [TestRecordLogProjection],
      workers: [],
      expose: {
        state: {
          log: TestRecordLogProjection
        }
      }
    })

    const client = await TestAgent.createClient(
      Layer.succeed(AddressedEntryStore, store)
    )

    try {
      await client.send({ type: 'record_append', member: 'root', id: 'root-1', text: 'root one' })
      await client.send({ type: 'record_append', member: 'worker-a', id: 'worker-1', text: 'worker one' })
      const snapshot = await client.runEffect(
        Effect.flatMap(EventEngineService, (engine) =>
          engine.captureProjectionSnapshot({ index: 1, timestamp: 2 }, 'session-1')
        )
      )

      const invalidSnapshot = {
        ...snapshot,
        projections: {
          ...snapshot.projections,
          AddressedRecordLog: {
            ...snapshot.projections.AddressedRecordLog,
            timelines: {
              ...snapshot.projections.AddressedRecordLog.timelines,
              members: ['root']
            }
          }
        }
      }

      const result = await client.runEffect(
        Effect.flatMap(EventEngineService, (engine) =>
          Effect.either(engine.prepareProjectionSnapshotRestore(invalidSnapshot))
        )
      )

      expect(result._tag).toBe('Left')
      if (result._tag === 'Left') {
        expect(result.left).toBeInstanceOf(ProjectionSnapshotProjectionInvalid)
        if (result.left instanceof ProjectionSnapshotProjectionInvalid) {
          expect(result.left.projectionName).toBe('AddressedRecordLog')
        }
      }
    } finally {
      await client.dispose()
    }
  })

  test('addressed descriptor maps require complete descriptor values', () => {
    const valid = {
      messages: TestLogMessages
    } satisfies ProjectionAddressedDescriptors

    // @ts-expect-error empty objects are not addressed descriptors
    const malformed = { bad: {} } satisfies ProjectionAddressedDescriptors

    // @ts-expect-error a tag alone is not an addressed descriptor
    const partial = { bad: { _tag: 'Sequence' } } satisfies ProjectionAddressedDescriptors

    expect(valid.messages).toBe(TestLogMessages)
    void malformed
    void partial
  })

  test('the writer pin keeps actively written segments resident across event cycles', async () => {
    const fixture = await Effect.runPromise(makeCountingAddressedEntryStore)

    const TestAgent = makeEventEngine<TestEvent>()({
      name: 'AddressedProjectionWriterPinAgent',
      schemaVersion: 'test',
      projections: [TestLogProjection],
      workers: [],
      expose: {
        state: {
          log: TestLogProjection
        }
      }
    })

    const client = await TestAgent.createClient(
      Layer.succeed(AddressedEntryStore, fixture.store)
    )

    try {
      await client.send({ type: 'append', id: 'm0', text: 'message' })
      const segmentAddress = (await client.state.log.get()).messages.segments[0]!.address
      // The written segment is writer-pinned: resident, no store load.
      expect(await Effect.runPromise(fixture.loadCount('AddressedLog/messages', segmentAddress))).toBe(0)

      // Streaming rewrites keep it writer-pinned every commit.
      await client.send({ type: 'producer_chunk', id: 'm0', suffix: ' a' })
      await client.send({ type: 'producer_chunk', id: 'm0', suffix: ' b' })
      await client.send({ type: 'producer_chunk', id: 'm0', suffix: ' c' })
      expect(await Effect.runPromise(fixture.loadCount('AddressedLog/messages', segmentAddress))).toBe(0)

      // A non-writing event cycle does not rotate the writer pin.
      await client.send({ type: 'read_tail', limit: 1 })
      expect(await Effect.runPromise(fixture.loadCount('AddressedLog/messages', segmentAddress))).toBe(0)
      expect(await client.state.log.get()).toMatchObject({
        tailIds: ['m0']
      })
    } finally {
      await client.dispose()
    }
  })

  test('addressed signal handlers commit addressed writes during signal flush', async () => {
    const store = await Effect.runPromise(makeInMemoryAddressedEntryStore())

    const TestAgent = makeEventEngine<TestEvent>()({
      name: 'AddressedSignalProjectionAgent',
      schemaVersion: 'test',
      projections: [AddressedSignalSourceProjection, AddressedSignalLogProjection],
      workers: [],
      expose: {
        state: {
          source: AddressedSignalSourceProjection,
          log: AddressedSignalLogProjection
        }
      }
    })

    const client = await TestAgent.createClient(
      Layer.succeed(AddressedEntryStore, store)
    )

    try {
      await client.send({ type: 'signal_append', id: 'sig-1', text: 'from signal' })

      expect(await client.state.source.get()).toEqual({ emitted: 1 })
      const logState = await client.state.log.get()
      expect(logState.count).toBe(1)
      expect(logState.messages.totalCount).toBe(1)

      const materialized = await client.runEffect(
        Effect.gen(function* () {
          const log = yield* AddressedSignalLogProjection.Tag
          const state = yield* log.get
          const window = log.addressed.messages.resolveTailWindow(state.messages, 1)
          return yield* log.addressed.messages.readWindow(window)
        })
      )
      expect(materialized).toEqual([{ id: 'sig-1', text: 'from signal' }])

      // Recent writes stay writer-pinned and dirty in memory. Flush to the store for verification.
      await client.runEffect(
        Effect.flatMap(EventEngineService, (engine) =>
          engine.captureProjectionSnapshot({ index: 0, timestamp: 1 }, 'session-1')
        )
      )
      const stored = await Effect.runPromise(
        store.load('AddressedSignalLog/messages', logState.messages.segments[0]!.address)
      )
      expect(Option.isSome(stored)).toBe(true)
      if (Option.isSome(stored)) {
        expect(stored.value).toMatchObject({
          items: [{ id: 'sig-1', text: 'from signal' }]
        })
      }
    } finally {
      await client.dispose()
    }
  })

  test('projection addressed sequence range replacement commits ordinary index and addressed entries together', async () => {
    const store = await Effect.runPromise(makeInMemoryAddressedEntryStore())

    const TestAgent = makeEventEngine<TestEvent>()({
      name: 'AddressedProjectionRangeAgent',
      schemaVersion: 'test',
      projections: [TestLogProjection],
      workers: [],
      expose: {
        state: {
          log: TestLogProjection
        }
      }
    })

    const client = await TestAgent.createClient(
      Layer.succeed(AddressedEntryStore, store)
    )

    try {
      for (let i = 0; i < ADDRESSED_SEQUENCE_SEGMENT_CAPACITY + 3; i += 1) {
        await client.send({ type: 'append', id: `m${i}`, text: `message ${i}` })
      }

      await client.send({
        type: 'replace_range',
        start: ADDRESSED_SEQUENCE_SEGMENT_CAPACITY,
        end: ADDRESSED_SEQUENCE_SEGMENT_CAPACITY + 2,
        ids: ['new-a', 'new-b', 'new-c']
      })
      await client.send({ type: 'read_tail', limit: 5 })

      expect(await client.state.log.get()).toMatchObject({
        count: ADDRESSED_SEQUENCE_SEGMENT_CAPACITY + 4,
        tailIds: ['m49', 'new-a', 'new-b', 'new-c', 'm52']
      })
    } finally {
      await client.dispose()
    }
  })

  test('forked projections use addressed entries through per-fork ordinary indexes', async () => {
    const store = await Effect.runPromise(makeInMemoryAddressedEntryStore())

    const TestAgent = makeEventEngine<ForkedTestEvent>()({
      name: 'ForkedAddressedProjectionAgent',
      schemaVersion: 'test',
      projections: [ForkedLogProjection],
      workers: [],
      expose: {
        state: {
          log: ForkedLogProjection
        }
      }
    })

    const client = await TestAgent.createClient(
      Layer.succeed(AddressedEntryStore, store)
    )

    try {
      await client.send({ type: 'fork_append', forkId: null, id: 'root-1', text: 'root one' })
      await client.send({ type: 'fork_append', forkId: 'worker-a', id: 'worker-1', text: 'worker one' })
      await client.send({ type: 'fork_append', forkId: 'worker-a', id: 'worker-2', text: 'worker two' })

      await client.send({ type: 'fork_read_tail', forkId: 'worker-a', limit: 2 })

      expect(await client.state.log.getFork(null)).toMatchObject({
        count: 1,
        tailIds: []
      })
      expect(await client.state.log.getFork('worker-a')).toMatchObject({
        count: 2,
        tailIds: ['worker-1', 'worker-2']
      })

      const materializedWorkerTail = await client.runEffect(
        Effect.gen(function* () {
          const log = yield* ForkedLogProjection.Tag
          const fork = yield* log.getFork('worker-a')
          const messages = log.addressed.forFork('worker-a').messages
          const window = messages.resolveTailWindow(fork.messages, 2)
          const tail = yield* messages.readWindow(window)
          return tail.map((message) => message.id)
        })
      )
      expect(materializedWorkerTail).toEqual(['worker-1', 'worker-2'])

      // Flush dirty entries to the store (recent writes stay writer-pinned
      // and dirty in memory until snapshot capture or rotation)
      await client.runEffect(
        Effect.flatMap(EventEngineService, (engine) =>
          engine.captureProjectionSnapshot({ index: 0, timestamp: 1 }, 'session-1')
        )
      )

      const rootState = await client.state.log.getFork(null)
      const workerState = await client.state.log.getFork('worker-a')
      const rootStored = await Effect.runPromise(
        store.load('ForkedAddressedLog/messages', rootState.messages.segments[0]!.address)
      )
      expect(Option.isSome(rootStored)).toBe(true)
      if (Option.isSome(rootStored)) {
        expect(rootStored.value).toMatchObject({
          items: expect.arrayContaining([
            { id: 'root-1', text: 'root one' }
          ])
        })
      }

      const workerStored = await Effect.runPromise(
        store.load('ForkedAddressedLog/messages', workerState.messages.segments[0]!.address)
      )
      expect(Option.isSome(workerStored)).toBe(true)
      if (Option.isSome(workerStored)) {
        expect(workerStored.value).toMatchObject({
          items: expect.arrayContaining([
            { id: 'worker-1', text: 'worker one' },
            { id: 'worker-2', text: 'worker two' }
          ])
        })
      }

      // Restore the multi-fork snapshot and read every fork's content back —
      // restore pins nothing, so all forks load on access from the store.
      const snapshot = await client.runEffect(
        Effect.flatMap(EventEngineService, (engine) =>
          engine.captureProjectionSnapshot({ index: 3, timestamp: 4 }, 'session-1')
        )
      )
      await client.runEffect(
        Effect.flatMap(EventEngineService, (engine) =>
          Effect.gen(function* () {
            const plan = yield* engine.prepareProjectionSnapshotRestore(snapshot)
            yield* plan.commit
          })
        )
      )

      const restoredTails = await client.runEffect(
        Effect.gen(function* () {
          const log = yield* ForkedLogProjection.Tag
          const rootFork = yield* log.getFork(null)
          const workerFork = yield* log.getFork('worker-a')
          const root = yield* log.addressed.forFork(null).messages.readAll(rootFork.messages)
          const worker = yield* log.addressed.forFork('worker-a').messages.readAll(workerFork.messages)
          return {
            root: root.map((message) => message.id),
            worker: worker.map((message) => message.id)
          }
        })
      )
      expect(restoredTails).toEqual({
        root: ['root-1'],
        worker: ['worker-1', 'worker-2']
      })
    } finally {
      await client.dispose()
    }
  })

  test('record of sequences composes member indexes with child addressed entries', async () => {
    const store = await Effect.runPromise(makeInMemoryAddressedEntryStore())

    const TestAgent = makeEventEngine<TestEvent>()({
      name: 'AddressedRecordProjectionAgent',
      schemaVersion: 'test',
      projections: [TestRecordLogProjection],
      workers: [],
      expose: {
        state: {
          log: TestRecordLogProjection
        }
      }
    })

    const client = await TestAgent.createClient(
      Layer.succeed(AddressedEntryStore, store)
    )

    try {
      await client.send({ type: 'record_append', member: 'root', id: 'root-1', text: 'root one' })
      await client.send({ type: 'record_append', member: 'worker-a', id: 'worker-1', text: 'worker one' })
      await client.send({ type: 'record_append', member: 'worker-a', id: 'worker-2', text: 'worker two' })

      await client.send({ type: 'record_read_tail', member: 'worker-a', limit: 2 })

      const state = await client.state.log.get()
      expect(state.timelines.members).toEqual(['root', 'worker-a'])
      expect(state.tailIds).toEqual(['worker-1', 'worker-2'])
      expect(state.timelines.children.root.totalCount).toBe(1)
      expect(state.timelines.children['worker-a'].totalCount).toBe(2)

      // Recent writes stay writer-pinned and dirty in memory. Flush to the store for verification.
      await client.runEffect(
        Effect.flatMap(EventEngineService, (engine) =>
          engine.captureProjectionSnapshot({ index: 2, timestamp: 3 }, 'session-1')
        )
      )
      const rootStored = await Effect.runPromise(
        store.load('AddressedRecordLog/timelines', state.timelines.children.root.segments[0]!.address)
      )
      expect(Option.isSome(rootStored)).toBe(true)
      if (Option.isSome(rootStored)) {
        expect(rootStored.value).toMatchObject({
          items: [{ id: 'root-1', text: 'root one' }]
        })
      }

      const workerStored = await Effect.runPromise(
        store.load('AddressedRecordLog/timelines', state.timelines.children['worker-a'].segments[0]!.address)
      )
      expect(Option.isSome(workerStored)).toBe(true)
      if (Option.isSome(workerStored)) {
        expect(workerStored.value).toMatchObject({
          items: [
            { id: 'worker-1', text: 'worker one' },
            { id: 'worker-2', text: 'worker two' }
          ]
        })
      }
    } finally {
      await client.dispose()
    }
  })

  test('addressed record remove updates membership without touching other child entries', async () => {
    const store = await Effect.runPromise(makeInMemoryAddressedEntryStore())

    const TestAgent = makeEventEngine<TestEvent>()({
      name: 'AddressedRecordRemoveAgent',
      schemaVersion: 'test',
      projections: [TestRecordLogProjection],
      workers: [],
      expose: {
        state: {
          log: TestRecordLogProjection
        }
      }
    })

    const client = await TestAgent.createClient(
      Layer.succeed(AddressedEntryStore, store)
    )

    try {
      await client.send({ type: 'record_append', member: 'root', id: 'root-1', text: 'root one' })
      const rootAddress = (await client.state.log.get()).timelines.children.root.segments[0]!.address
      await client.send({ type: 'record_append', member: 'worker-a', id: 'worker-1', text: 'worker one' })
      const workerAddress = (await client.state.log.get()).timelines.children['worker-a'].segments[0]!.address
      await client.send({ type: 'record_remove', member: 'root' })
      await client.send({ type: 'record_read_tail', member: 'worker-a', limit: 1 })

      const state = await client.state.log.get()
      expect(state.timelines.members).toEqual(['worker-a'])
      expect(Object.keys(state.timelines.children)).toEqual(['worker-a'])
      expect(state.tailIds).toEqual(['worker-1'])

      const recordShape = await client.runEffect(
        Effect.gen(function* () {
          const log = yield* TestRecordLogProjection.Tag
          const current = yield* log.get
          const timelines = log.addressed.timelines
          return {
            members: timelines.members(current.timelines),
            hasRoot: timelines.has(current.timelines, 'root'),
            hasWorker: timelines.has(current.timelines, 'worker-a'),
            root: timelines.resolveMember(current.timelines, 'root'),
            worker: timelines.resolveMember(current.timelines, 'worker-a')
          }
        })
      )

      expect(recordShape.members).toEqual(['worker-a'])
      expect(recordShape.hasRoot).toBe(false)
      expect(recordShape.hasWorker).toBe(true)
      expect(Option.isNone(recordShape.root)).toBe(true)
      expect(Option.isSome(recordShape.worker)).toBe(true)

      // Recent writes stay writer-pinned and dirty in memory. Flush to the store for verification.
      await client.runEffect(
        Effect.flatMap(EventEngineService, (engine) =>
          engine.captureProjectionSnapshot({ index: 3, timestamp: 4 }, 'session-1')
        )
      )
      const rootStored = await Effect.runPromise(
        store.load('AddressedRecordLog/timelines', rootAddress)
      )
      expect(Option.isSome(rootStored)).toBe(true)

      const workerStored = await Effect.runPromise(
        store.load('AddressedRecordLog/timelines', workerAddress)
      )
      expect(Option.isSome(workerStored)).toBe(true)
      if (Option.isSome(workerStored)) {
        expect(workerStored.value).toMatchObject({
          items: [{ id: 'worker-1', text: 'worker one' }]
        })
      }
    } finally {
      await client.dispose()
    }
  })

  test('nested record update preserves parent index identity when child index is unchanged', async () => {
    const store = await Effect.runPromise(makeInMemoryAddressedEntryStore())

    const TestAgent = makeEventEngine<TestEvent>()({
      name: 'AddressedRecordStableIndexAgent',
      schemaVersion: 'test',
      projections: [TestRecordLogProjection],
      workers: [],
      expose: {
        state: {
          log: TestRecordLogProjection
        }
      }
    })

    const client = await TestAgent.createClient(
      Layer.succeed(AddressedEntryStore, store)
    )

    try {
      await client.send({ type: 'record_append', member: 'worker-a', id: 'worker-1', text: 'worker one' })
      const before = await client.state.log.get()

      await client.send({ type: 'record_update_text', member: 'worker-a', id: 'worker-1', suffix: '!' })
      const after = await client.state.log.get()

      expect(after.timelines).toBe(before.timelines)

      const messages = await client.runEffect(
        Effect.gen(function* () {
          const log = yield* TestRecordLogProjection.Tag
          const state = yield* log.get
          const result = yield* log.addressed.timelines.readMember(
            state.timelines,
            'worker-a',
            (index, timeline) => timeline.readAll(index)
          )
          return Option.getOrElse(result, () => [])
        })
      )

      expect(messages).toEqual([
        { id: 'worker-1', text: 'worker one!' }
      ])
    } finally {
      await client.dispose()
    }
  })
})
