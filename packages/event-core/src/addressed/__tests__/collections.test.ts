import { describe, expect, test } from 'bun:test'
import { Effect, Option, Ref, Schema } from 'effect'
import { makeAddressSpaceRuntime } from '../address-space'
import { makeSchemaCodec } from '../codec'
import {
  ADDRESSED_SEQUENCE_SEGMENT_CAPACITY,
  makeAddressedSequence
} from '../collections/sequence'
import {
  AddressedEntryStore,
  estimateAddressedStoredBytes,
  type AddressedEntryStore as AddressedEntryStoreService
} from '../entry-store'

const TestMessageSchema = Schema.Struct({
  id: Schema.String,
  text: Schema.String
})

const TestMessageSegmentSchema = Schema.Struct({
  items: Schema.Array(TestMessageSchema)
})

type TestMessage = Schema.Schema.Type<typeof TestMessageSchema>
type TestMessageSegment = Schema.Schema.Type<typeof TestMessageSegmentSchema>

const keyFor = (namespace: string, address: string) => `${namespace}\u0000${address}`

const makeCountingStore = (
  initial: Iterable<readonly [namespace: string, address: string, value: unknown]> = []
) =>
  Effect.gen(function* () {
    const entries = new Map<string, unknown>()
    for (const [namespace, address, value] of initial) {
      entries.set(keyFor(namespace, address), value)
    }

    const entriesRef = yield* Ref.make(entries)
    const loadCountsRef = yield* Ref.make(new Map<string, number>())
    const flushCountsRef = yield* Ref.make(new Map<string, number>())

    const increment = (ref: Ref.Ref<Map<string, number>>, namespace: string, address: string) =>
      Ref.update(ref, (counts) => {
        const next = new Map(counts)
        const key = keyFor(namespace, address)
        next.set(key, (next.get(key) ?? 0) + 1)
        return next
      })

    const countFor = (ref: Ref.Ref<Map<string, number>>, namespace: string, address: string) =>
      Effect.map(Ref.get(ref), (counts) => counts.get(keyFor(namespace, address)) ?? 0)

    const valueFor = (namespace: string, address: string) =>
      Effect.map(Ref.get(entriesRef), (entries) => entries.get(keyFor(namespace, address)))

    const store: AddressedEntryStoreService = {
      load: (namespace, address) =>
        Effect.gen(function* () {
          yield* increment(loadCountsRef, namespace, address)
          const entries = yield* Ref.get(entriesRef)
          const key = keyFor(namespace, address)
          return entries.has(key)
            ? Option.some(entries.get(key))
            : Option.none()
        }),

      stat: (namespace, address) =>
        Effect.map(Ref.get(entriesRef), (entries) => {
          const key = keyFor(namespace, address)
          return entries.has(key)
            ? Option.some({
                storedBytes: estimateAddressedStoredBytes(entries.get(key))
              })
            : Option.none()
        }),

      flush: (namespace, address, value) =>
        Effect.gen(function* () {
          yield* increment(flushCountsRef, namespace, address)
          yield* Ref.update(entriesRef, (current) => {
            const next = new Map(current)
            next.set(keyFor(namespace, address), value)
            return next
          })
        })
    }

    return {
      store,
      valueFor,
      loadCount: (namespace: string, address: string) => countFor(loadCountsRef, namespace, address),
      flushCount: (namespace: string, address: string) => countFor(flushCountsRef, namespace, address)
    }
  })

const makeSegmentRuntime = (store: AddressedEntryStoreService) =>
  makeAddressSpaceRuntime<TestMessageSegment>({
    namespace: 'test/sequence',
    codec: makeSchemaCodec(TestMessageSegmentSchema)
  }).pipe(Effect.provideService(AddressedEntryStore, store))

describe('addressed sequence collection', () => {
  test('append touches the tail segment until rollover creates the next segment', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/root/messages',
        runtime
      })

      let index = sequence.empty
      const mutation = runtime.makeMutation()
      const transaction = mutation.transaction
      for (let i = 0; i < ADDRESSED_SEQUENCE_SEGMENT_CAPACITY + 1; i += 1) {
        index = yield* sequence.append(transaction, index, {
          id: `m${i}`,
          text: `message ${i}`
        })
      }

      const changed = yield* mutation.commit

      expect(index.totalCount).toBe(ADDRESSED_SEQUENCE_SEGMENT_CAPACITY + 1)
      expect(index.segments.map((segment) => segment.count)).toEqual([
        ADDRESSED_SEQUENCE_SEGMENT_CAPACITY,
        1
      ])
      expect(index.nextAddressNumber).toBe(2)
      expect(index.segments.map((segment) => segment.address)).toEqual([
        'timeline/root/messages/entries/entry-0',
        'timeline/root/messages/entries/entry-1'
      ])
      expect(changed).toEqual(new Set(index.segments.map((segment) => segment.address)))
      // Written segments stay writer-pinned: nothing flushes at the appending commit itself.
      expect(yield* fixture.flushCount('test/sequence', index.segments[0]!.address)).toBe(0)
      expect(yield* fixture.flushCount('test/sequence', index.segments[1]!.address)).toBe(0)
      yield* runtime.flushDirty
      expect(yield* fixture.flushCount('test/sequence', index.segments[0]!.address)).toBe(1)
      expect(yield* fixture.flushCount('test/sequence', index.segments[1]!.address)).toBe(1)
    }))
  })

  test('update by item identity resolves through the index to one segment', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/root/messages',
        runtime
      })

      let index = sequence.empty
      let mutation = runtime.makeMutation()
      let transaction = mutation.transaction
      for (let i = 0; i < ADDRESSED_SEQUENCE_SEGMENT_CAPACITY + 1; i += 1) {
        index = yield* sequence.append(transaction, index, {
          id: `m${i}`,
          text: `message ${i}`
        })
      }
      yield* mutation.commit
      // Evict to the store so load counts observe exactly what the update reads.
      yield* runtime.flushDirty
      yield* runtime.reset

      mutation = runtime.makeMutation()
      transaction = mutation.transaction
      const originalIndex = index
      index = yield* sequence.updateById(transaction, index, 'm50', (message) => ({
        ...message,
        text: `${message.text}!`
      }))
      const changed = yield* mutation.commit
      const updatedAddress = index.segments[1]!.address

      expect(index).toBe(originalIndex)
      expect(changed).toEqual(new Set([updatedAddress]))
      expect(yield* fixture.loadCount('test/sequence', index.segments[0]!.address)).toBe(0)
      expect(yield* fixture.loadCount('test/sequence', updatedAddress)).toBe(1)
      expect(yield* fixture.flushCount('test/sequence', updatedAddress)).toBe(1)

      const updatedSegment = yield* runtime.get(updatedAddress)
      expect(Option.isSome(updatedSegment)).toBe(true)
      if (Option.isSome(updatedSegment)) {
        expect(updatedSegment.value.items[0]).toEqual({
          id: 'm50',
          text: 'message 50!'
        })
      }
    }))
  })

  test('update by item identity does not dirty a segment when the item is unchanged', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/root/messages',
        runtime
      })

      let mutation = runtime.makeMutation()
      let transaction = mutation.transaction
      const index = yield* sequence.append(transaction, sequence.empty, {
        id: 'stable',
        text: 'message'
      })
      yield* mutation.commit
      // Evict to the store so load counts observe exactly what the update reads.
      yield* runtime.flushDirty
      yield* runtime.reset

      mutation = runtime.makeMutation()
      transaction = mutation.transaction
      const updatedIndex = yield* sequence.updateById(transaction, index, 'stable', (message) => message)
      const changed = yield* mutation.commit
      const address = index.segments[0]!.address

      expect(updatedIndex).toBe(index)
      expect(changed).toEqual(new Set())
      expect(yield* fixture.loadCount('test/sequence', address)).toBe(1)
      expect(yield* fixture.flushCount('test/sequence', address)).toBe(1)
    }))
  })

  test('update by item identity updates the ordinary index when identity changes', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/root/messages',
        runtime
      })

      let mutation = runtime.makeMutation()
      let transaction = mutation.transaction
      const originalIndex = yield* sequence.append(transaction, sequence.empty, {
        id: 'draft',
        text: 'draft'
      })
      yield* mutation.commit

      mutation = runtime.makeMutation()
      transaction = mutation.transaction
      const updatedIndex = yield* sequence.updateById(transaction, originalIndex, 'draft', (message) => ({
        ...message,
        id: 'committed'
      }))
      const changed = yield* mutation.commit

      expect(updatedIndex).not.toBe(originalIndex)
      expect(updatedIndex.segments[0]?.itemIds).toEqual(['committed'])
      expect(updatedIndex.segments[0]?.address).not.toBe(originalIndex.segments[0]?.address)
      expect(sequence.resolveAddressForItem(updatedIndex, 'draft')).toEqual(Option.none())
      expect(sequence.resolveAddressForItem(updatedIndex, 'committed')).toEqual(Option.some(updatedIndex.segments[0]!.address))
      expect(changed).toEqual(new Set([updatedIndex.segments[0]!.address]))
    }))
  })

  test('append rejects duplicate item identities', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/root/messages',
        runtime
      })

      const mutation = runtime.makeMutation()
      const transaction = mutation.transaction
      let index = yield* sequence.append(transaction, sequence.empty, {
        id: 'duplicate',
        text: 'first'
      })
      const result = yield* sequence.append(transaction, index, {
        id: 'duplicate',
        text: 'second'
      }).pipe(Effect.either)

      expect(result._tag).toBe('Left')
      if (result._tag === 'Left') {
        expect(result.left).toMatchObject({
          _tag: 'AddressedCollectionError',
          collection: 'timeline/root/messages',
          operation: 'append',
          reason: 'item id "duplicate" appears more than once in sequence index'
        })
      }
      index = yield* sequence.append(transaction, index, {
        id: 'unique',
        text: 'still usable'
      })
      expect(index.totalCount).toBe(2)
    }))
  })

  test('identity-changing updates reject duplicate item identities', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/root/messages',
        runtime
      })

      let mutation = runtime.makeMutation()
      let transaction = mutation.transaction
      let index = yield* sequence.append(transaction, sequence.empty, {
        id: 'a',
        text: 'first'
      })
      index = yield* sequence.append(transaction, index, {
        id: 'b',
        text: 'second'
      })
      yield* mutation.commit

      mutation = runtime.makeMutation()
      transaction = mutation.transaction
      const result = yield* sequence.updateById(transaction, index, 'b', (message) => ({
        ...message,
        id: 'a'
      })).pipe(Effect.either)

      expect(result._tag).toBe('Left')
      if (result._tag === 'Left') {
        expect(result.left).toMatchObject({
          _tag: 'AddressedCollectionError',
          collection: 'timeline/root/messages',
          operation: 'updateById',
          reason: 'item id "a" appears more than once in sequence index'
        })
      }
    }))
  })

  test('range replacement rejects duplicate item identities across the resulting sequence', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/root/messages',
        runtime
      })

      let mutation = runtime.makeMutation()
      let transaction = mutation.transaction
      let index = yield* sequence.append(transaction, sequence.empty, {
        id: 'a',
        text: 'first'
      })
      index = yield* sequence.append(transaction, index, {
        id: 'b',
        text: 'second'
      })
      yield* mutation.commit

      mutation = runtime.makeMutation()
      transaction = mutation.transaction
      const result = yield* sequence.replaceRange(transaction, index, 1, 2, [
        { id: 'a', text: 'replacement' }
      ]).pipe(Effect.either)

      expect(result._tag).toBe('Left')
      if (result._tag === 'Left') {
        expect(result.left).toMatchObject({
          _tag: 'AddressedCollectionError',
          collection: 'timeline/root/messages',
          operation: 'replaceRange',
          reason: 'item id "a" appears more than once in sequence index'
        })
      }
    }))
  })

  test('tail windows resolve to the minimal segment set and read the requested slice', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/root/messages',
        runtime
      })

      let index = sequence.empty
      const mutation = runtime.makeMutation()
      const transaction = mutation.transaction
      for (let i = 0; i < ADDRESSED_SEQUENCE_SEGMENT_CAPACITY + 10; i += 1) {
        index = yield* sequence.append(transaction, index, {
          id: `m${i}`,
          text: `message ${i}`
        })
      }
      yield* mutation.commit

      const window = sequence.resolveTailWindow(index, 12)
      const messages = yield* sequence.readWindow(window)

      expect(window.map((part) => [part.segmentId, part.start, part.end])).toEqual([
        ['seg-0', ADDRESSED_SEQUENCE_SEGMENT_CAPACITY - 2, ADDRESSED_SEQUENCE_SEGMENT_CAPACITY],
        ['seg-1', 0, 10]
      ])
      expect(messages.map((message) => message.id)).toEqual([
        'm48',
        'm49',
        'm50',
        'm51',
        'm52',
        'm53',
        'm54',
        'm55',
        'm56',
        'm57',
        'm58',
        'm59'
      ])
    }))
  })

  test('range windows resolve arbitrary slices without reading unrelated segments', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/root/messages',
        runtime
      })

      let index = sequence.empty
      const mutation = runtime.makeMutation()
      const transaction = mutation.transaction
      for (let i = 0; i < (ADDRESSED_SEQUENCE_SEGMENT_CAPACITY * 2) + 5; i += 1) {
        index = yield* sequence.append(transaction, index, {
          id: `m${i}`,
          text: `message ${i}`
        })
      }
      yield* mutation.commit
      // Evict to the store so load counts observe exactly what the window reads.
      yield* runtime.flushDirty
      yield* runtime.reset

      const window = sequence.resolveRangeWindow(
        index,
        ADDRESSED_SEQUENCE_SEGMENT_CAPACITY - 2,
        5
      )
      const messages = yield* sequence.readWindow(window)

      expect(window.map((part) => [part.segmentId, part.start, part.end])).toEqual([
        ['seg-0', ADDRESSED_SEQUENCE_SEGMENT_CAPACITY - 2, ADDRESSED_SEQUENCE_SEGMENT_CAPACITY],
        ['seg-1', 0, 3]
      ])
      expect(messages.map((message) => message.id)).toEqual(['m48', 'm49', 'm50', 'm51', 'm52'])
      expect(yield* fixture.loadCount('test/sequence', index.segments[0]!.address)).toBe(1)
      expect(yield* fixture.loadCount('test/sequence', index.segments[1]!.address)).toBe(1)
      expect(yield* fixture.loadCount('test/sequence', index.segments[2]!.address)).toBe(0)
    }))
  })

  test('readAll materializes the logical sequence defined by the index', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const address = 'timeline/root/messages/entries/entry-0'
      const fixture = yield* makeCountingStore([
        ['test/sequence', address, {
          items: [
            { id: 'm0', text: 'indexed' },
            { id: 'stale', text: 'outside index' }
          ]
        }]
      ])
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/root/messages',
        runtime
      })

      const messages = yield* sequence.readAll({
        nextSegmentNumber: 1,
        nextAddressNumber: 1,
        totalCount: 1,
        segments: [{
          id: 'seg-0',
          address,
          start: 0,
          count: 1,
          itemIds: ['m0']
        }]
      })

      expect(messages).toEqual([{ id: 'm0', text: 'indexed' }])
    }))
  })

  test('readAll fails when the ordinary index expects items absent from the segment', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const address = 'timeline/root/messages/entries/entry-0'
      const fixture = yield* makeCountingStore([
        ['test/sequence', address, {
          items: [
            { id: 'm0', text: 'indexed' }
          ]
        }]
      ])
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/root/messages',
        runtime
      })

      const result = yield* sequence.readAll({
        nextSegmentNumber: 1,
        nextAddressNumber: 1,
        totalCount: 2,
        segments: [{
          id: 'seg-0',
          address,
          start: 0,
          count: 2,
          itemIds: ['m0', 'm1']
        }]
      }).pipe(Effect.either)

      expect(result._tag).toBe('Left')
      if (result._tag === 'Left') {
        expect(result.left._tag).toBe('AddressedCollectionError')
        if (result.left._tag === 'AddressedCollectionError') {
          expect(result.left.address).toBe(address)
          expect(result.left.reason).toBe('index expected 2 item(s), but segment contained 1')
        }
      }
    }))
  })

  test('readAll fails when segment count contradicts item-location metadata', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const address = 'timeline/root/messages/entries/entry-0'
      const fixture = yield* makeCountingStore([
        ['test/sequence', address, {
          items: [
            { id: 'm0', text: 'indexed' },
            { id: 'm1', text: 'also indexed' }
          ]
        }]
      ])
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/root/messages',
        runtime
      })

      const result = yield* sequence.readAll({
        nextSegmentNumber: 1,
        nextAddressNumber: 1,
        totalCount: 2,
        segments: [{
          id: 'seg-0',
          address,
          start: 0,
          count: 2,
          itemIds: ['m0']
        }]
      }).pipe(Effect.either)

      expect(result._tag).toBe('Left')
      if (result._tag === 'Left') {
        expect(result.left._tag).toBe('AddressedCollectionError')
        if (result.left._tag === 'AddressedCollectionError') {
          expect(result.left.address).toBe(address)
          expect(result.left.operation).toBe('readAll')
          expect(result.left.reason).toBe('segment count 2 did not match 1 item id(s)')
        }
      }
    }))
  })

  test('readAll fails when total count contradicts indexed segment counts', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const address = 'timeline/root/messages/entries/entry-0'
      const fixture = yield* makeCountingStore([
        ['test/sequence', address, {
          items: [
            { id: 'm0', text: 'indexed' }
          ]
        }]
      ])
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/root/messages',
        runtime
      })

      const result = yield* sequence.readAll({
        nextSegmentNumber: 1,
        nextAddressNumber: 1,
        totalCount: 2,
        segments: [{
          id: 'seg-0',
          address,
          start: 0,
          count: 1,
          itemIds: ['m0']
        }]
      }).pipe(Effect.either)

      expect(result._tag).toBe('Left')
      if (result._tag === 'Left') {
        expect(result.left._tag).toBe('AddressedCollectionError')
        if (result.left._tag === 'AddressedCollectionError') {
          expect(result.left.operation).toBe('readAll')
          expect(result.left.reason).toBe('total count 2 did not match 1 indexed item(s)')
        }
      }
    }))
  })

  test('readAll fails when indexed item identities contradict segment contents', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const address = 'timeline/root/messages/entries/entry-0'
      const fixture = yield* makeCountingStore([
        ['test/sequence', address, {
          items: [
            { id: 'actual', text: 'stored' }
          ]
        }]
      ])
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/root/messages',
        runtime
      })

      const result = yield* sequence.readAll({
        nextSegmentNumber: 1,
        nextAddressNumber: 1,
        totalCount: 1,
        segments: [{
          id: 'seg-0',
          address,
          start: 0,
          count: 1,
          itemIds: ['expected']
        }]
      }).pipe(Effect.either)

      expect(result._tag).toBe('Left')
      if (result._tag === 'Left') {
        expect(result.left._tag).toBe('AddressedCollectionError')
        if (result.left._tag === 'AddressedCollectionError') {
          expect(result.left.address).toBe(address)
          expect(result.left.operation).toBe('readAll')
          expect(result.left.reason).toBe('index expected item "expected" at offset 0, but segment contained "actual"')
        }
      }
    }))
  })

  test('append fails when indexed segment starts are not contiguous', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const address = 'timeline/root/messages/entries/entry-0'
      const fixture = yield* makeCountingStore([
        ['test/sequence', address, {
          items: [
            { id: 'm0', text: 'indexed' }
          ]
        }]
      ])
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/root/messages',
        runtime
      })
      const mutation = runtime.makeMutation()
      const transaction = mutation.transaction

      const result = yield* sequence.append(transaction, {
        nextSegmentNumber: 1,
        nextAddressNumber: 1,
        totalCount: 1,
        segments: [{
          id: 'seg-0',
          address,
          start: 1,
          count: 1,
          itemIds: ['m0']
        }]
      }, { id: 'm1', text: 'new' }).pipe(Effect.either)

      expect(result._tag).toBe('Left')
      if (result._tag === 'Left') {
        expect(result.left._tag).toBe('AddressedCollectionError')
        if (result.left._tag === 'AddressedCollectionError') {
          expect(result.left.address).toBe(address)
          expect(result.left.operation).toBe('append')
          expect(result.left.reason).toBe('segment start 1 did not match expected start 0')
        }
      }
    }))
  })

  test('readWindow fails when the requested logical window is absent from the segment', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const address = 'timeline/root/messages/entries/entry-0'
      const fixture = yield* makeCountingStore([
        ['test/sequence', address, {
          items: [
            { id: 'm0', text: 'indexed' }
          ]
        }]
      ])
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/root/messages',
        runtime
      })

      const result = yield* sequence.readWindow([{
        segmentId: 'seg-0',
        address,
        start: 0,
        end: 2,
        itemIds: ['m0', 'm1']
      }]).pipe(Effect.either)

      expect(result._tag).toBe('Left')
      if (result._tag === 'Left') {
        expect(result.left._tag).toBe('AddressedCollectionError')
        if (result.left._tag === 'AddressedCollectionError') {
          expect(result.left.address).toBe(address)
          expect(result.left.operation).toBe('readWindow')
          expect(result.left.reason).toBe('index expected 2 item(s), but segment contained 1')
        }
      }
    }))
  })

  test('readWindow fails when expected item identities contradict segment contents', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const address = 'timeline/root/messages/entries/entry-0'
      const fixture = yield* makeCountingStore([
        ['test/sequence', address, {
          items: [
            { id: 'actual', text: 'stored' }
          ]
        }]
      ])
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/root/messages',
        runtime
      })

      const result = yield* sequence.readWindow([{
        segmentId: 'seg-0',
        address,
        start: 0,
        end: 1,
        itemIds: ['expected']
      }]).pipe(Effect.either)

      expect(result._tag).toBe('Left')
      if (result._tag === 'Left') {
        expect(result.left._tag).toBe('AddressedCollectionError')
        if (result.left._tag === 'AddressedCollectionError') {
          expect(result.left.address).toBe(address)
          expect(result.left.operation).toBe('readWindow')
          expect(result.left.reason).toBe('window expected item "expected" at offset 0, but segment contained "actual"')
        }
      }
    }))
  })

  test('readWindow fails when the requested window is malformed', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const address = 'timeline/root/messages/entries/entry-0'
      const fixture = yield* makeCountingStore([
        ['test/sequence', address, {
          items: [
            { id: 'm0', text: 'indexed' }
          ]
        }]
      ])
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/root/messages',
        runtime
      })

      const result = yield* sequence.readWindow([{
        segmentId: 'seg-0',
        address,
        start: 2,
        end: 1,
        itemIds: []
      }]).pipe(Effect.either)

      expect(result._tag).toBe('Left')
      if (result._tag === 'Left') {
        expect(result.left._tag).toBe('AddressedCollectionError')
        if (result.left._tag === 'AddressedCollectionError') {
          expect(result.left.address).toBe(address)
          expect(result.left.operation).toBe('readWindow')
          expect(result.left.reason).toBe('invalid window 2..1')
        }
      }
    }))
  })

  test('append uses the indexed tail prefix instead of carrying a stale stored suffix', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const address = 'timeline/root/messages/entries/entry-0'
      const fixture = yield* makeCountingStore([
        ['test/sequence', address, {
          items: [
            { id: 'm0', text: 'indexed' },
            { id: 'stale', text: 'outside index' }
          ]
        }]
      ])
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/root/messages',
        runtime
      })
      const index = {
        nextSegmentNumber: 1,
        nextAddressNumber: 1,
        totalCount: 1,
        segments: [{
          id: 'seg-0',
          address,
          start: 0,
          count: 1,
          itemIds: ['m0']
        }]
      }

      const mutation = runtime.makeMutation()
      const transaction = mutation.transaction
      const nextIndex = yield* sequence.append(transaction, index, {
        id: 'm1',
        text: 'appended'
      })
      yield* mutation.commit
      yield* runtime.flushDirty

      expect(nextIndex.totalCount).toBe(2)
      expect(nextIndex.segments[0]?.itemIds).toEqual(['m0', 'm1'])
      expect(yield* fixture.valueFor('test/sequence', address)).toEqual({
        items: [
          { id: 'm0', text: 'indexed' },
          { id: 'm1', text: 'appended' }
        ]
      })
    }))
  })

  test('replaceAll rewrites the visible sequence through a new ordinary index', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/root/messages',
        runtime
      })

      let mutation = runtime.makeMutation()
      let transaction = mutation.transaction
      let index = yield* sequence.replaceAll(
        transaction,
        sequence.empty,
        Array.from({ length: ADDRESSED_SEQUENCE_SEGMENT_CAPACITY + 1 }, (_, i) => ({
          id: `old-${i}`,
          text: `old ${i}`
        }))
      )
      yield* mutation.commit
      yield* runtime.flushDirty

      mutation = runtime.makeMutation()
      transaction = mutation.transaction
      index = yield* sequence.replaceAll(transaction, index, [
        { id: 'new-1', text: 'new one' },
        { id: 'new-2', text: 'new two' }
      ])
      const changed = yield* mutation.commit
      yield* runtime.flushDirty

      const window = sequence.resolveTailWindow(index, 10)
      const messages = yield* sequence.readWindow(window)

      expect(index.totalCount).toBe(2)
      expect(index.segments.map((segment) => segment.id)).toEqual(['seg-0'])
      expect(changed).toEqual(new Set([index.segments[0]!.address]))
      expect(messages.map((message) => message.id)).toEqual(['new-1', 'new-2'])
    }))
  })

  test('replaceRange rewrites the affected suffix and leaves earlier segments untouched', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/root/messages',
        runtime
      })

      let mutation = runtime.makeMutation()
      let transaction = mutation.transaction
      let index = sequence.empty
      for (let i = 0; i < (ADDRESSED_SEQUENCE_SEGMENT_CAPACITY * 2) + 3; i += 1) {
        index = yield* sequence.append(transaction, index, {
          id: `m${i}`,
          text: `message ${i}`
        })
      }
      yield* mutation.commit
      // Evict to the store so load counts observe exactly what the rewrite reads.
      yield* runtime.flushDirty
      yield* runtime.reset
      const previousAddresses = index.segments.map((segment) => segment.address)

      mutation = runtime.makeMutation()
      transaction = mutation.transaction
      index = yield* sequence.replaceRange(
        transaction,
        index,
        ADDRESSED_SEQUENCE_SEGMENT_CAPACITY + 1,
        ADDRESSED_SEQUENCE_SEGMENT_CAPACITY + 3,
        [
          { id: 'inserted-a', text: 'inserted A' },
          { id: 'inserted-b', text: 'inserted B' },
          { id: 'inserted-c', text: 'inserted C' }
        ]
      )
      const changed = yield* mutation.commit

      expect(index.totalCount).toBe((ADDRESSED_SEQUENCE_SEGMENT_CAPACITY * 2) + 4)
      expect(changed).toEqual(new Set([
        index.segments[1]!.address,
        index.segments[2]!.address
      ]))
      expect(yield* fixture.loadCount('test/sequence', previousAddresses[0]!)).toBe(0)
      expect(yield* fixture.loadCount('test/sequence', previousAddresses[1]!)).toBe(1)
      expect(yield* fixture.loadCount('test/sequence', previousAddresses[2]!)).toBe(1)

      const messages = yield* sequence.readAll(index)
      expect(messages.slice(ADDRESSED_SEQUENCE_SEGMENT_CAPACITY - 1, ADDRESSED_SEQUENCE_SEGMENT_CAPACITY + 5).map((message) => message.id)).toEqual([
        'm49',
        'm50',
        'inserted-a',
        'inserted-b',
        'inserted-c',
        'm53'
      ])
    }))
  })

  test('structural insertion preserves old windows by moving the rewritten segment to a new address', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/root/messages',
        runtime
      })

      let mutation = runtime.makeMutation()
      let transaction = mutation.transaction
      let index = sequence.empty
      for (let i = 0; i < 6; i += 1) {
        index = yield* sequence.append(transaction, index, {
          id: `m${i}`,
          text: `message ${i}`
        })
      }
      yield* mutation.commit

      const oldAddress = index.segments[0]!.address
      const oldWindow = sequence.resolveRangeWindow(index, 4, 2)

      mutation = runtime.makeMutation()
      transaction = mutation.transaction
      const nextIndex = yield* sequence.replaceRange(transaction, index, 4, 4, [
        { id: 'inserted', text: 'inserted' }
      ])
      const changed = yield* mutation.commit

      expect(nextIndex.segments[0]!.address).not.toBe(oldAddress)
      expect(nextIndex.segments[0]!.address).toBe('timeline/root/messages/entries/entry-1')
      expect(nextIndex.nextAddressNumber).toBe(2)
      expect(changed).toEqual(new Set([nextIndex.segments[0]!.address]))

      const oldMessages = yield* sequence.readWindow(oldWindow)
      expect(oldMessages.map((message) => message.id)).toEqual(['m4', 'm5'])

      const nextMessages = yield* sequence.readWindow(sequence.resolveRangeWindow(nextIndex, 4, 3))
      expect(nextMessages.map((message) => message.id)).toEqual(['inserted', 'm4', 'm5'])
    }))
  })

  test('replaceRange sees writes staged earlier in the same transaction', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/root/messages',
        runtime
      })

      const mutation = runtime.makeMutation()
      const transaction = mutation.transaction
      let index = yield* sequence.append(transaction, sequence.empty, {
        id: 'draft',
        text: 'draft'
      })
      index = yield* sequence.replaceRange(transaction, index, 0, 1, [
        { id: 'committed', text: 'committed' }
      ])

      const messages = yield* sequence.readAllInTransaction(transaction, index)
      expect(messages).toEqual([{ id: 'committed', text: 'committed' }])
      expect(yield* fixture.loadCount('test/sequence', index.segments[0]!.address)).toBe(0)

      yield* mutation.commit
      const committed = yield* sequence.readAll(index)
      expect(committed).toEqual([{ id: 'committed', text: 'committed' }])
    }))
  })

  test('transaction reads see writes staged earlier in the same transaction', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/root/messages',
        runtime
      })

      const mutation = runtime.makeMutation()
      const transaction = mutation.transaction
      const index = yield* sequence.append(transaction, sequence.empty, {
        id: 'draft',
        text: 'draft'
      })

      const messages = yield* sequence.readAllInTransaction(transaction, index)

      expect(messages).toEqual([{ id: 'draft', text: 'draft' }])
      expect(yield* fixture.loadCount('test/sequence', index.segments[0]!.address)).toBe(0)
    }))
  })

  test('positionOfItem locates items by pure index arithmetic', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/root/messages',
        runtime
      })

      let index = sequence.empty
      const mutation = runtime.makeMutation()
      const transaction = mutation.transaction
      for (let i = 0; i < ADDRESSED_SEQUENCE_SEGMENT_CAPACITY + 3; i += 1) {
        index = yield* sequence.append(transaction, index, {
          id: `m${i}`,
          text: `message ${i}`
        })
      }
      yield* mutation.commit

      expect(sequence.positionOfItem(index, 'm0')).toEqual(Option.some(0))
      expect(sequence.positionOfItem(index, `m${ADDRESSED_SEQUENCE_SEGMENT_CAPACITY + 1}`))
        .toEqual(Option.some(ADDRESSED_SEQUENCE_SEGMENT_CAPACITY + 1))
      expect(Option.isNone(sequence.positionOfItem(index, 'absent'))).toBe(true)
      expect(yield* fixture.loadCount('test/sequence', index.segments[0]!.address)).toBe(0)
      expect(yield* fixture.loadCount('test/sequence', index.segments[1]!.address)).toBe(0)
    }))
  })

  test('removeById rewrites only the affected suffix and preserves captured windows', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/root/messages',
        runtime
      })

      let mutation = runtime.makeMutation()
      let transaction = mutation.transaction
      let index = sequence.empty
      for (let i = 0; i < (ADDRESSED_SEQUENCE_SEGMENT_CAPACITY * 2) + 3; i += 1) {
        index = yield* sequence.append(transaction, index, {
          id: `m${i}`,
          text: `message ${i}`
        })
      }
      yield* mutation.commit

      const untouchedAddress = index.segments[0]!.address
      const oldSecondAddress = index.segments[1]!.address
      const oldWindow = sequence.resolveRangeWindow(index, ADDRESSED_SEQUENCE_SEGMENT_CAPACITY, 3)

      mutation = runtime.makeMutation()
      transaction = mutation.transaction
      const nextIndex = yield* sequence.removeById(
        transaction,
        index,
        `m${ADDRESSED_SEQUENCE_SEGMENT_CAPACITY + 1}`
      )
      yield* mutation.commit

      expect(nextIndex.totalCount).toBe((ADDRESSED_SEQUENCE_SEGMENT_CAPACITY * 2) + 2)
      expect(nextIndex.segments[0]!.address).toBe(untouchedAddress)
      expect(nextIndex.segments[1]!.address).not.toBe(oldSecondAddress)
      expect(yield* fixture.loadCount('test/sequence', untouchedAddress)).toBe(0)
      expect(Option.isNone(sequence.positionOfItem(nextIndex, `m${ADDRESSED_SEQUENCE_SEGMENT_CAPACITY + 1}`))).toBe(true)

      const oldMessages = yield* sequence.readWindow(oldWindow)
      expect(oldMessages.map((message) => message.id)).toEqual([
        `m${ADDRESSED_SEQUENCE_SEGMENT_CAPACITY}`,
        `m${ADDRESSED_SEQUENCE_SEGMENT_CAPACITY + 1}`,
        `m${ADDRESSED_SEQUENCE_SEGMENT_CAPACITY + 2}`
      ])

      const nextMessages = yield* sequence.readWindow(
        sequence.resolveRangeWindow(nextIndex, ADDRESSED_SEQUENCE_SEGMENT_CAPACITY, 2)
      )
      expect(nextMessages.map((message) => message.id)).toEqual([
        `m${ADDRESSED_SEQUENCE_SEGMENT_CAPACITY}`,
        `m${ADDRESSED_SEQUENCE_SEGMENT_CAPACITY + 2}`
      ])
    }))
  })

  test('removeById of an absent item is a collection integrity error', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/root/messages',
        runtime
      })

      const mutation = runtime.makeMutation()
      const transaction = mutation.transaction
      const index = yield* sequence.append(transaction, sequence.empty, {
        id: 'present',
        text: 'present'
      })

      const removed = yield* Effect.either(sequence.removeById(transaction, index, 'absent'))
      expect(removed._tag).toBe('Left')
      if (removed._tag === 'Left') {
        expect(removed.left).toMatchObject({
          _tag: 'AddressedCollectionError',
          operation: 'removeById'
        })
      }
    }))
  })

  test('a streamed segment stays writer-pinned and flushes once on rotation', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeSegmentRuntime(fixture.store)
      const sequence = makeAddressedSequence<TestMessage>({
        prefix: 'timeline/hidden/messages',
        runtime
      })

      let index = sequence.empty
      let mutation = runtime.makeMutation()
      let transaction = mutation.transaction
      for (let i = 0; i < ADDRESSED_SEQUENCE_SEGMENT_CAPACITY + 1; i += 1) {
        index = yield* sequence.append(transaction, index, {
          id: `m${i}`,
          text: `message ${i}`
        })
      }
      yield* mutation.commit
      yield* runtime.flushDirty
      yield* runtime.reset

      const target = sequence.resolveAddressForItem(index, 'm50')
      expect(Option.isSome(target)).toBe(true)
      if (Option.isNone(target)) return

      // Streaming: repeated updates rewrite the same segment. The first update
      // loads it from the store; each commit keeps it writer-pinned —
      // resident and unflushed across the whole stream.
      for (let i = 0; i < 3; i += 1) {
        mutation = runtime.makeMutation()
        transaction = mutation.transaction
        index = yield* sequence.updateById(transaction, index, 'm50', (message) => ({
          ...message,
          text: `${message.text}.${i}`
        }))
        yield* mutation.commit
      }
      expect(yield* fixture.loadCount('test/sequence', target.value)).toBe(1)
      expect(yield* fixture.flushCount('test/sequence', target.value)).toBe(1)

      // A write elsewhere rotates the writer pin: the streamed segment is
      // flushed exactly once and dropped.
      mutation = runtime.makeMutation()
      yield* mutation.transaction.set('timeline/hidden/messages/entries/entry-99', { items: [] })
      yield* mutation.commit
      expect(yield* fixture.flushCount('test/sequence', target.value)).toBe(2)

      yield* runtime.get(target.value)
      expect(yield* fixture.loadCount('test/sequence', target.value)).toBe(2)
    }))
  })
})
