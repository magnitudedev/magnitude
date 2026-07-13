import { describe, expect, test } from 'bun:test'
import { Cause, Effect, Exit, Option, Ref, Schema } from 'effect'
import { makeAddressSpaceRuntime } from '../address-space'
import { makeSchemaCodec } from '../codec'
import {
  AddressedEntryStore,
  estimateAddressedStoredBytes,
  makeInMemoryAddressedEntryStore,
  type AddressedEntryStore as AddressedEntryStoreService
} from '../entry-store'
import { AddressedStoreError } from '../errors'

const TestMessageSchema = Schema.Struct({
  id: Schema.String,
  text: Schema.String
})

type TestMessage = Schema.Schema.Type<typeof TestMessageSchema>

const keyFor = (namespace: string, address: string) => `${namespace} ${address}`

const makeCountingStore = (
  initial: Iterable<readonly [namespace: string, address: string, value: unknown]> = [],
  options: {
    readonly failFlushFor?: ReadonlySet<string>
  } = {}
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
          if (options.failFlushFor?.has(keyFor(namespace, address))) {
            return yield* new AddressedStoreError({
              operation: 'flush',
              namespace,
              address,
              cause: 'flush failed in test'
            })
          }
          yield* Ref.update(entriesRef, (current) => {
            const next = new Map(current)
            next.set(keyFor(namespace, address), value)
            return next
          })
        })
    }

    return {
      store,
      loadCount: (namespace: string, address: string) => countFor(loadCountsRef, namespace, address),
      flushCount: (namespace: string, address: string) => countFor(flushCountsRef, namespace, address)
    }
  })

const makeRuntime = (store: AddressedEntryStoreService) =>
  makeAddressSpaceRuntime<TestMessage>({
    namespace: 'test/messages',
    codec: makeSchemaCodec(TestMessageSchema)
  }).pipe(
    Effect.provideService(AddressedEntryStore, store)
  )

describe('address-space runtime', () => {
  test('in-memory entry store preserves stored null distinctly from missing entries', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const store = yield* makeInMemoryAddressedEntryStore([
        ['test/nulls', 'present-null', null]
      ])

      const present = yield* store.load('test/nulls', 'present-null')
      const missing = yield* store.load('test/nulls', 'missing')

      expect(present).toEqual(Option.some(null))
      expect(Option.isNone(missing)).toBe(true)
    }))
  })

  test('reads load once and retain: repeat reads are pure memory', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore([
        ['test/messages', 'a', { id: 'a', text: 'hello' }]
      ])
      const runtime = yield* makeRuntime(fixture.store)

      const first = yield* runtime.get('a')
      const second = yield* runtime.get('a')
      expect(first).toEqual(Option.some({ id: 'a', text: 'hello' }))
      expect(second).toEqual(first)
      expect(yield* fixture.loadCount('test/messages', 'a')).toBe(1)
    }))
  })

  test('retained reads drop at the next settle boundary unless pinned', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore([
        ['test/messages', 'a', { id: 'a', text: 'transient' }]
      ])
      const runtime = yield* makeRuntime(fixture.store)

      yield* runtime.get('a')
      expect(yield* fixture.loadCount('test/messages', 'a')).toBe(1)

      // A write commit is a settle boundary: the unpinned retained read drops.
      yield* runtime.transact((transaction) =>
        transaction.set('b', { id: 'b', text: 'boundary' })
      )

      yield* runtime.get('a')
      expect(yield* fixture.loadCount('test/messages', 'a')).toBe(2)
    }))
  })

  test('a read of a missing address returns none and caches nothing', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeRuntime(fixture.store)

      expect(Option.isNone(yield* runtime.get('missing'))).toBe(true)
      expect(Option.isNone(yield* runtime.get('missing'))).toBe(true)
      expect(yield* fixture.loadCount('test/messages', 'missing')).toBe(2)
    }))
  })

  test('pins hold residency: one load on acquisition, none while pinned', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore([
        ['test/messages', 'a', { id: 'a', text: 'pinned' }]
      ])
      const runtime = yield* makeRuntime(fixture.store)

      yield* runtime.pin('view', ['a'])
      expect(yield* fixture.loadCount('test/messages', 'a')).toBe(1)

      yield* runtime.get('a')
      yield* runtime.get('a')
      expect(yield* fixture.loadCount('test/messages', 'a')).toBe(1)

      // Release drops the entry; the next read reloads.
      yield* runtime.unpin('view')
      yield* runtime.get('a')
      expect(yield* fixture.loadCount('test/messages', 'a')).toBe(2)
    }))
  })

  test('an entry stays resident until all pin owners release it', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore([
        ['test/messages', 'a', { id: 'a', text: 'shared' }]
      ])
      const runtime = yield* makeRuntime(fixture.store)

      yield* runtime.pin('view-1', ['a'])
      yield* runtime.pin('view-2', ['a'])
      expect(yield* fixture.loadCount('test/messages', 'a')).toBe(1)

      yield* runtime.unpin('view-1')
      yield* runtime.get('a')
      expect(yield* fixture.loadCount('test/messages', 'a')).toBe(1)

      yield* runtime.unpin('view-2')
      yield* runtime.get('a')
      expect(yield* fixture.loadCount('test/messages', 'a')).toBe(2)
    }))
  })

  test('committed writes stay resident under the writer pin and flush when rotated out', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeRuntime(fixture.store)

      const { changed } = yield* runtime.transact((transaction) =>
        transaction.set('a', { id: 'a', text: 'committed' })
      )
      expect(changed).toEqual(new Set(['a']))

      // Writer-pinned: resident and dirty, no flush yet.
      expect(yield* fixture.flushCount('test/messages', 'a')).toBe(0)
      expect(yield* runtime.get('a')).toEqual(Option.some({ id: 'a', text: 'committed' }))
      expect(yield* fixture.loadCount('test/messages', 'a')).toBe(0)

      // The next write transaction rotates the writer pin: 'a' loses its
      // last pin, gets flushed exactly once, and drops.
      yield* runtime.transact((transaction) =>
        transaction.set('b', { id: 'b', text: 'next' })
      )
      expect(yield* fixture.flushCount('test/messages', 'a')).toBe(1)
      expect(yield* runtime.get('a')).toEqual(Option.some({ id: 'a', text: 'committed' }))
      expect(yield* fixture.loadCount('test/messages', 'a')).toBe(1)
    }))
  })

  test('transactions that stage nothing do not rotate the writer pin', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeRuntime(fixture.store)

      yield* runtime.transact((transaction) =>
        transaction.set('a', { id: 'a', text: 'hot' })
      )

      yield* runtime.transact(() => Effect.void)

      expect(yield* fixture.flushCount('test/messages', 'a')).toBe(0)
      expect(yield* runtime.get('a')).toEqual(Option.some({ id: 'a', text: 'hot' }))
      expect(yield* fixture.loadCount('test/messages', 'a')).toBe(0)
    }))
  })

  test('streaming pattern: rewriting the same address never flushes until rotation or snapshot', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeRuntime(fixture.store)

      for (const text of ['h', 'he', 'hel', 'hell', 'hello']) {
        yield* runtime.transact((transaction) =>
          transaction.set('a', { id: 'a', text })
        )
      }
      expect(yield* fixture.flushCount('test/messages', 'a')).toBe(0)
      expect(yield* fixture.loadCount('test/messages', 'a')).toBe(0)

      yield* runtime.transact((transaction) =>
        transaction.set('b', { id: 'b', text: 'rotates' })
      )
      expect(yield* fixture.flushCount('test/messages', 'a')).toBe(1)

      expect(yield* runtime.get('a')).toEqual(Option.some({ id: 'a', text: 'hello' }))
      expect(yield* fixture.loadCount('test/messages', 'a')).toBe(1)
    }))
  })

  test('a reader pin carries a dirty entry across writer rotation; it flushes once on last release', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeRuntime(fixture.store)

      yield* runtime.transact((transaction) =>
        transaction.set('a', { id: 'a', text: 'displayed while streaming' })
      )
      yield* runtime.pin('view', ['a'])

      // Rotation releases the writer pin, but the reader still holds the
      // entry: resident and dirty, no flush.
      yield* runtime.transact((transaction) =>
        transaction.set('b', { id: 'b', text: 'rotates' })
      )
      expect(yield* fixture.flushCount('test/messages', 'a')).toBe(0)
      expect(yield* runtime.get('a')).toEqual(Option.some({ id: 'a', text: 'displayed while streaming' }))
      expect(yield* fixture.loadCount('test/messages', 'a')).toBe(0)

      // The last pin releasing flushes the dirty entry exactly once and drops it.
      yield* runtime.unpin('view')
      expect(yield* fixture.flushCount('test/messages', 'a')).toBe(1)
      expect(yield* runtime.get('a')).toEqual(Option.some({ id: 'a', text: 'displayed while streaming' }))
      expect(yield* fixture.loadCount('test/messages', 'a')).toBe(1)
    }))
  })

  test('markChanged joins the changed set without creating entries or rotating the writer pin', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeRuntime(fixture.store)

      yield* runtime.transact((transaction) =>
        transaction.set('a', { id: 'a', text: 'hot' })
      )

      const { changed } = yield* runtime.transact((transaction) =>
        Effect.sync(() => {
          transaction.markChanged('prefix#index')
        })
      )
      expect(changed).toEqual(new Set(['prefix#index']))

      // Marks-only: nothing flushed, nothing dropped, no entry created.
      expect(yield* fixture.flushCount('test/messages', 'a')).toBe(0)
      expect(yield* fixture.loadCount('test/messages', 'a')).toBe(0)
      expect(yield* runtime.get('a')).toEqual(Option.some({ id: 'a', text: 'hot' }))
      expect(Option.isNone(yield* runtime.get('prefix#index'))).toBe(true)
    }))
  })

  test('a write and a mark in the same transaction report both as changed', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeRuntime(fixture.store)

      const { changed } = yield* runtime.transact((transaction) =>
        Effect.gen(function* () {
          yield* transaction.set('a', { id: 'a', text: 'appended' })
          transaction.markChanged('prefix#index')
        })
      )
      expect(changed).toEqual(new Set(['a', 'prefix#index']))
    }))
  })

  test('transaction reads see staged writes before commit', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeRuntime(fixture.store)

      yield* runtime.transact((transaction) =>
        Effect.gen(function* () {
          yield* transaction.set('a', { id: 'a', text: 'staged' })
          const staged = yield* transaction.get('a')
          expect(staged).toEqual(Option.some({ id: 'a', text: 'staged' }))
        })
      )
    }))
  })

  test('flushDirty cleans writer-pinned entries in place for snapshot capture', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeRuntime(fixture.store)

      yield* runtime.transact((transaction) =>
        transaction.set('a', { id: 'a', text: 'snapshotted' })
      )

      yield* runtime.flushDirty
      expect(yield* fixture.flushCount('test/messages', 'a')).toBe(1)

      // Still resident (still writer-pinned), now clean.
      expect(yield* runtime.get('a')).toEqual(Option.some({ id: 'a', text: 'snapshotted' }))
      expect(yield* fixture.loadCount('test/messages', 'a')).toBe(0)

      // Rotation drops it without a second flush — it is already clean.
      yield* runtime.transact((transaction) =>
        transaction.set('b', { id: 'b', text: 'rotates' })
      )
      expect(yield* fixture.flushCount('test/messages', 'a')).toBe(1)
      yield* runtime.get('a')
      expect(yield* fixture.loadCount('test/messages', 'a')).toBe(1)
    }))
  })

  test('pin replacement acquires new addresses and releases old ones atomically', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore([
        ['test/messages', 'a', { id: 'a', text: 'old' }],
        ['test/messages', 'b', { id: 'b', text: 'new' }]
      ])
      const runtime = yield* makeRuntime(fixture.store)

      yield* runtime.pin('view', ['a'])
      yield* runtime.pin('view', ['b'])
      expect(yield* fixture.loadCount('test/messages', 'a')).toBe(1)
      expect(yield* fixture.loadCount('test/messages', 'b')).toBe(1)

      yield* runtime.get('b')
      expect(yield* fixture.loadCount('test/messages', 'b')).toBe(1)

      // 'a' was released by the replacement and dropped.
      yield* runtime.get('a')
      expect(yield* fixture.loadCount('test/messages', 'a')).toBe(2)
    }))
  })

  test('pinning an address with invalid stored contents fails with a codec error', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore([
        ['test/messages', 'bad', { id: 42, text: 'not valid' }]
      ])
      const runtime = yield* makeRuntime(fixture.store)

      const pinned = yield* Effect.either(runtime.pin('view', ['bad']))
      expect(pinned._tag).toBe('Left')
      if (pinned._tag === 'Left') {
        expect(pinned.left._tag).toBe('AddressedCodecError')
      }
      expect(yield* fixture.loadCount('test/messages', 'bad')).toBe(1)
    }))
  })

  test('pinning an address that is neither resident nor stored is a defect', async () => {
    const fixture = await Effect.runPromise(makeCountingStore())
    const exit = await Effect.runPromiseExit(Effect.gen(function* () {
      const runtime = yield* makeRuntime(fixture.store)
      yield* runtime.pin('view', ['missing'])
    }))

    expect(Exit.isFailure(exit)).toBe(true)
    if (Exit.isFailure(exit)) {
      expect(Cause.isDie(exit.cause)).toBe(true)
    }
  })

  test('flush failure surfaces at the commit that rotates the entry out', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore([], {
        failFlushFor: new Set(['test/messages a'])
      })
      const runtime = yield* makeRuntime(fixture.store)

      // The write itself commits fine — it stays writer-pinned, unflushed.
      yield* runtime.transact((transaction) =>
        transaction.set('a', { id: 'a', text: 'dirty' })
      )
      expect(yield* fixture.flushCount('test/messages', 'a')).toBe(0)

      // Rotation attempts the flush; the failure aborts that commit and the
      // dirty entry stays resident under the writer pin.
      const result = yield* runtime.transact((transaction) =>
        transaction.set('b', { id: 'b', text: 'rotates' })
      ).pipe(Effect.either)

      expect(result._tag).toBe('Left')
      if (result._tag === 'Left') {
        expect(result.left).toMatchObject({
          _tag: 'AddressedStoreError',
          operation: 'flush',
          namespace: 'test/messages',
          address: 'a'
        })
      }
      expect(yield* fixture.flushCount('test/messages', 'a')).toBe(1)
      expect(yield* runtime.get('a')).toEqual(Option.some({ id: 'a', text: 'dirty' }))
      expect(yield* fixture.loadCount('test/messages', 'a')).toBe(0)
    }))
  })

  test('reset clears resident entries and all pins', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore([
        ['test/messages', 'a', { id: 'a', text: 'stored' }]
      ])
      const runtime = yield* makeRuntime(fixture.store)

      yield* runtime.pin('view', ['a'])
      expect(yield* fixture.loadCount('test/messages', 'a')).toBe(1)

      yield* runtime.reset
      yield* runtime.get('a')
      expect(yield* fixture.loadCount('test/messages', 'a')).toBe(2)

      // The pin table was cleared too: unpin after reset is a no-op.
      yield* runtime.unpin('view')
      expect(yield* fixture.loadCount('test/messages', 'a')).toBe(2)
    }))
  })

  test('bound mutation commits are idempotent', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fixture = yield* makeCountingStore()
      const runtime = yield* makeRuntime(fixture.store)

      const mutation = runtime.makeMutation()
      yield* mutation.transaction.set('a', { id: 'a', text: 'once' })
      const firstCommit = yield* mutation.commit
      const secondCommit = yield* mutation.commit

      expect(firstCommit).toEqual(new Set(['a']))
      expect(secondCommit).toEqual(new Set())

      const committed = yield* runtime.get('a')
      expect(committed).toEqual(Option.some({ id: 'a', text: 'once' }))
    }))
  })
})
