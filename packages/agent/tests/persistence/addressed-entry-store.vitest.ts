import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, test, beforeEach, afterEach } from 'vitest'
import { Effect, Layer, Option } from 'effect'
import { BunFileSystem, BunPath } from '@effect/platform-bun'
import { Addressed, type EventCursor } from '@magnitudedev/event-core'
import {
  GlobalStorage,
  MagnitudeStorage,
  ProjectStorage,
  StorageLive,
  Version,
  makeGlobalStoragePaths,
  makeProjectStoragePaths
} from '@magnitudedev/storage'
import type { Timestamped } from '@magnitudedev/event-core'
import type { AppEvent } from '../../src/events'
import {
  ChatPersistence,
  PersistenceError,
  type ChatPersistenceService,
  type SessionMetadata
} from '../../src/persistence/chat-persistence-service'
import { makeChatAddressedEntryStoreLayer } from '../../src/persistence/addressed-entry-store'

const VERSION = '0.0.1'

const metadata: SessionMetadata = {
  sessionId: 'session-1',
  chatName: 'Addressed Entry Store Test',
  workingDirectory: '/repo',
  gitBranch: null,
  created: '2026-01-01T00:00:00.000Z',
  updated: '2026-01-01T00:00:00.000Z',
  initialVersion: VERSION,
  lastActiveVersion: VERSION
}

const makeStorageLayer = (root: string) => {
  const base = Layer.mergeAll(
    BunFileSystem.layer,
    BunPath.layer,
    Layer.succeed(Version, Version.of({ getVersion: () => VERSION })),
    Layer.succeed(GlobalStorage, GlobalStorage.of({
      root,
      paths: makeGlobalStoragePaths(root)
    })),
    Layer.succeed(ProjectStorage, ProjectStorage.of({
      cwd: '/repo',
      root: join(root, 'project'),
      paths: makeProjectStoragePaths(root)
    }))
  )
  return StorageLive.pipe(Layer.provide(base))
}

const makePersistence = (metadataCalls: string[]): ChatPersistenceService => ({
  loadEvents: () => Effect.succeed([]),
  loadEventsAfterCursor: () => Effect.succeed([]),
  persistNewEvents: (_events: Timestamped<AppEvent>[]) =>
    Effect.succeed({ index: 0, timestamp: 1 } satisfies EventCursor),
  loadProjectionSnapshot: () => Effect.succeed(null),
  saveProjectionSnapshot: () => Effect.void,
  getSessionMetadata: () =>
    Effect.sync(() => {
      metadataCalls.push('metadata')
      return metadata
    }),
  saveSessionMetadata: () => Effect.void
})

describe('addressed entry store persistence binding', () => {
  let tmpDir: string

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), 'magnitude-agent-addressed-store-'))
  })

  afterEach(async () => {
    await rm(tmpDir, { recursive: true, force: true })
  })

  test('writes event-core addressed entries through session storage', async () => {
    const metadataCalls: string[] = []

    await Effect.runPromise(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage
        const layer = Layer.provideMerge(
          makeChatAddressedEntryStoreLayer(storage),
          Layer.succeed(ChatPersistence, makePersistence(metadataCalls))
        )

        yield* Effect.gen(function* () {
          const store = yield* Addressed.AddressedEntryStore
          const namespace = 'DisplayTimeline/messages'
          const address = 'DisplayTimeline/messages/members/root/entries/entry-0'
          const value = { items: [{ id: 'm1', text: 'hello' }] }

          yield* store.flush(namespace, address, value)

          const loaded = yield* store.load(namespace, address)
          expect(loaded).toEqual(Option.some(value))

          const missing = yield* store.load(namespace, 'missing')
          expect(Option.isNone(missing)).toBe(true)

          const raw = yield* storage.sessions.readAddressedEntry(
            metadata.sessionId,
            namespace,
            address
          )
          expect(raw).toEqual({ value })
        }).pipe(Effect.provide(layer))
      }).pipe(Effect.provide(makeStorageLayer(tmpDir)))
    )

    expect(metadataCalls).toEqual(['metadata'])
  })

  test('maps session metadata lookup failure into addressed store errors', async () => {
    await Effect.runPromise(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage
        const failingPersistence: ChatPersistenceService = {
          ...makePersistence([]),
          getSessionMetadata: () =>
            Effect.fail(new PersistenceError({
              reason: 'LoadFailed',
              message: 'metadata unavailable'
            }))
        }
        const layer = Layer.provideMerge(
          makeChatAddressedEntryStoreLayer(storage),
          Layer.succeed(ChatPersistence, failingPersistence)
        )

        const result = yield* Effect.gen(function* () {
          const store = yield* Addressed.AddressedEntryStore
          return yield* Effect.either(store.load('DisplayTimeline/messages', 'root/seg-0'))
        }).pipe(Effect.provide(layer))

        expect(result._tag).toBe('Left')
        if (result._tag === 'Left') {
          expect(result.left).toMatchObject({
            _tag: 'AddressedStoreError',
            operation: 'load',
            namespace: 'DisplayTimeline/messages',
            address: 'root/seg-0'
          })
        }
      }).pipe(Effect.provide(makeStorageLayer(tmpDir)))
    )
  })
})
