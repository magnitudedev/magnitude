import { mkdir, mkdtemp, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import { Effect, Layer, Schema } from 'effect'
import { BunFileSystem, BunPath } from '@effect/platform-bun'
import { describe, expect, test } from 'vitest'

import { makeGlobalStoragePaths, makeProjectStoragePaths } from '../paths'
import { GlobalStorage } from '../services/global-storage'
import { ProjectStorage } from '../services/project-storage'
import { Version } from '../services/version'
import { MagnitudeStorage, StorageLive } from '../storage'
import { makeStoredSessionMetaSchema } from './session'

const VERSION = '9.9.9'

function makeTestLayer(root: string) {
  const base = Layer.mergeAll(
    BunFileSystem.layer,
    BunPath.layer,
    Layer.succeed(Version, Version.of({ getVersion: () => VERSION })),
    Layer.succeed(GlobalStorage, GlobalStorage.of({
      root,
      paths: makeGlobalStoragePaths(root),
    })),
    Layer.succeed(ProjectStorage, ProjectStorage.of({
      cwd: '/repo',
      root: join(root, 'project'),
      paths: makeProjectStoragePaths(root),
    })),
  )
  return StorageLive.pipe(Layer.provide(base))
}

describe('StoredSessionMetaSchema', () => {
  test('decodes valid metadata with version fields', async () => {
    const schema = makeStoredSessionMetaSchema(VERSION)
    const result = await Effect.runPromise(
      Schema.decodeUnknown(schema)({
        sessionId: 'session-1',
        created: '2026-01-01T00:00:00.000Z',
        updated: '2026-01-01T00:00:00.000Z',
        chatName: 'Chat',
        workingDirectory: '/repo',
        initialVersion: '0.0.1',
        lastActiveVersion: '0.0.1',
        gitBranch: null,
        firstUserMessage: null,
        lastMessage: null,
        messageCount: 0,
      })
    )

    expect(result.initialVersion).toBe('0.0.1')
    expect(result.lastActiveVersion).toBe('0.0.1')
  })

  test('missing version fields default from provided version', async () => {
    const schema = makeStoredSessionMetaSchema(VERSION)
    const result = await Effect.runPromise(
      Schema.decodeUnknown(schema)({
        sessionId: 'session-1',
        created: '2026-01-01T00:00:00.000Z',
        updated: '2026-01-01T00:00:00.000Z',
        chatName: 'Chat',
        workingDirectory: '/repo',
        gitBranch: null,
        firstUserMessage: null,
        lastMessage: null,
        messageCount: 0,
      })
    )

    expect(result.initialVersion).toBe(VERSION)
    expect(result.lastActiveVersion).toBe(VERSION)
  })

  test('readMeta returns version defaults for missing fields via StorageLive', async () => {
    const root = await mkdtemp(join(tmpdir(), 'magnitude-storage-session-'))
    const sessionId = 'session-1'
    const paths = makeGlobalStoragePaths(root)

    try {
      await mkdir(paths.sessionDir(sessionId), { recursive: true })
      await Bun.write(
        paths.sessionMetaFile(sessionId),
        JSON.stringify({
          sessionId,
          created: '2026-01-01T00:00:00.000Z',
          updated: '2026-01-01T00:00:00.000Z',
          chatName: 'Chat',
          workingDirectory: '/repo',
          gitBranch: null,
          firstUserMessage: null,
          lastMessage: null,
          messageCount: 0,
        })
      )

      const result = await Effect.runPromise(
        Effect.gen(function* () {
          const storage = yield* MagnitudeStorage
          return yield* storage.sessions.readMeta(sessionId)
        }).pipe(Effect.provide(makeTestLayer(root)))
      )

      expect(result?.initialVersion).toBe(VERSION)
      expect(result?.lastActiveVersion).toBe(VERSION)
    } finally {
      await rm(root, { recursive: true, force: true })
    }
  })
})
