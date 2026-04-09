import { mkdir, mkdtemp } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import { Effect, Schema } from 'effect'
import { describe, expect, test } from 'vitest'

import { makeGlobalStoragePaths } from '../paths'
import { VersionLive } from '../services/version'
import { readRawSessionMeta } from '../sessions'
import { StoredSessionMetaSchema } from './session'

describe('StoredSessionMetaSchema', () => {
  test('decodes valid metadata with version fields', async () => {
    const result = await Effect.runPromise(
      Schema.decodeUnknown(StoredSessionMetaSchema)({
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
      }).pipe(Effect.provide(VersionLive('9.9.9')))
    )

    expect(result.initialVersion).toBe('0.0.1')
    expect(result.lastActiveVersion).toBe('0.0.1')
  })

  test('missing version fields default from Version service', async () => {
    const result = await Effect.runPromise(
      Schema.decodeUnknown(StoredSessionMetaSchema)({
        sessionId: 'session-1',
        created: '2026-01-01T00:00:00.000Z',
        updated: '2026-01-01T00:00:00.000Z',
        chatName: 'Chat',
        workingDirectory: '/repo',
        gitBranch: null,
        firstUserMessage: null,
        lastMessage: null,
        messageCount: 0,
      }).pipe(Effect.provide(VersionLive('9.9.9')))
    )

    expect(result.initialVersion).toBe('9.9.9')
    expect(result.lastActiveVersion).toBe('9.9.9')
  })

  test('raw session metadata can be decoded with Version-backed defaults', async () => {
    const root = await mkdtemp(join(tmpdir(), 'magnitude-storage-session-'))
    const paths = makeGlobalStoragePaths(root)
    const sessionId = 'session-1'

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

    const raw = await readRawSessionMeta(paths, sessionId)
    const result = await Effect.runPromise(
      Schema.decodeUnknown(StoredSessionMetaSchema)(raw).pipe(
        Effect.provide(VersionLive('9.9.9'))
      )
    )

    expect(result.initialVersion).toBe('9.9.9')
    expect(result.lastActiveVersion).toBe('9.9.9')
  })
})
