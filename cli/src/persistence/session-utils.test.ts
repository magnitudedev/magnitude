import { describe, expect, test } from 'vitest'
import { listAllSessions, loadSessionSummary } from './session-utils'

describe('session-utils', () => {
  test('listAllSessions reads titles from metadata only', async () => {
    const storage = {
      sessions: {
        list: async () => ['a', 'b'],
        readMeta: async (id: string) => ({
          sessionId: id,
          created: '2026-01-01T00:00:00.000Z',
          updated: id === 'a' ? '2026-01-02T00:00:00.000Z' : '2026-01-03T00:00:00.000Z',
          chatName: id === 'a' ? 'Alpha' : 'Beta',
          workingDirectory: '/tmp',
          initialVersion: '0.0.1',
          lastActiveVersion: '0.0.1',
          gitBranch: null,
          firstUserMessage: null,
          lastMessage: 'hello',
          messageCount: 1,
        }),
        readEvents: async () => {
          throw new Error('should not read events')
        },
      },
    } as any

    const result = await listAllSessions(storage)

    expect(result.map((session) => session.title)).toEqual(['Beta', 'Alpha'])
  })

  test('loadSessionSummary returns chatName from metadata', async () => {
    const storage = {
      sessions: {
        readMeta: async () => ({
          sessionId: 'a',
          created: '2026-01-01T00:00:00.000Z',
          updated: '2026-01-02T00:00:00.000Z',
          chatName: 'Stored Title',
          workingDirectory: '/tmp',
          initialVersion: '0.0.1',
          lastActiveVersion: '0.0.1',
          gitBranch: null,
          firstUserMessage: null,
          lastMessage: null,
          messageCount: 0,
        }),
      },
    } as any

    const result = await loadSessionSummary(storage, 'a')

    expect(result?.title).toBe('Stored Title')
  })
})
