import type { StorageClient, StoredSessionMeta } from '@magnitudedev/storage'

const DEFAULT_CHAT_NAME = 'New Chat'

export interface SessionSummary {
  sessionId: string
  title: string
  lastMessage: string
  timestamp: number
  messageCount: number
}

export async function listAllSessions(storage: StorageClient, limit?: number): Promise<SessionSummary[]> {
  const ids = await storage.sessions.list()
  const summaries: SessionSummary[] = []

  for (const id of ids) {
    try {
      const meta = await storage.sessions.readMeta(id)
      if (meta) {
        summaries.push(buildSessionSummary(meta))
      }
    } catch (error) {
      console.error(`Failed to load session ${id}:`, error)
    }
  }

  summaries.sort((a, b) => b.timestamp - a.timestamp)
  return limit ? summaries.slice(0, limit) : summaries
}

export async function loadSessionSummary(storage: StorageClient, sessionId: string): Promise<SessionSummary | null> {
  const meta = await storage.sessions.readMeta(sessionId)
  if (!meta) return null
  return buildSessionSummary(meta)
}

function buildSessionSummary(meta: StoredSessionMeta): SessionSummary {
  const timestamp = Date.parse(meta.updated)

  return {
    sessionId: meta.sessionId,
    title: meta.chatName,
    lastMessage: meta.lastMessage ?? 'No messages yet',
    timestamp: Number.isNaN(timestamp) ? Date.parse(meta.created) : timestamp,
    messageCount: meta.messageCount,
  }
}
