/**
 * Recent chats data layer
 * 
 * Uses centralized session utilities to load session summaries
 */

import type { StorageClient } from '@magnitudedev/storage'
import { listAllSessions, type SessionSummary } from '../persistence/session-utils'

export interface RecentChat {
  id: string
  title: string
  lastMessage: string
  timestamp: number
  messageCount: number
}

const MAX_RECENT_CHATS = 100

/**
 * Get recent chats from all available sessions
 */
export async function getRecentChats(storage: StorageClient, limit = MAX_RECENT_CHATS): Promise<RecentChat[]> {
  const summaries = await listAllSessions(storage, limit)
  
  return summaries.map(summary => ({
    id: summary.sessionId,
    title: summary.title,
    lastMessage: summary.lastMessage,
    timestamp: summary.timestamp,
    messageCount: summary.messageCount,
  }))
}

/**
 * Format a timestamp as relative time
 */
export function formatRelativeTime(timestamp: number): string {
  const now = Date.now()
  const diffMs = now - timestamp
  const diffMinutes = Math.floor(diffMs / (1000 * 60))
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60))
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))

  if (diffMinutes < 1) return 'just now'
  if (diffMinutes < 60) return `${diffMinutes}m ago`
  if (diffHours < 24) return `${diffHours}h ago`
  if (diffDays === 1) return 'yesterday'
  return `${diffDays}d ago`
}
