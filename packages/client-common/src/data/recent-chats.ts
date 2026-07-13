/**
 * Recent chats data types and formatting helpers.
 *
 * The actual session data is fetched from the ACN daemon via the SDK
 * client's `listSessions` RPC. This module only holds the local view
 * types and time formatting.
 */

export interface RecentChat {
  id: string
  title: string
  lastMessage: string
  timestamp: number
  messageCount: number
  workingDirectory: string
}

export interface RecentChatsPage {
  items: RecentChat[]
  hasMore: boolean
}

export interface FormatCwdOptions {
  readonly maxLen?: number
  readonly abbreviateHome?: boolean
  /** Explicit home directory — when provided, used for precise abbreviation. */
  readonly homeDir?: string
}

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

function abbreviateHomePath(cwd: string, homeDir?: string): string {
  if (homeDir) {
    if (cwd === homeDir) return '~'
    if (cwd.startsWith(homeDir + '/') || cwd.startsWith(homeDir + '\\'))
      return '~' + cwd.slice(homeDir.length)
  }
  // macOS:  /Users/<name>
  // Linux:  /home/<name>
  let replaced = cwd.replace(/^\/(?:Users|home)\/[^/]+/, '~')
  // Windows:  C:\Users\<name>  or  C:/Users/<name>
  replaced = replaced.replace(/^[A-Za-z]:[\\/](?:Users|home)[\\/][^\\/]+/i, '~')
  return replaced
}

export function formatCwdForDisplay(
  cwd: string,
  options: FormatCwdOptions = {},
): string {
  const maxLen = options.maxLen ?? 40
  const display = options.abbreviateHome === false
    ? cwd
    : abbreviateHomePath(cwd, options.homeDir)

  if (display.length <= maxLen) return display

  const parts = display.split('/')
  if (parts.length <= 2) return '…' + display.slice(-maxLen + 1)
  return '…/' + parts.slice(-2).join('/')
}

import type { SessionMetadata } from "@magnitudedev/sdk"

/** Map daemon session metadata to the client RecentChat view model. */
export function sessionsToRecentChats(sessions: readonly SessionMetadata[]): RecentChat[] {
  return sessions.map((session) => ({
    id: session.sessionId,
    title: session.title ?? "Untitled",
    lastMessage: session.lastMessage ?? "",
    timestamp: session.updatedAt,
    messageCount: session.messageCount,
    workingDirectory: session.cwd,
  }))
}
