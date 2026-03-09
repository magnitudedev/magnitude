/**
 * Session utilities for loading and listing sessions
 * 
 * Supports UTC timestamp directory format only:
 * - Directory name: YYYY-MM-DDTHH-MM-SSZ (e.g., 2026-02-11T06-24-32Z)
 * - Contains: meta.json, events.jsonl, logs.jsonl
 */

import { readdirSync, statSync, existsSync } from 'fs'
import { join } from 'path'
import { homedir } from 'os'

const SESSIONS_DIR = join(homedir(), '.magnitude', 'sessions')
const DEFAULT_CHAT_NAME = 'New Chat'

export interface SessionSummary {
  sessionId: string
  title: string
  lastMessage: string
  timestamp: number
  messageCount: number
}

interface MetadataFile {
  sessionId: string
  created: string
  updated: string
  chatName: string
  workingDirectory: string
  gitBranch: string | null
  firstUserMessage: string | null
  lastMessage: string | null
  messageCount: number
}


/**
 * List all sessions from UTC timestamp directories
 */
export async function listAllSessions(limit = 100): Promise<SessionSummary[]> {
  if (!existsSync(SESSIONS_DIR)) return []

  const entries = readdirSync(SESSIONS_DIR, { withFileTypes: true })
  const summaries: SessionSummary[] = []

  for (const entry of entries) {
    // Only process directories with UTC timestamp format
    if (!entry.isDirectory() || !/^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$/.test(entry.name)) {
      continue
    }

    try {
      const summary = await loadDirectorySessionSummary(entry.name)
      if (summary) summaries.push(summary)
    } catch (error) {
      // Skip sessions that fail to load
      console.error(`Failed to load session ${entry.name}:`, error)
    }
  }

  return summaries
    .sort((a, b) => b.timestamp - a.timestamp)
    .slice(0, limit)
}

/**
 * Load a session summary by ID
 */
export async function loadSessionSummary(sessionId: string): Promise<SessionSummary | null> {
  const dirPath = join(SESSIONS_DIR, sessionId)
  if (!existsSync(dirPath) || !statSync(dirPath).isDirectory()) {
    return null
  }

  return loadDirectorySessionSummary(sessionId)
}

/**
 * Load summary from directory format
 */
async function loadDirectorySessionSummary(dirName: string): Promise<SessionSummary | null> {
  const dirPath = join(SESSIONS_DIR, dirName)
  const metaPath = join(dirPath, 'meta.json')

  const metaFile = Bun.file(metaPath)
  if (!(await metaFile.exists())) return null

  const meta = await metaFile.json() as MetadataFile

  const title = deriveTitle(meta.chatName, meta.firstUserMessage)
  const timestamp = Date.parse(meta.updated)

  return {
    sessionId: meta.sessionId,
    title,
    lastMessage: meta.lastMessage ?? 'No messages yet',
    timestamp: Number.isNaN(timestamp) ? Date.parse(meta.created) : timestamp,
    messageCount: meta.messageCount,
  }
}


/**
 * Derive title from chat name or first user message
 */
function deriveTitle(chatName: string, _firstUserMessage: string | null): string {
  if (chatName.trim().length > 0 && chatName !== DEFAULT_CHAT_NAME) {
    return chatName
  }
  return DEFAULT_CHAT_NAME
}
