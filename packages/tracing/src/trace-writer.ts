/**
 * Trace Writer — persists LLM call traces to ~/.magnitude/traces/<session-id>/
 *
 * Call initTraceSession() once at startup, then pass writeTrace to onTrace().
 */

import {
  appendTracesSync,
  defaultGlobalStorageRoot,
  getTraceDir,
  initTraceSessionSync,
  makeGlobalStoragePaths,
  updateTraceMetaSync,
} from '@magnitudedev/storage'
import { logger } from '@magnitudedev/logger'
import type { AgentCallTrace, TraceSessionMeta } from './types'

const globalPaths = makeGlobalStoragePaths(defaultGlobalStorageRoot())

let currentSessionId: string | null = null

export function initTraceSession(
  sessionId: string,
  context: { cwd: string | null; platform: string | null; gitBranch: string | null }
): void {
  currentSessionId = sessionId

  const meta: TraceSessionMeta = {
    sessionId,
    created: new Date().toISOString(),
    cwd: context.cwd,
    platform: context.platform,
    gitBranch: context.gitBranch,
    chatName: null,
  }

  try {
    initTraceSessionSync(globalPaths, sessionId, meta)
    logger.info(
      { sessionId, dir: getTraceDir(globalPaths, sessionId) },
      '[Tracing] Session initialized'
    )
  } catch (e) {
    logger.error({ error: e, sessionId }, '[Tracing] Failed to initialize session')
  }
}

export function writeTrace(trace: AgentCallTrace): void {
  if (!currentSessionId) {
    logger.warn('[Tracing] writeTrace called before initTraceSession')
    return
  }
  try {
    appendTracesSync(globalPaths, currentSessionId, [trace])
  } catch (e) {
    logger.error({ error: e }, '[Tracing] Failed to write trace')
  }
}

export function updateTraceMeta(updates: Partial<TraceSessionMeta>): void {
  if (!currentSessionId) return

  try {
    const sessionId = currentSessionId
    updateTraceMetaSync(globalPaths, sessionId, (existing) => ({
      sessionId,
      created: existing?.created ?? new Date().toISOString(),
      cwd: existing?.cwd ?? null,
      platform: existing?.platform ?? null,
      gitBranch: existing?.gitBranch ?? null,
      chatName: existing?.chatName ?? null,
      ...updates,
    }))
  } catch (e) {
    logger.error({ error: e }, '[Tracing] Failed to update meta')
  }
}

export function getTraceSessionId(): string | null {
  return currentSessionId
}
