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
import type { TraceData } from './types'
import type { TraceSessionMeta } from './types'

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

export function writeTrace(trace: TraceData): void {
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

export function updateTraceMeta(updates: Record<string, unknown>): void {
  if (!currentSessionId) return

  try {
    updateTraceMetaSync(globalPaths, currentSessionId, (existing) => ({
      ...(existing ?? {}),
      ...updates,
    }))
  } catch (e) {
    logger.error({ error: e }, '[Tracing] Failed to update meta')
  }
}

export function getTraceSessionId(): string | null {
  return currentSessionId
}