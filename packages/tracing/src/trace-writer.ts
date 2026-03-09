/**
 * Trace Writer — persists LLM call traces to ~/.magnitude/traces/<session-id>/
 *
 * Call initTraceSession() once at startup, then pass writeTrace to onTrace().
 */

import { mkdirSync, appendFileSync, writeFileSync, readFileSync, existsSync } from 'fs'
import { join } from 'path'
import { homedir } from 'os'
import { logger } from '@magnitudedev/logger'
import type { TraceData } from './types'
import type { TraceSessionMeta } from './types'

const TRACES_DIR = join(homedir(), '.magnitude', 'traces')

let currentSessionId: string | null = null
let currentSessionDir: string | null = null
let tracesFilePath: string | null = null

/**
 * Initialize a trace session. Creates the directory and writes meta.json.
 */
export function initTraceSession(
  sessionId: string,
  context: { cwd: string | null; platform: string | null; gitBranch: string | null }
): void {
  currentSessionId = sessionId
  currentSessionDir = join(TRACES_DIR, sessionId)
  tracesFilePath = join(currentSessionDir, 'traces.jsonl')

  mkdirSync(currentSessionDir, { recursive: true })

  const meta: TraceSessionMeta = {
    sessionId,
    created: new Date().toISOString(),
    cwd: context.cwd,
    platform: context.platform,
    gitBranch: context.gitBranch,
  }
  writeFileSync(join(currentSessionDir, 'meta.json'), JSON.stringify(meta, null, 2))
  logger.info({ sessionId, dir: currentSessionDir }, '[Tracing] Session initialized')
}

/**
 * Write a trace directly to disk. Pass this to onTrace().
 */
export function writeTrace(trace: TraceData<any>): void {
  if (!tracesFilePath) {
    logger.warn('[Tracing] writeTrace called before initTraceSession')
    return
  }
  try {
    appendFileSync(tracesFilePath, JSON.stringify(trace) + '\n')
  } catch (e) {
    logger.error({ error: e }, '[Tracing] Failed to write trace')
  }
}

/**
 * Update session meta.json with additional fields (e.g. title).
 */
export function updateTraceMeta(updates: Record<string, unknown>): void {
  if (!currentSessionDir) return
  const metaPath = join(currentSessionDir, 'meta.json')
  try {
    const existing = existsSync(metaPath)
      ? JSON.parse(readFileSync(metaPath, 'utf-8'))
      : {}
    writeFileSync(metaPath, JSON.stringify({ ...existing, ...updates }, null, 2))
  } catch (e) {
    logger.error({ error: e }, '[Tracing] Failed to update meta')
  }
}

export function getTraceSessionId(): string | null {
  return currentSessionId
}
