/**
 * Result Persistence — Persist tool results to files for retroactive disclosure.
 * 
 * Every tool result is written as JSON to {resultsDir}/{turnId}-{callId}.json
 * for later access via file read operations.
 * 
 * resultsDir is always passed explicitly — no env var fallbacks.
 */

import * as fs from 'fs'
import * as path from 'path'

function ensureDir(dir: string): void {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true })
  }
}

/**
 * Get the deterministic file path for a tool result.
 */
export function getResultPath(resultsDir: string, turnId: string, callId: string): string {
  ensureDir(resultsDir)
  return path.join(resultsDir, `${turnId}-${callId}.json`)
}

/**
 * Persist a tool result to a JSON file.
 * Returns the path to the persisted file.
 */
export function persistResult(result: unknown, turnId: string, callId: string, resultsDir: string): string {
  const resultPath = getResultPath(resultsDir, turnId, callId)
  const json = JSON.stringify(result, null, 2)
  fs.writeFileSync(resultPath, json, 'utf-8')
  return resultPath
}

/**
 * Load a previously persisted result.
 */
export function loadResult(resultsDir: string, turnId: string, callId: string): unknown {
  const resultPath = getResultPath(resultsDir, turnId, callId)
  if (!fs.existsSync(resultPath)) {
    throw new Error(`Result not found: ${resultPath}`)
  }
  const json = fs.readFileSync(resultPath, 'utf-8')
  return JSON.parse(json)
}

/**
 * Load a result from a specific path.
 */
export function loadResultFromPath(resultPath: string): unknown {
  if (!fs.existsSync(resultPath)) {
    throw new Error(`Result not found: ${resultPath}`)
  }
  const json = fs.readFileSync(resultPath, 'utf-8')
  return JSON.parse(json)
}

/**
 * Check if a result exists for the given turn/call IDs.
 */
export function hasResult(resultsDir: string, turnId: string, callId: string): boolean {
  return fs.existsSync(getResultPath(resultsDir, turnId, callId))
}

/**
 * Delete a persisted result.
 */
export function deleteResult(resultsDir: string, turnId: string, callId: string): void {
  const resultPath = getResultPath(resultsDir, turnId, callId)
  if (fs.existsSync(resultPath)) {
    fs.unlinkSync(resultPath)
  }
}

/**
 * List all persisted results in the results directory.
 */
export function listResults(resultsDir: string): Array<{ turnId: string; callId: string; path: string }> {
  if (!fs.existsSync(resultsDir)) return []
  
  const files = fs.readdirSync(resultsDir)
  const results: Array<{ turnId: string; callId: string; path: string }> = []
  
  for (const file of files) {
    if (!file.endsWith('.json')) continue
    const match = file.match(/^(.+)-(.+)\.json$/)
    if (match) {
      results.push({
        turnId: match[1],
        callId: match[2],
        path: path.join(resultsDir, file)
      })
    }
  }
  
  return results
}

/**
 * Clean up old results.
 * Removes results older than the specified age in milliseconds.
 */
export function cleanupResults(resultsDir: string, maxAgeMs: number): number {
  if (!fs.existsSync(resultsDir)) return 0
  
  const now = Date.now()
  let deleted = 0
  
  const files = fs.readdirSync(resultsDir)
  for (const file of files) {
    if (!file.endsWith('.json')) continue
    const filePath = path.join(resultsDir, file)
    const stats = fs.statSync(filePath)
    if (now - stats.mtimeMs > maxAgeMs) {
      fs.unlinkSync(filePath)
      deleted++
    }
  }
  
  return deleted
}
