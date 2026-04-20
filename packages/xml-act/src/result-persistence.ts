
/**
 * Result Persistence — Persist tool results to files for retroactive disclosure.
 * 
 * Every tool result is written as JSON to $M/results/{turnId}-{callId}.json
 * for later access via file read operations.
 */

import * as fs from 'fs'
import * as path from 'path'

const DEFAULT_RESULTS_DIR = '.magnitude/results'

/**
 * Get the results directory path.
 * Uses $M environment variable if available, otherwise defaults to ./.magnitude/results
 */
export function getResultsDir(): string {
  const baseDir = process.env.M || process.cwd()
  return path.join(baseDir, 'results')
}

/**
 * Ensure the results directory exists.
 */
export function ensureResultsDir(): string {
  const dir = getResultsDir()
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true })
  }
  return dir
}

/**
 * Get the deterministic file path for a tool result.
 */
export function getResultPath(turnId: string, callId: string): string {
  return path.join(ensureResultsDir(), `${turnId}-${callId}.json`)
}

/**
 * Persist a tool result to a JSON file.
 * Returns the path to the persisted file.
 */
export function persistResult(result: unknown, turnId: string, callId: string): string {
  const resultPath = getResultPath(turnId, callId)
  const json = JSON.stringify(result, null, 2)
  fs.writeFileSync(resultPath, json, 'utf-8')
  return resultPath
}

/**
 * Load a previously persisted result.
 */
export function loadResult(turnId: string, callId: string): unknown {
  const resultPath = getResultPath(turnId, callId)
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
export function hasResult(turnId: string, callId: string): boolean {
  return fs.existsSync(getResultPath(turnId, callId))
}

/**
 * Delete a persisted result.
 */
export function deleteResult(turnId: string, callId: string): void {
  const resultPath = getResultPath(turnId, callId)
  if (fs.existsSync(resultPath)) {
    fs.unlinkSync(resultPath)
  }
}

/**
 * List all persisted results in the results directory.
 */
export function listResults(): Array<{ turnId: string; callId: string; path: string }> {
  const dir = getResultsDir()
  if (!fs.existsSync(dir)) return []
  
  const files = fs.readdirSync(dir)
  const results: Array<{ turnId: string; callId: string; path: string }> = []
  
  for (const file of files) {
    if (!file.endsWith('.json')) continue
    const match = file.match(/^(.+)-(.+)\.json$/)
    if (match) {
      results.push({
        turnId: match[1],
        callId: match[2],
        path: path.join(dir, file)
      })
    }
  }
  
  return results
}

/**
 * Clean up old results (optional utility).
 * Removes results older than the specified age in milliseconds.
 */
export function cleanupResults(maxAgeMs: number): number {
  const dir = getResultsDir()
  if (!fs.existsSync(dir)) return 0
  
  const now = Date.now()
  let deleted = 0
  
  const files = fs.readdirSync(dir)
  for (const file of files) {
    if (!file.endsWith('.json')) continue
    const filePath = path.join(dir, file)
    const stats = fs.statSync(filePath)
    if (now - stats.mtimeMs > maxAgeMs) {
      fs.unlinkSync(filePath)
      deleted++
    }
  }
  
  return deleted
}
