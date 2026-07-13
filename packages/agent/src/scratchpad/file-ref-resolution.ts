import path from 'node:path'
import { existsSync } from 'node:fs'
import { normalizeReferencedPath } from './file-refs'
import { expandScratchpadPath } from '@magnitudedev/scratchpad'

export interface ResolvedFileRef {
  resolvedPath: string
  displayPath: string
}

/**
 * Pure path resolution — resolves a file reference to an absolute path
 * without checking existence on disk.
 */
export function resolveFileRefPath(
  refPath: string,
  cwd: string,
  scratchpadPath: string,
): ResolvedFileRef | null {
  const normalized = normalizeReferencedPath(refPath)
  if (!normalized) return null

  const result = expandScratchpadPath(normalized, scratchpadPath)

  if (result.expanded) {
    return {
      resolvedPath: result.path,
      displayPath: result.displayPath,
    }
  }

  // Non-scratchpad path — resolve relative to cwd
  const projectResolved = path.resolve(cwd, normalized)
  return {
    resolvedPath: projectResolved,
    displayPath: normalized,
  }
}

/**
 * Resolves a file reference and verifies existence on disk.
 * Use resolveFileRefPath for pure path resolution without disk checks.
 */
export function resolveFileRef(
  refPath: string,
  cwd: string,
  scratchpadPath: string,
): ResolvedFileRef | null {
  const resolved = resolveFileRefPath(refPath, cwd, scratchpadPath)
  if (!resolved) return null
  if (!existsSync(resolved.resolvedPath)) return null
  return resolved
}
