import path from 'node:path'
import { existsSync } from 'node:fs'
import { normalizeReferencedPath } from './file-refs'

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
  workspacePath: string,
): ResolvedFileRef | null {
  const normalized = normalizeReferencedPath(refPath)
  if (!normalized) return null

  const workspacePrefix = normalized.startsWith('$M/') || normalized.startsWith('${M}/')
  if (workspacePrefix) {
    const innerPath = normalized.startsWith('${M}/')
      ? normalized.slice('${M}/'.length)
      : normalized.slice('$M/'.length)
    const resolvedPath = path.resolve(workspacePath, innerPath)
    return {
      resolvedPath,
      displayPath: innerPath,
    }
  }

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
  workspacePath: string,
): ResolvedFileRef | null {
  const resolved = resolveFileRefPath(refPath, cwd, workspacePath)
  if (!resolved) return null
  if (!existsSync(resolved.resolvedPath)) return null
  return resolved
}
