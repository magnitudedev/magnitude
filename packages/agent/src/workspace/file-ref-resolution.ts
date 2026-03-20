import path from 'node:path'
import { existsSync } from 'node:fs'
import { normalizeReferencedPath } from './file-refs'

export interface ResolvedFileRef {
  resolvedPath: string
  displayPath: string
  sourceRoot: 'workspace' | 'project'
}

export function resolveFileRef(
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
    if (!existsSync(resolvedPath)) return null
    return {
      resolvedPath,
      displayPath: innerPath,
      sourceRoot: 'workspace',
    }
  }

  const projectResolved = path.resolve(cwd, normalized)
  if (!existsSync(projectResolved)) return null

  return {
    resolvedPath: projectResolved,
    displayPath: normalized,
    sourceRoot: 'project',
  }
}
