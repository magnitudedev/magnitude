/**
 * Directory Walking
 * 
 * Parallel directory traversal with gitignore support.
 */

import { readdir, stat } from 'fs/promises'
import { join, relative } from 'path'
import ignore, { type Ignore } from 'ignore'
import { ALWAYS_EXCLUDED, parseGitignore } from './gitignore'

// =============================================================================
// Types
// =============================================================================

export interface Entry {
  /** Absolute path */
  fullPath: string
  /** Relative path from base */
  relativePath: string
  /** Entry name */
  name: string
  /** Entry type */
  type: 'file' | 'dir'
  /** Depth in tree */
  depth: number
  /** File size in bytes (files only) */
  size?: number
  /** File mtime in milliseconds (files only) */
  mtimeMs?: number
}

// =============================================================================
// Walk Function (Always Parallel)
// =============================================================================

export interface WalkOptions {
  respectGitignore?: boolean
  collectSizes?: boolean
  collectMtimes?: boolean
}

/**
 * Walk directory tree in parallel.
 * All subdirectories are processed concurrently.
 */
export async function walk(
  dirPath: string,
  basePath: string,
  depth: number,
  maxDepth: number | undefined,
  currentIgnore: Ignore | null,
  respectGitignoreOrOpts: boolean | WalkOptions = true
): Promise<Entry[]> {
  const opts: WalkOptions = typeof respectGitignoreOrOpts === 'boolean'
    ? { respectGitignore: respectGitignoreOrOpts }
    : respectGitignoreOrOpts
  const respectGitignore = opts.respectGitignore ?? true
  const collectSizes = opts.collectSizes ?? false
  const collectMtimes = opts.collectMtimes ?? false

  if (maxDepth !== undefined && depth > maxDepth) {
    return []
  }

  let ignoreFilter: Ignore | null = currentIgnore
  if (respectGitignore) {
    const relativeDirPath = relative(basePath, dirPath)
    const patterns = await parseGitignore(dirPath, relativeDirPath)

    if (patterns.length > 0) {
      ignoreFilter = currentIgnore
        ? ignore().add(currentIgnore).add(patterns)
        : ignore().add(patterns)
    }
  }

  let items
  try {
    items = await readdir(dirPath, { withFileTypes: true })
  } catch {
    return [] // Skip unreadable directories (EPERM, EACCES, etc.)
  }
  const entries: Entry[] = []
  const subdirs: string[] = []
  const fileStatPromises: Promise<void>[] = []

  for (const item of items) {
    const fullPath = join(dirPath, item.name)
    const relativePath = relative(basePath, fullPath)

    if (ALWAYS_EXCLUDED.has(item.name)) continue
    if (ignoreFilter && ignoreFilter.ignores(relativePath)) continue

    const entry: Entry = {
      fullPath,
      relativePath,
      name: item.name,
      type: item.isDirectory() ? 'dir' : 'file',
      depth
    }

    entries.push(entry)

    if (item.isDirectory()) {
      subdirs.push(fullPath)
    } else if (collectSizes || collectMtimes) {
      fileStatPromises.push(
        stat(fullPath).then(s => {
          if (collectSizes) entry.size = s.size
          if (collectMtimes) entry.mtimeMs = s.mtimeMs
        }).catch(() => {})
      )
    }
  }

  // Stat files + recurse subdirectories in parallel
  const [, ...subResults] = await Promise.all([
    Promise.all(fileStatPromises),
    ...subdirs.map(subdir => walk(subdir, basePath, depth + 1, maxDepth, ignoreFilter, opts))
  ]) as [void[], ...Entry[][]]

  for (const subEntries of subResults) {
    entries.push(...subEntries)
  }

  return entries
}
