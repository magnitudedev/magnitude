/**
 * Gitignore Utilities
 * 
 * Clean, single-responsibility functions for gitignore handling.
 */

import { readFile } from 'fs/promises'
import { join } from 'path'
import ignore, { type Ignore } from 'ignore'

// =============================================================================
// Constants
// =============================================================================

export const ALWAYS_EXCLUDED = new Set(['.git'])

export const DEFAULT_IGNORE_PATTERNS = [
  'node_modules',
  'dist',
  'build',
  'out',
  '__pycache__',
  '.pytest_cache',
  '.next',
  '.nuxt',
  'coverage',
  '.turbo',
  '.cache',
  'target',
  'vendor'
]

// =============================================================================
// Pattern Rebasing
// =============================================================================

export function rebasePattern(pattern: string, relativeDirPath: string): string {
  const isNegated = pattern.startsWith('!')
  let p = isNegated ? pattern.slice(1) : pattern

  const dirOnly = p.endsWith('/')
  const core = dirOnly ? p.slice(0, -1) : p

  const anchored = core.startsWith('/')
  const coreNoLead = anchored ? core.slice(1) : core
  const hasSlash = coreNoLead.includes('/')

  const base = relativeDirPath.replace(/\\/g, '/')

  let rebased: string
  if (anchored) {
    rebased = base ? `${base}/${coreNoLead}` : coreNoLead
  } else if (!hasSlash) {
    rebased = base ? `${base}/**/${coreNoLead}` : coreNoLead
  } else {
    rebased = base ? `${base}/${coreNoLead}` : coreNoLead
  }

  if (dirOnly && !rebased.endsWith('/')) {
    rebased += '/'
  }

  return isNegated ? `!${rebased}` : rebased
}

// =============================================================================
// Gitignore Parsing
// =============================================================================

export async function parseGitignore(dir: string, relativeDirPath: string): Promise<string[]> {
  try {
    const content = await readFile(join(dir, '.gitignore'), 'utf8')
    const patterns: string[] = []

    for (let line of content.split('\n')) {
      line = line.trim()
      if (line === '' || line.startsWith('#')) continue
      patterns.push(rebasePattern(line, relativeDirPath))
    }

    return patterns
  } catch {
    return []
  }
}

export function createDefaultIgnore(): Ignore {
  return ignore().add(DEFAULT_IGNORE_PATTERNS)
}
