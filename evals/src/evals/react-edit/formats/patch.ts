/**
 * Patch format — unified diff hunks.
 * LLM sees raw file, returns diff with @@ headers, context, and +/- lines.
 */
import type { EditFormat } from './types'

interface DiffHunk {
  anchor: string
  contextLines: string[]
  changes: { type: '+' | '-' | ' '; line: string }[]
}

function parseHunks(diff: string): DiffHunk[] {
  const lines = diff.split('\n')
  const hunks: DiffHunk[] = []
  let current: DiffHunk | null = null

  for (const line of lines) {
    if (line.startsWith('@@')) {
      if (current) hunks.push(current)
      // Extract anchor: @@ anchor @@ or just @@
      const anchorMatch = line.match(/^@@\s*(.*?)(?:\s*@@)?\s*$/)
      current = {
        anchor: anchorMatch?.[1]?.trim() ?? '',
        contextLines: [],
        changes: [],
      }
      continue
    }

    if (!current) continue

    if (line.startsWith('+')) {
      current.changes.push({ type: '+', line: line.slice(1) })
    } else if (line.startsWith('-')) {
      current.changes.push({ type: '-', line: line.slice(1) })
    } else if (line.startsWith(' ')) {
      current.changes.push({ type: ' ', line: line.slice(1) })
    }
  }

  if (current) hunks.push(current)
  return hunks
}

function findHunkLocation(fileLines: string[], hunk: DiffHunk): number {
  // Build the "old" side of the hunk (context + removed lines)
  const oldSide: string[] = []
  for (const c of hunk.changes) {
    if (c.type === ' ' || c.type === '-') {
      oldSide.push(c.line)
    }
  }

  if (oldSide.length === 0) {
    throw new Error('Hunk has no context or removed lines to match against')
  }

  // Try exact match first
  const matches: number[] = []
  for (let i = 0; i <= fileLines.length - oldSide.length; i++) {
    let match = true
    for (let j = 0; j < oldSide.length; j++) {
      if (fileLines[i + j] !== oldSide[j]) {
        match = false
        break
      }
    }
    if (match) matches.push(i)
  }

  if (matches.length === 1) return matches[0]

  // Try trimmed match
  const trimMatches: number[] = []
  for (let i = 0; i <= fileLines.length - oldSide.length; i++) {
    let match = true
    for (let j = 0; j < oldSide.length; j++) {
      if (fileLines[i + j].trim() !== oldSide[j].trim()) {
        match = false
        break
      }
    }
    if (match) trimMatches.push(i)
  }

  if (trimMatches.length === 1) return trimMatches[0]
  if (trimMatches.length > 1) {
    throw new Error(`Hunk context matches ${trimMatches.length} locations (ambiguous)`)
  }

  throw new Error('Hunk context not found in file')
}

function extractDiff(response: string): string {
  let text = response.trim()
  // Strip markdown code fences
  const fenceMatch = text.match(/```(?:diff|patch)?\n([\s\S]*?)```/)
  if (fenceMatch) text = fenceMatch[1].trim()
  return text
}

export const patchFormat: EditFormat = {
  id: 'patch',

  formatFile(_filename: string, content: string): string {
    return content
  },

  systemInstructions(): string {
    return [
      '## Edit format: unified diff patch',
      '',
      'Return a diff with one or more hunks. Each hunk starts with `@@` optionally followed by an anchor (a unique line copied from the file).',
      '',
      'Lines starting with ` ` (space) are context (unchanged).',
      'Lines starting with `-` are removed.',
      'Lines starting with `+` are added.',
      '',
      'Include enough context lines to uniquely identify where the change goes.',
      '',
      'Example:',
      '```',
      '@@ const timeout',
      ' const port = 3000;',
      '-const timeout = 5000;',
      '+const timeout = 30_000;',
      ' const host = "localhost";',
      '```',
      '',
      'Return ONLY the diff. No explanation or commentary.',
    ].join('\n')
  },

  applyResponse(response: string, originalContent: string): string {
    const diff = extractDiff(response)
    const hunks = parseHunks(diff)

    if (hunks.length === 0) {
      throw new Error('No diff hunks found in response')
    }

    const fileLines = originalContent.split('\n')

    // Apply hunks bottom-up to preserve line numbers
    const located = hunks.map(hunk => ({
      hunk,
      startIndex: findHunkLocation(fileLines, hunk),
    }))
    located.sort((a, b) => b.startIndex - a.startIndex)

    for (const { hunk, startIndex } of located) {
      const oldSide: string[] = []
      const newSide: string[] = []
      for (const c of hunk.changes) {
        if (c.type === ' ' || c.type === '-') oldSide.push(c.line)
        if (c.type === ' ' || c.type === '+') newSide.push(c.line)
      }
      fileLines.splice(startIndex, oldSide.length, ...newSide)
    }

    return fileLines.join('\n')
  },
}
