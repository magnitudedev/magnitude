/**
 * OpenAI Patch format — Codex-style custom patch syntax.
 * LLM sees raw file, returns *** Begin Patch / *** Update File / @@ hunks / *** End Patch.
 */
import type { EditFormat } from './types'

interface DiffHunk {
  anchor: string
  changes: { type: '+' | '-' | ' '; line: string }[]
}

function parseOpenAIPatch(response: string): { filename: string; hunks: DiffHunk[] }[] {
  let text = response.trim()
  // Strip markdown fences
  const fenceMatch = text.match(/```(?:\w*)\n([\s\S]*?)```/)
  if (fenceMatch) text = fenceMatch[1].trim()

  // Find patch envelope
  const beginIdx = text.indexOf('*** Begin Patch')
  const endIdx = text.indexOf('*** End Patch')
  if (beginIdx === -1) throw new Error('No *** Begin Patch found in response')
  
  const patchBody = endIdx !== -1 
    ? text.slice(beginIdx + '*** Begin Patch'.length, endIdx)
    : text.slice(beginIdx + '*** Begin Patch'.length)

  const lines = patchBody.split('\n')
  const files: { filename: string; hunks: DiffHunk[] }[] = []
  let currentFile: { filename: string; hunks: DiffHunk[] } | null = null
  let currentHunk: DiffHunk | null = null

  for (const line of lines) {
    if (line.startsWith('*** Update File: ')) {
      if (currentHunk && currentFile) currentFile.hunks.push(currentHunk)
      if (currentFile) files.push(currentFile)
      currentFile = { filename: line.slice('*** Update File: '.length).trim(), hunks: [] }
      currentHunk = null
      continue
    }
    if (line.startsWith('*** Add File: ')) {
      if (currentHunk && currentFile) currentFile.hunks.push(currentHunk)
      if (currentFile) files.push(currentFile)
      currentFile = { filename: line.slice('*** Add File: '.length).trim(), hunks: [] }
      currentHunk = null
      continue
    }
    if (line.startsWith('*** Delete File: ') || line.startsWith('*** End Patch')) {
      if (currentHunk && currentFile) currentFile.hunks.push(currentHunk)
      if (currentFile) files.push(currentFile)
      currentFile = null
      currentHunk = null
      continue
    }

    if (!currentFile) continue

    if (line.startsWith('@@')) {
      if (currentHunk) currentFile.hunks.push(currentHunk)
      const anchorMatch = line.match(/^@@\s*(.*?)(?:\s*@@)?\s*$/)
      currentHunk = {
        anchor: anchorMatch?.[1]?.trim() ?? '',
        changes: [],
      }
      continue
    }

    if (!currentHunk) continue

    if (line.startsWith('+')) {
      currentHunk.changes.push({ type: '+', line: line.slice(1) })
    } else if (line.startsWith('-')) {
      currentHunk.changes.push({ type: '-', line: line.slice(1) })
    } else if (line.startsWith(' ')) {
      currentHunk.changes.push({ type: ' ', line: line.slice(1) })
    }
  }

  if (currentHunk && currentFile) currentFile.hunks.push(currentHunk)
  if (currentFile) files.push(currentFile)

  return files
}

function findHunkLocation(fileLines: string[], hunk: DiffHunk): number {
  const oldSide: string[] = []
  for (const c of hunk.changes) {
    if (c.type === ' ' || c.type === '-') {
      oldSide.push(c.line)
    }
  }

  if (oldSide.length === 0) {
    throw new Error('Hunk has no context or removed lines to match against')
  }

  // Exact match
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

  // Trimmed match fallback
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

export const openaiPatchFormat: EditFormat = {
  id: 'openai-patch',

  formatFile(_filename: string, content: string): string {
    return content
  },

  systemInstructions(): string {
    return [
      '## Edit format: OpenAI patch',
      '',
      'Return a patch using this format:',
      '',
      '```',
      '*** Begin Patch',
      '*** Update File: <filename>',
      '@@ <context hint> @@',
      ' unchanged line',
      '-removed line',
      '+added line',
      ' unchanged line',
      '*** End Patch',
      '```',
      '',
      'Rules:',
      '- Wrap the entire patch in `*** Begin Patch` and `*** End Patch`',
      '- Use `*** Update File: <filename>` before each file\'s hunks',
      '- Each hunk starts with `@@ <context hint> @@` where the hint is a recognizable line from the file',
      '- Lines starting with ` ` (space) are context (unchanged)',
      '- Lines starting with `-` are removed',
      '- Lines starting with `+` are added',
      '- Include enough context lines to uniquely identify where the change goes',
      '',
      'Return ONLY the patch. No explanation or commentary.',
    ].join('\n')
  },

  applyResponse(response: string, originalContent: string): string {
    const files = parseOpenAIPatch(response)
    if (files.length === 0) {
      throw new Error('No file sections found in patch')
    }

    // Use the first file section (single-file eval)
    const fileSection = files[0]
    if (fileSection.hunks.length === 0) {
      throw new Error('No hunks found in patch')
    }

    const fileLines = originalContent.split('\n')

    const located = fileSection.hunks.map(hunk => ({
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