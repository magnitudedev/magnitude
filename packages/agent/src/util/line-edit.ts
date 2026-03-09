/**
 * Line-edit — plain line-number based file editing.
 *
 * Each line gets a simple line number prefix: "N|content"
 * Edit operations use XML tags: <replace>, <remove>, <insert>
 */

// =============================================================================
// Display formatting
// =============================================================================

/**
 * Format file content with plain line numbers.
 * Each line gets: {lineNumber}|{content}
 *
 * Example output:
 *   1|import express from 'express'
 *   2|import cors from 'cors'
 *   3|
 */
export function formatNumberedLines(content: string): string {
  const lines = content.split('\n')
  return lines
    .map((line, i) => `${i + 1}|${line}`)
    .join('\n')
}

// =============================================================================
// Edit operation types
// =============================================================================

export type ParsedOp =
  | { type: 'replace'; from: number; to: number; content: string }
  | { type: 'remove'; from: number; to: number }
  | { type: 'insert'; after: number; content: string }

/** Diff data for a single edit operation (for UI display) */
export interface EditDiff {
  startLine: number      // 1-based line where edit starts
  removedLines: string[] // original lines that were removed/replaced
  addedLines: string[]   // new lines that replaced them
}

// =============================================================================
// Parsing
// =============================================================================

/**
 * Parse XML edit operations from LLM response.
 *
 * Supports:
 *   <replace from=N to=N>content</replace>
 *   <remove from=N to=N />
 *   <insert after=N>content</insert>
 */
export function parseEditOps(response: string): ParsedOp[] {
  const ops: ParsedOp[] = []
  const tagNames = ['remove', 'replace', 'insert'] as const
  let i = 0

  while (i < response.length) {
    // Look for '<' followed by one of our tag names
    if (response[i] !== '<') { i++; continue }

    let matched: typeof tagNames[number] | null = null
    for (const tag of tagNames) {
      if (response.substring(i + 1, i + 1 + tag.length) === tag) {
        const charAfter = response[i + 1 + tag.length]
        if (charAfter === ' ' || charAfter === '\t' || charAfter === '\n' || charAfter === '\r' || charAfter === '/' || charAfter === '>') {
          matched = tag
          break
        }
      }
    }

    if (!matched) { i++; continue }

    const tagStart = i
    const tagNameLen = matched.length
    // Find end of opening tag (either > or />)
    let j = i + 1 + tagNameLen
    let selfClosing = false
    let openTagEnd = -1

    while (j < response.length) {
      if (response[j] === '/' && j + 1 < response.length && response[j + 1] === '>') {
        selfClosing = true
        openTagEnd = j + 2
        break
      }
      if (response[j] === '>') {
        openTagEnd = j + 1
        break
      }
      // Skip quoted strings in attributes
      if (response[j] === '"') {
        j++
        while (j < response.length && response[j] !== '"') j++
      } else if (response[j] === "'") {
        j++
        while (j < response.length && response[j] !== "'") j++
      }
      j++
    }

    if (openTagEnd === -1) { i++; continue }

    // Extract attribute string
    const attrStr = response.substring(i + 1 + tagNameLen, selfClosing ? openTagEnd - 2 : openTagEnd - 1)
    const attrs = parseAttributes(attrStr)

    if (selfClosing) {
      // Only remove can be self-closing
      if (matched === 'remove' && attrs['from'] != null && attrs['to'] != null) {
        ops.push({ type: 'remove', from: parseInt(attrs['from'], 10), to: parseInt(attrs['to'], 10) })
      }
      i = openTagEnd
      continue
    }

    // Content tag — track nesting depth
    const closeTag = '</' + matched + '>'
    const openPrefix = '<' + matched
    let depth = 1
    let k = openTagEnd

    while (k < response.length && depth > 0) {
      if (response[k] === '<') {
        // Check close tag
        if (response.substring(k, k + closeTag.length) === closeTag) {
          depth--
          if (depth === 0) break
          k += closeTag.length
          continue
        }
        // Check open tag (same name)
        if (response.substring(k + 1, k + 1 + matched.length) === matched) {
          const ca = response[k + 1 + matched.length]
          if (ca === ' ' || ca === '\t' || ca === '\n' || ca === '\r' || ca === '/' || ca === '>') {
            depth++
          }
        }
      }
      k++
    }

    if (depth !== 0) { i++; continue }

    let content = response.substring(openTagEnd, k)
    if (content.startsWith('\n')) content = content.slice(1)
    if (content.endsWith('\n')) content = content.slice(0, -1)

    if (matched === 'replace' && attrs['from'] != null && attrs['to'] != null) {
      ops.push({ type: 'replace', from: parseInt(attrs['from'], 10), to: parseInt(attrs['to'], 10), content })
    } else if (matched === 'insert' && attrs['after'] != null) {
      ops.push({ type: 'insert', after: parseInt(attrs['after'], 10), content })
    }

    i = k + closeTag.length
  }

  return ops
}

function parseAttributes(attrStr: string): Record<string, string> {
  const result: Record<string, string> = {}
  const re = /(\w+)\s*=\s*(?:"([^"]*)"|'([^']*)'|(\S+))/g
  let m: RegExpExecArray | null
  while ((m = re.exec(attrStr)) !== null) {
    result[m[1]] = m[2] ?? m[3] ?? m[4]
  }
  return result
}

// =============================================================================
// Apply operations
// =============================================================================

/**
 * Apply parsed edit operations to file content.
 *
 * Operations are applied bottom-up (highest line first) to preserve line numbers.
 * Returns { content, summary, diffs }.
 */
export function applyOps(content: string, ops: ParsedOp[]): { content: string; summary: string[]; diffs: EditDiff[] } {
  const lines = content.split('\n')

  // Validate all ops
  for (const op of ops) {
    if (op.type === 'replace' || op.type === 'remove') {
      if (op.from < 1 || op.from > lines.length) {
        throw new Error(`'from' line ${op.from} is out of range (file has ${lines.length} lines)`)
      }
      if (op.to < op.from) {
        throw new Error(`'to' line ${op.to} is before 'from' line ${op.from}`)
      }
      if (op.to > lines.length) {
        throw new Error(`'to' line ${op.to} is out of range (file has ${lines.length} lines)`)
      }
    } else if (op.type === 'insert') {
      if (op.after < 0 || op.after > lines.length) {
        throw new Error(`'after' line ${op.after} is out of range (file has ${lines.length} lines)`)
      }
    }
  }

  // Sort bottom-up: highest affected line first
  const sorted = [...ops].sort((a, b) => {
    const lineA = a.type === 'insert' ? a.after : a.from
    const lineB = b.type === 'insert' ? b.after : b.from
    return lineB - lineA
  })

  const resultLines = [...lines]
  const summary: string[] = []
  const diffs: EditDiff[] = []

  for (const op of sorted) {
    if (op.type === 'replace') {
      const removeCount = op.to - op.from + 1
      const removedLines = resultLines.slice(op.from - 1, op.to)
      const newLines = op.content.split('\n')
      resultLines.splice(op.from - 1, removeCount, ...newLines)
      diffs.push({ startLine: op.from, removedLines, addedLines: newLines })
      summary.push(`replaced lines ${op.from}-${op.to} (${removeCount} lines) with ${newLines.length} line(s)`)
    } else if (op.type === 'remove') {
      const removeCount = op.to - op.from + 1
      const removedLines = resultLines.slice(op.from - 1, op.to)
      resultLines.splice(op.from - 1, removeCount)
      diffs.push({ startLine: op.from, removedLines, addedLines: [] })
      summary.push(`deleted lines ${op.from}-${op.to} (${removeCount} lines)`)
    } else if (op.type === 'insert') {
      const newLines = op.content.split('\n')
      resultLines.splice(op.after, 0, ...newLines)
      diffs.push({ startLine: op.after + 1, removedLines: [], addedLines: newLines })
      summary.push(`inserted ${newLines.length} line(s) after line ${op.after}`)
    }
  }

  return { content: resultLines.join('\n'), summary, diffs }
}
