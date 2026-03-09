/**
 * String find-replace edit algorithm.
 *
 * Each edit finds an exact substring in the file content and replaces it.
 */

import type { EditDiff } from './line-edit'

// =============================================================================
// Types
// =============================================================================

export interface AppliedEdit {
  result: string
  startLine: number
  removedLines: string[]
  addedLines: string[]
  replaceCount: number
}

// =============================================================================
// Helpers
// =============================================================================

/**
 * Count non-overlapping occurrences of a substring.
 */
function countOccurrences(haystack: string, needle: string): number {
  let count = 0
  let pos = 0
  while ((pos = haystack.indexOf(needle, pos)) !== -1) {
    count++
    pos += needle.length
  }
  return count
}

/**
 * Find the 1-based line number where a character index falls.
 */
function lineNumberAt(content: string, charIndex: number): number {
  let line = 1
  for (let i = 0; i < charIndex; i++) {
    if (content[i] === '\n') line++
  }
  return line
}

// =============================================================================
// Core algorithm
// =============================================================================

/**
 * Validate and apply a find-replace edit.
 *
 * - Finds oldStr in the content
 * - If not replaceAll: verifies exactly one occurrence
 * - Returns the new content + diff info for UI display
 */
export function validateAndApply(
  content: string,
  oldStr: string,
  newStr: string,
  replaceAll: boolean,
): AppliedEdit {
  const occurrences = countOccurrences(content, oldStr)

  if (occurrences === 0) {
    throw new Error('<old> content not found in file. Ensure it matches the file exactly.')
  }

  if (!replaceAll && occurrences > 1) {
    throw new Error(
      `<old> content matches ${occurrences} locations in the file. ` +
      `Include more surrounding context to make the match unique, or use replaceAll="true".`
    )
  }

  const removedLines = oldStr.split('\n')
  const addedLines = newStr.split('\n')
  const firstIdx = content.indexOf(oldStr)
  const startLine = lineNumberAt(content, firstIdx)

  if (replaceAll) {
    const result = content.split(oldStr).join(newStr)
    return { result, startLine, removedLines, addedLines, replaceCount: occurrences }
  }

  const result = content.slice(0, firstIdx) + newStr + content.slice(firstIdx + oldStr.length)
  return { result, startLine, removedLines, addedLines, replaceCount: 1 }
}

/**
 * Convert an AppliedEdit to an EditDiff for UI display.
 */
export function toEditDiff(applied: AppliedEdit): EditDiff {
  return {
    startLine: applied.startLine,
    removedLines: applied.removedLines,
    addedLines: applied.addedLines,
  }
}
