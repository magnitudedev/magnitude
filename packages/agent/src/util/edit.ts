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
  if (oldStr.length === 0) {
    throw new Error('"old" parameter content must not be empty.')
  }

  const occurrences = countOccurrences(content, oldStr)

  if (occurrences === 0) {
    // Virtual boundary matching fallback:
    // Treat the file as having a virtual \n at the start and end.
    // This absorbs formatting newlines the model may place after <old> / before </old>
    // when the matched content is at file boundaries.
    const virtualResult = tryVirtualMatch(content, oldStr, newStr, replaceAll)
    if (virtualResult) return virtualResult
    throw new Error('"old" parameter content not found in file. Ensure it matches the file exactly.')
  }

  if (!replaceAll && occurrences > 1) {
    throw new Error(
      `"old" parameter content matches ${occurrences} locations in the file. ` +
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

// =============================================================================
// Virtual boundary matching
// =============================================================================

interface VirtualMatch {
  pos: number
  touchesLeading: boolean
  touchesTrailing: boolean
  realStart: number
  realEnd: number
}

/**
 * Try matching oldStr against a virtual version of the file with \n prepended and appended.
 * This allows the model's formatting newlines (after <old> / before </old>) to be absorbed
 * at file boundaries (SOF/EOF) without affecting matches in the middle of the file.
 *
 * Returns null if no virtual match is found.
 */
function tryVirtualMatch(
  content: string,
  oldStr: string,
  newStr: string,
  replaceAll: boolean,
): AppliedEdit | null {
  const virtualFile = '\n' + content + '\n'
  const virtualLen = virtualFile.length

  // Find all matches in the virtual file
  const matches: VirtualMatch[] = []
  let searchFrom = 0
  while (true) {
    const idx = virtualFile.indexOf(oldStr, searchFrom)
    if (idx === -1) break

    const touchesLeading = idx === 0
    const touchesTrailing = idx + oldStr.length === virtualLen
    const realStart = Math.max(idx - 1, 0)
    const realEnd = Math.min(idx - 1 + oldStr.length, content.length)

    // Only accept matches with non-zero real region
    if (realEnd > realStart) {
      matches.push({ pos: idx, touchesLeading, touchesTrailing, realStart, realEnd })
    }

    searchFrom = idx + oldStr.length
  }

  if (matches.length === 0) return null

  if (matches.length > 1 && !replaceAll) {
    throw new Error(
      `"old" parameter content matches ${matches.length} locations in the file. ` +
      `Include more surrounding context to make the match unique, or use replaceAll="true".`,
    )
  }

  const toApply = replaceAll ? matches : [matches[0]]

  // Apply replacements in reverse order to preserve positions
  let result = content
  for (const match of [...toApply].reverse()) {
    // Clip newString based on which virtual boundaries were consumed
    let adjusted = newStr
    if (match.touchesLeading && adjusted.startsWith('\n')) {
      adjusted = adjusted.slice(1)
    }
    if (match.touchesTrailing && adjusted.endsWith('\n')) {
      adjusted = adjusted.slice(0, -1)
    }

    result = result.slice(0, match.realStart) + adjusted + result.slice(match.realEnd)
  }

  const firstMatch = toApply[0]
  return {
    result,
    startLine: lineNumberAt(content, firstMatch.realStart),
    removedLines: oldStr.split('\n'),
    addedLines: newStr.split('\n'),
    replaceCount: toApply.length,
  }
}

/**
 * Convert an AppliedEdit to an EditDiff for UI display.
 */
export function toEditDiff(
  applied: AppliedEdit,
  postEditContent: string,
  contextRadius = 5,
): EditDiff {
  const postLines = postEditContent.split('\n')
  const startIdx = Math.max(0, applied.startLine - 1)
  const beforeStart = Math.max(0, startIdx - contextRadius)
  const contextBefore = postLines.slice(beforeStart, startIdx)
  const afterStart = startIdx + applied.addedLines.length
  const contextAfter = postLines.slice(afterStart, afterStart + contextRadius)

  return {
    startLine: applied.startLine,
    removedLines: applied.removedLines,
    addedLines: applied.addedLines,
    contextBefore,
    contextAfter,
  }
}
