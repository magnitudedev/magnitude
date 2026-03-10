import type { InputPasteSegment, InputValue } from '../types/store'
import { readClipboardText } from './clipboard'

// Re-export InputValue type for backwards compatibility
export type { InputValue } from '../types/store'

export const LIST_BULLET_GLYPH = '• '

/** Max number of lines to show in collapsed previews */
export const PREVIEW_LINE_CAP = 3

export function formatPastePlaceholder(charCount: number): string {
  return `[${charCount.toLocaleString()} characters pasted]`
}

function sortSegments(segments: InputPasteSegment[]): InputPasteSegment[] {
  return [...segments].sort((a, b) => a.start - b.start)
}

export function insertPasteSegment(
  input: InputValue,
  pastedText: string,
  id: string,
): InputValue {
  const placeholder = formatPastePlaceholder(pastedText.length)
  const insertAt = Math.max(0, Math.min(input.cursorPosition, input.text.length))
  const before = input.text.slice(0, insertAt)
  const after = input.text.slice(insertAt)
  const delta = placeholder.length

  const shifted = input.pasteSegments.map((segment) => {
    if (segment.end <= insertAt) return segment
    return {
      ...segment,
      start: segment.start + delta,
      end: segment.end + delta,
    }
  })

  return {
    ...input,
    text: before + placeholder + after,
    cursorPosition: insertAt + delta,
    lastEditDueToNav: false,
    selectedPasteSegmentId: null,
    pasteSegments: sortSegments([
      ...shifted,
      {
        id,
        placeholder,
        content: pastedText,
        start: insertAt,
        end: insertAt + delta,
      },
    ]),
  }
}

export function reconstituteInputText(input: InputValue): string {
  if (input.pasteSegments.length === 0) return input.text

  const segments = sortSegments(input.pasteSegments)
  let cursor = 0
  let output = ''

  for (const segment of segments) {
    if (segment.start > cursor) {
      output += input.text.slice(cursor, segment.start)
    }
    output += segment.content
    cursor = segment.end
  }

  if (cursor < input.text.length) {
    output += input.text.slice(cursor)
  }

  return output
}

export function applyTextEditWithSegments(
  input: InputValue,
  start: number,
  end: number,
  insertedText: string,
): InputValue {
  const safeStart = Math.max(0, Math.min(start, input.text.length))
  const safeEnd = Math.max(safeStart, Math.min(end, input.text.length))

  let effectiveStart = safeStart
  let effectiveEnd = safeEnd
  for (const segment of input.pasteSegments) {
    const overlaps = !(segment.end <= safeStart || segment.start >= safeEnd)
    if (!overlaps) continue
    effectiveStart = Math.min(effectiveStart, segment.start)
    effectiveEnd = Math.max(effectiveEnd, segment.end)
  }

  const nextText =
    input.text.slice(0, effectiveStart) +
    insertedText +
    input.text.slice(effectiveEnd)

  const removed = effectiveEnd - effectiveStart
  const inserted = insertedText.length
  const delta = inserted - removed

  const remainingSegments: InputPasteSegment[] = []
  for (const segment of input.pasteSegments) {
    const overlaps = !(segment.end <= effectiveStart || segment.start >= effectiveEnd)
    if (overlaps) {
      continue
    }
    if (segment.end <= effectiveStart) {
      remainingSegments.push(segment)
      continue
    }
    remainingSegments.push({
      ...segment,
      start: segment.start + delta,
      end: segment.end + delta,
    })
  }

  const proposedCursor = effectiveStart + inserted
  let nextCursor = proposedCursor
  for (const segment of remainingSegments) {
    if (nextCursor > segment.start && nextCursor < segment.end) {
      nextCursor = segment.end
      break
    }
  }

  return {
    ...input,
    text: nextText,
    cursorPosition: nextCursor,
    lastEditDueToNav: false,
    pasteSegments: sortSegments(remainingSegments),
    selectedPasteSegmentId: null,
  }
}

/**
 * Truncate a command to a single line for display.
 * Flattens newlines to spaces, collapses whitespace, truncates with '...' if over maxLen.
 */
export function shortenCommandPreview(commandText: string, maxLen: number): string {
  const singleLine = commandText.replace(/\n/g, ' ').replace(/\s+/g, ' ').trim()
  if (singleLine.length > maxLen) {
    return singleLine.slice(0, maxLen - 3) + '...'
  }
  return singleLine
}

/**
 * Find indices where substring characters appear in string (for fuzzy matching).
 */
export function locateSubsequence(
  source: string,
  query: string,
): number[] | null {
  let sourceIndex = 0
  let queryIndex = 0
  const matchedIndices: number[] = []

  while (sourceIndex < source.length && queryIndex < query.length) {
    if (source[sourceIndex] === query[queryIndex]) {
      matchedIndices.push(sourceIndex)
      queryIndex++
    }
    sourceIndex++
  }

  return queryIndex >= query.length ? matchedIndices : null
}

/**
 * Truncate text to a maximum number of lines, adding '...' if truncated.
 * Returns the input unchanged if it's null/undefined/empty.
 */
export function clipTextLines(
  text: string | null | undefined,
  maxVisibleLines: number,
): string | null | undefined {
  if (!text) return text
  const lines = text.split('\n')
  if (lines.length > maxVisibleLines) {
    return lines.slice(0, maxVisibleLines).join('\n').trimEnd() + '...'
  }
  return text
}

/**
 * Insert text at cursor position and return the new text and cursor position.
 */
function spliceTextAtCursor(
  currentText: string,
  caretIndex: number,
  insertion: string,
): { newText: string; newCursor: number } {
  const beforeCursor = currentText.slice(0, caretIndex)
  const afterCursor = currentText.slice(caretIndex)
  return {
    newText: beforeCursor + insertion + afterCursor,
    newCursor: beforeCursor.length + insertion.length,
  }
}

/**
 * Format a timestamp as HH:MM (no seconds)
 */
export function formatShortTimestamp(ts: number): string {
  const date = new Date(ts)
  const minutes = date.getMinutes().toString().padStart(2, '0')
  const hours = date.getHours().toString().padStart(2, '0')
  return `${hours}:${minutes}`
}

/**
 * Format a timestamp as HH:MM:SS
 */
export function formatFullTimestamp(ts: number): string {
  const date = new Date(ts)
  const seconds = date.getSeconds().toString().padStart(2, '0')
  const minutes = date.getMinutes().toString().padStart(2, '0')
  const hours = date.getHours().toString().padStart(2, '0')
  return `${hours}:${minutes}:${seconds}`
}

/**
 * Creates a paste handler for text-only inputs (feedback, ask-user, etc.).
 * Reads from clipboard with event text fallback, then inserts at cursor.
 */
export function createTextInputPasteHandler(
  text: string,
  cursorPosition: number,
  onChange: (value: InputValue) => void,
): (eventText?: string) => void {
  return (eventText) => {
    const pasteText = eventText || readClipboardText()
    if (!pasteText) return
    const { newText, newCursor } = spliceTextAtCursor(
      text,
      cursorPosition,
      pasteText,
    )
    onChange({
      text: newText,
      cursorPosition: newCursor,
      lastEditDueToNav: false,
      pasteSegments: [],
      selectedPasteSegmentId: null,
    })
  }
}

