import type { InputValue } from '../types/store'
import { readClipboardText } from './clipboard'

// Re-export InputValue type for backwards compatibility
export type { InputValue } from '../types/store'

export const LIST_BULLET_GLYPH = '• '
export const PASTE_ATTACHMENT_CHAR_LIMIT = 2000
/** Max number of lines to show in collapsed previews */
export const PREVIEW_LINE_CAP = 3

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
    })
  }
}

/**
 * Creates a paste handler that supports text paste with optional long-text handling.
 *
 * When eventText is provided (from the terminal's paste event), uses that directly.
 * Only when NO eventText is provided do we fall back to reading from clipboard.
 */
export function createSmartPasteHandler(options: {
  text: string
  cursorPosition: number
  onChange: (value: InputValue) => void
  onPasteLongText?: (text: string) => void
}): (eventText?: string) => void {
  const { text, cursorPosition, onChange, onPasteLongText } = options

  return (eventText) => {
    const pastedText = eventText || readClipboardText()
    if (!pastedText) return

    if (onPasteLongText && pastedText.length > PASTE_ATTACHMENT_CHAR_LIMIT) {
      onPasteLongText(pastedText)
      return
    }

    const { newText, newCursor } = spliceTextAtCursor(text, cursorPosition, pastedText)
    onChange({
      text: newText,
      cursorPosition: newCursor,
      lastEditDueToNav: false,
    })
  }
}