import type { KeyEvent } from '@opentui/core'

export const INPUT_CURSOR_CHAR = '▍'
export const CONTROL_CHAR_REGEX = /[\u0000-\u0008\u000b-\u000c\u000e-\u001f\u007f]/
export const TAB_WIDTH = 4

export function hasAltStyleModifier(key: KeyEvent): boolean {
  const ESC = '\x1b'
  return Boolean(
    key.option ||
      (key.sequence?.length === 2 &&
        key.sequence[0] === ESC &&
        key.sequence[1] !== '['),
  )
}

/**
 * Check if a key event represents printable character input (not a special key).
 * Uses a positive heuristic based on key.name length rather than a brittle deny-list.
 *
 * The key insight is that OpenTUI's parser assigns descriptive multi-character names
 * to special keys (like 'backspace', 'up', 'f1') while regular printable characters
 * either have no name (multi-byte input like Chinese) or a single-character name.
 */
export function isLikelyPrintableKey(key: KeyEvent): boolean {
  const name = key.name
  if (!name) return true
  if (name.length === 1) return true
  if (name === 'space') return true
  return false
}

export function locateLineStart(text: string, cursor: number): number {
  let pos = Math.max(0, Math.min(cursor, text.length))
  while (pos > 0 && text[pos - 1] !== '\n') {
    pos--
  }
  return pos
}

export function locateLineEnd(text: string, cursor: number): number {
  let pos = Math.max(0, Math.min(cursor, text.length))
  while (pos < text.length && text[pos] !== '\n') {
    pos++
  }
  return pos
}

export function findWordStartBefore(text: string, cursor: number): number {
  let pos = Math.max(0, Math.min(cursor, text.length))

  while (pos > 0 && /\s/.test(text[pos - 1])) {
    pos--
  }
  while (pos > 0 && !/\s/.test(text[pos - 1])) {
    pos--
  }

  return pos
}

export function findWordEndAfter(text: string, cursor: number): number {
  let pos = Math.max(0, Math.min(cursor, text.length))

  while (pos < text.length && !/\s/.test(text[pos])) {
    pos++
  }
  while (pos < text.length && /\s/.test(text[pos])) {
    pos++
  }

  return pos
}

export function stepCursorVertical(params: {
  cursorPosition: number
  lineStarts: number[]
  cursorIsChar: boolean
  direction: 'up' | 'down'
  targetColumn: number
}): number {
  if (params.direction === 'down') {
    return stepCursorDown(params)
  }
  if (params.direction === 'up') {
    return stepCursorUp(params)
  }
  params.direction satisfies never
  throw new Error(`Invalid direction: ${params.direction}`)
}

export function stepCursorUp(params: {
  lineStarts: number[]
  cursorPosition: number
  targetColumn: number
}): number {
  const { lineStarts, cursorPosition, targetColumn } = params
  const lineIndex = lineStarts.findLastIndex((start) => start <= cursorPosition)

  if (lineIndex <= 0) {
    return 0
  }

  const priorLineStart = lineStarts[lineIndex - 1]
  const priorLineEndExclusive = lineStarts[lineIndex] - 1
  return Math.min(priorLineEndExclusive, priorLineStart + targetColumn)
}

export function stepCursorDown(params: {
  lineStarts: number[]
  cursorPosition: number
  targetColumn: number
}): number {
  const { lineStarts, cursorPosition, targetColumn } = params
  const lineIndex = lineStarts.findLastIndex((start) => start <= cursorPosition)

  if (lineIndex === -1 || lineIndex >= lineStarts.length - 1) {
    return Infinity
  }

  return Math.min(
    (lineStarts[lineIndex + 2] ?? Infinity) - 1,
    lineStarts[lineIndex + 1] + targetColumn,
  )
}