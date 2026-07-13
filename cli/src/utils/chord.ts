/**
 * Keyboard chord helpers for error-action CTAs.
 *
 * A "chord" is a string like `"ctrl+s"` representing a Ctrl+letter shortcut.
 * Used by ErrorPresentation.cta to declare keyboard shortcuts the CLI then
 * matches against KeyEvent and renders as `"Ctrl+S"`.
 */

import type { KeyEvent } from '@opentui/core'

/**
 * Convert a KeyEvent to the chord string format ("ctrl+s") if it matches a
 * Ctrl+letter pattern. Returns null otherwise.
 *
 * Excludes events that also have meta/option/shift modifiers — those are
 * different chords we don't currently support.
 */
export function matchKeyToChord(key: KeyEvent): string | null {
  if (!key.ctrl) return null
  if (key.meta || key.option || key.shift) return null
  if (!key.name || key.name.length !== 1) return null
  if (!/^[a-z]$/.test(key.name)) return null
  return `ctrl+${key.name}`
}

/**
 * Render a chord string for display: "ctrl+s" → "Ctrl+S".
 */
export function formatChord(chord: string): string {
  return chord
    .split('+')
    .map((part) => part.length === 1 ? part.toUpperCase() : part.charAt(0).toUpperCase() + part.slice(1))
    .join('+')
}
