/**
 * DOM keyboard event helpers.
 *
 * Provides a generic key-event interface compatible with the
 * `KeyEvent` abstraction used in client-common hooks (spec §4.2 Phase 2).
 */

export interface GenericKeyEvent {
  readonly name: string
  readonly ctrl: boolean
  readonly meta: boolean
  readonly shift: boolean
  readonly option?: boolean
  readonly defaultPrevented?: boolean
}

/**
 * Convert a DOM KeyboardEvent into the generic key-event interface
 * used by shared hooks from client-common.
 */
export function toGenericKeyEvent(e: KeyboardEvent): GenericKeyEvent {
  return {
    name: e.key,
    ctrl: e.ctrlKey,
    meta: e.metaKey,
    shift: e.shiftKey,
    option: e.altKey,
    defaultPrevented: e.defaultPrevented,
  }
}

/**
 * Check if a keyboard event is a send/submit combo (Enter without shift).
 */
export function isSendKey(e: KeyboardEvent): boolean {
  return e.key === "Enter" && !e.shiftKey
}

/**
 * Check if a keyboard event is a newline combo (Shift+Enter).
 */
export function isNewlineKey(e: KeyboardEvent): boolean {
  return e.key === "Enter" && e.shiftKey
}

/**
 * Check if a keyboard event is Escape.
 */
export function isEscapeKey(e: KeyboardEvent): boolean {
  return e.key === "Escape"
}

/**
 * Check if the platform uses Cmd (macOS) vs Ctrl (other).
 */
export function isModKey(e: KeyboardEvent): boolean {
  return e.metaKey || e.ctrlKey
}
