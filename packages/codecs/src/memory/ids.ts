/**
 * ID constructors for TurnPart identity tracking.
 *
 * All IDs are plain strings — no branded types.
 * These functions are the only producers; consumers treat the value as opaque.
 *
 * Format:
 *   thought  → "thought-<base36-ts>-<counter>"
 *   message  → "msg-<base36-ts>-<counter>"
 *   toolcall → "call-<ord>-<base36-ts>"
 */

let _thoughtCounter = 0
let _messageCounter = 0

function ts36(): string {
  return Date.now().toString(36)
}

/**
 * newThoughtId — unique ID for a ThoughtPart.
 * Counter resets each process; collisions across sessions are impossible
 * because IDs are never persisted as keys.
 */
export function newThoughtId(): string {
  return `thought-${ts36()}-${_thoughtCounter++}`
}

/**
 * newMessageId — unique ID for a MessagePart (assistant content text).
 */
export function newMessageId(): string {
  return `msg-${ts36()}-${_messageCounter++}`
}

/**
 * newToolCallId — unique ID for a ToolCallPart.
 * @param ord — the parallel tool-call slot index (0-based) from the wire chunk.
 *              Used as part of the ID so sibling calls are distinguishable.
 */
export function newToolCallId(ord: number): string {
  return `call-${ord}-${ts36()}`
}
