/**
 * Coalescing layer — merges adjacent streaming chunk events of the same type
 * to reduce event volume without losing data.
 */

import type { TurnEngineEvent } from '../types'

type Writable<T> = { -readonly [K in keyof T]: T[K] }

export interface CoalescingBuffer {
  key: string
  event: Writable<TurnEngineEvent>
}

/**
 * Returns a stable key for coalescable events, or null for non-coalescable ones.
 * Events with the same key are merged by appending their text/delta fields.
 */
export function classifyEvent(event: TurnEngineEvent): string | null {
  switch (event._tag) {
    case 'LensChunk': return 'lens'
    case 'MessageChunk': return `message:${event.id}`
    case 'ProseChunk': return 'prose'
    case 'ToolInputFieldChunk': return `field:${event.toolCallId}:${event.field}`
    default: return null
  }
}

/**
 * Merge source into target in-place by appending text or delta.
 */
export function mergeEvent(target: Writable<TurnEngineEvent>, source: TurnEngineEvent): void {
  if ('text' in target && 'text' in source) {
    (target as { text: string }).text += (source as { text: string }).text
  } else if ('delta' in target && 'delta' in source) {
    (target as { delta: string }).delta += (source as { delta: string }).delta
  }
}
