import type { BaseEvent } from '../core/event-bus-core'

/**
 * Extract forkId from an event if present.
 * Events that support forking include a forkId field (string | null).
 */
export function extractForkIdFromEvent(event: BaseEvent): string | null {
  if ('forkId' in event) {
    const forkId = (event as Record<string, unknown>).forkId
    if (typeof forkId === 'string' || forkId === null) {
      return forkId
    }
  }
  return null
}

/**
 * Extract forkId from a signal value if present.
 * Signal values can optionally include a forkId field to enable fork-specific reads.
 */
export function extractForkIdFromSignal(value: unknown): string | null {
  if (value && typeof value === 'object' && 'forkId' in value) {
    const forkId = (value as Record<string, unknown>).forkId
    if (typeof forkId === 'string' || forkId === null) {
      return forkId
    }
  }
  return null
}
