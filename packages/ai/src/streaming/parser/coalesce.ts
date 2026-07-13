import type { FieldEvent } from '../types'

/**
 * Merges adjacent field_delta events with the same path.
 * Any non-delta event breaks coalescing.
 */
export function coalesce(events: readonly FieldEvent[]): FieldEvent[] {
  const result: FieldEvent[] = []

  for (const event of events) {
    if (
      event._tag === 'field_delta' &&
      result.length > 0
    ) {
      const prev = result[result.length - 1]
      if (
        prev._tag === 'field_delta' &&
        pathsEqual(prev.path, event.path)
      ) {
        result[result.length - 1] = {
          _tag: 'field_delta',
          path: prev.path,
          delta: prev.delta + event.delta,
        }
        continue
      }
    }
    result.push(event)
  }

  return result
}

function pathsEqual(a: readonly string[], b: readonly string[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false
  }
  return true
}
