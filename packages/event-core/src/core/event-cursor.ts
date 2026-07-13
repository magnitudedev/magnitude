import { Schema } from 'effect'

import type { BaseEvent, Timestamped } from './event-bus-core'

export const EventCursorSchema = Schema.Struct({
  index: Schema.Number,
  timestamp: Schema.Number,
})

export type EventCursor = typeof EventCursorSchema.Type

export function cursorFromEvent<E extends BaseEvent>(
  event: Timestamped<E>,
  index: number
): EventCursor {
  return { index, timestamp: event.timestamp }
}

export function isCursorEvent<E extends BaseEvent>(
  event: Timestamped<E> | undefined,
  cursor: EventCursor
): boolean {
  return event !== undefined && event.timestamp === cursor.timestamp
}
