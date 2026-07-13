import type { DisplayMessage, DisplayState, DisplayTimeline, DisplayTimelineMessages } from '@magnitudedev/sdk'
import { forkIdToKey } from '@magnitudedev/sdk'

export function getFork(state: DisplayState, forkId: string | null): DisplayTimeline | undefined {
  return state.timelines[forkIdToKey(forkId)]
}

/**
 * Materialize the ordered message list from the normalized byId/order form.
 * Pure — consumers memoize on `messages` identity (the ref-preserving store
 * keeps it stable when untouched).
 */
export function orderedMessages(messages: DisplayTimelineMessages): DisplayMessage[] {
  const result: DisplayMessage[] = []
  for (const id of messages.order) {
    const message = messages.byId[id]
    if (message !== undefined) result.push(message)
  }
  return result
}

/** Tail message of a timeline, or null when empty. */
export function lastMessage(messages: DisplayTimelineMessages): DisplayMessage | null {
  const lastId = messages.order[messages.order.length - 1]
  if (lastId === undefined) return null
  return messages.byId[lastId] ?? null
}
