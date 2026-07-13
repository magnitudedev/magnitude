import type {
  DisplayMessage,
  DisplayTimeline,
  DisplayTimelineMessages,
  DisplayTimelinePresentation,
  DisplayTimelineWindowInfo,
} from '@magnitudedev/protocol'

export interface DisplayTimelineOrdinary {
  readonly mode: DisplayTimeline['mode']
  readonly streamingMessageId: string | null
}

/**
 * Normalize windowed messages for transport: bodies keyed by id, order as an
 * id array. Window movement then patches as cheap `order` churn plus one
 * `byId` entry per newly-covered message.
 */
const normalizeMessages = (messages: readonly DisplayMessage[]): DisplayTimelineMessages => {
  const byId: Record<string, DisplayMessage> = {}
  const order: string[] = []
  for (const message of messages) {
    byId[message.id] = message
    order.push(message.id)
  }
  return { byId, order }
}

export const materializeDisplayTimeline = (
  ordinary: DisplayTimelineOrdinary,
  messages: readonly DisplayMessage[],
  window: DisplayTimelineWindowInfo,
  presentation: DisplayTimelinePresentation,
): DisplayTimeline => ({
  mode: ordinary.mode,
  messages: normalizeMessages(messages),
  streamingMessageId: ordinary.streamingMessageId,
  window,
  presentation,
})
