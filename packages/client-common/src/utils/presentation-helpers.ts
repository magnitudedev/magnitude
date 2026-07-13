import type {
  DisplayMessage,
  DisplayTimeline,
  DisplayTimelineEntry,
} from '@magnitudedev/sdk'

/**
 * Maximum number of output lines rendered in transcript mode before the
 * tail is hidden behind an "N lines hidden" marker. Shared policy across
 * CLI and web surfaces; both cap transcript output at the same length so
 * a huge shell/worker stream cannot blow up the scrollback.
 */
export const TRANSCRIPT_LINE_CAP = 1000

/**
 * Resolves the {@link DisplayMessage} referenced by a `message` timeline entry
 * from the timeline's message index, or `null` if the message is no longer
 * present (e.g. dropped from the window).
 */
export function messageForEntry(
  timeline: DisplayTimeline,
  entry: Extract<DisplayTimelineEntry, { kind: 'message' }>,
): DisplayMessage | null {
  return timeline.messages.byId[entry.messageId] ?? null
}
