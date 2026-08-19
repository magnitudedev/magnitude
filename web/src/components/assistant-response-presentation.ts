import type {
  DisplayTimeline,
  DisplayTimelineEntry,
  WorkSummaryMessage,
} from "@magnitudedev/sdk"
import { messageForEntry } from "@magnitudedev/client-common"

export interface AssistantResponsePresentation {
  readonly entries: readonly DisplayTimelineEntry[]
  readonly latestAssistantId: string | null
  readonly summaryByAssistantId: ReadonlyMap<string, WorkSummaryMessage>
}

function isUserEntry(entry: DisplayTimelineEntry): boolean {
  return entry.kind === "message" && entry.role === "user"
}

/**
 * Derives web-only response blocks from the authoritative chronological
 * timeline. A completed work summary belongs to the last assistant response
 * in the current user turn; summaries without a response remain standalone.
 */
export function deriveAssistantResponsePresentation(
  timeline: DisplayTimeline,
): AssistantResponsePresentation {
  const summaryByAssistantId = new Map<string, WorkSummaryMessage>()
  const incorporatedSummaryIds = new Set<string>()
  let latestAssistantId: string | null = null
  let currentResponseId: string | null = null

  for (const entry of timeline.presentation.entries) {
    if (isUserEntry(entry)) {
      currentResponseId = null
      continue
    }
    if (entry.kind !== "message") continue
    const message = messageForEntry(timeline, entry)
    if (message?.type === "assistant_message") {
      latestAssistantId = message.id
      currentResponseId = message.id
      continue
    }
    if (message?.type === "work_summary" && currentResponseId !== null) {
      summaryByAssistantId.set(currentResponseId, message)
      incorporatedSummaryIds.add(message.id)
      currentResponseId = null
    }
  }

  return {
    entries: timeline.presentation.entries.filter((entry) => {
      if (entry.kind !== "message") return true
      const message = messageForEntry(timeline, entry)
      return message?.type !== "work_summary" || !incorporatedSummaryIds.has(message.id)
    }),
    latestAssistantId,
    summaryByAssistantId,
  }
}
