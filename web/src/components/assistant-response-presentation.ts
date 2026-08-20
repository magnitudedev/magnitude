import type {
  AssistantMessage,
  DisplayTimeline,
  DisplayTimelineEntry,
  WorkSummaryMessage,
} from "@magnitudedev/sdk"
import { messageForEntry } from "@magnitudedev/client-common"

export interface AssistantResponsePresentation {
  readonly entries: readonly DisplayTimelineEntry[]
  readonly summaryByAssistantId: ReadonlyMap<string, WorkSummaryMessage>
  readonly footerByAssistantId: ReadonlyMap<string, AssistantResponseFooter>
}

export interface AssistantResponseFooter {
  readonly content: string
  readonly timestamp: number
  readonly isLatest: boolean
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
  const footerByAssistantId = new Map<string, AssistantResponseFooter>()
  const incorporatedSummaryIds = new Set<string>()
  let latestAssistantId: string | null = null
  let currentResponseId: string | null = null
  let turnResponses: AssistantMessage[] = []

  const finishTurn = (): void => {
    const finalResponse = turnResponses.at(-1)
    if (finalResponse !== undefined) {
      const content = turnResponses
        .map((message) => message.content.trim())
        .filter((message) => message.length > 0)
        .join("\n\n")
      if (content.length > 0) {
        footerByAssistantId.set(finalResponse.id, {
          content,
          timestamp: finalResponse.timestamp,
          isLatest: false,
        })
      }
    }
    turnResponses = []
  }

  for (const entry of timeline.presentation.entries) {
    if (isUserEntry(entry)) {
      turnResponses = []
      currentResponseId = null
      continue
    }
    if (entry.kind !== "message") continue
    const message = messageForEntry(timeline, entry)
    if (message?.type === "assistant_message") {
      latestAssistantId = message.id
      currentResponseId = message.id
      turnResponses.push(message)
      continue
    }
    if (message?.type === "work_summary" && currentResponseId !== null) {
      summaryByAssistantId.set(currentResponseId, message)
      incorporatedSummaryIds.add(message.id)
      finishTurn()
      currentResponseId = null
    }
  }
  if (latestAssistantId !== null) {
    const latestFooter = footerByAssistantId.get(latestAssistantId)
    if (latestFooter !== undefined) {
      footerByAssistantId.set(latestAssistantId, { ...latestFooter, isLatest: true })
    }
  }

  return {
    entries: timeline.presentation.entries.filter((entry) => {
      if (entry.kind !== "message") return true
      const message = messageForEntry(timeline, entry)
      return message?.type !== "work_summary" || !incorporatedSummaryIds.has(message.id)
    }),
    summaryByAssistantId,
    footerByAssistantId,
  }
}
