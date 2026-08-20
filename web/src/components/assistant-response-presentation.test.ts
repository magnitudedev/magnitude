import { describe, expect, it } from "vitest"
import { Option } from "effect"
import type { DisplayMessage, DisplayTimeline } from "@magnitudedev/sdk"
import {
  appendMessageToTimeline,
  emptyTimeline,
} from "@magnitudedev/client-common"
import { deriveAssistantResponsePresentation } from "./assistant-response-presentation"

const user = (id: string, timestamp: number): DisplayMessage => ({
  id,
  type: "user_message",
  content: id,
  timestamp,
  taskMode: false,
  attachments: [],
})

const assistant = (id: string, timestamp: number): DisplayMessage => ({
  id,
  type: "assistant_message",
  content: id,
  timestamp,
})

const summary = (id: string, timestamp: number): DisplayMessage => ({
  id,
  type: "work_summary",
  chainId: id,
  durationMs: 1_000,
  phase: "worked",
  performance: Option.none(),
  timestamp,
})

function timeline(messages: readonly DisplayMessage[]): DisplayTimeline {
  return messages.reduce(appendMessageToTimeline, emptyTimeline())
}

describe("deriveAssistantResponsePresentation", () => {
  it("incorporates each summary into the last assistant response in its user turn", () => {
    const view = deriveAssistantResponsePresentation(timeline([
      user("user-1", 1),
      assistant("assistant-1", 2),
      summary("summary-1", 3),
      user("user-2", 4),
      assistant("assistant-2a", 5),
      assistant("assistant-2b", 6),
      summary("summary-2", 7),
    ]))

    expect(view.summaryByAssistantId.get("assistant-1")?.id).toBe("summary-1")
    expect(view.summaryByAssistantId.has("assistant-2a")).toBe(false)
    expect(view.summaryByAssistantId.get("assistant-2b")?.id).toBe("summary-2")
    expect(view.footerByAssistantId.has("assistant-2a")).toBe(false)
    expect(view.footerByAssistantId.get("assistant-1")?.isLatest).toBe(false)
    expect(view.footerByAssistantId.get("assistant-2b")).toEqual({
      content: "assistant-2a\n\nassistant-2b",
      timestamp: 6,
      isLatest: true,
    })
    expect(view.entries.flatMap((entry) =>
      entry.kind === "message" ? [entry.messageId] : []
    )).toEqual(["user-1", "assistant-1", "user-2", "assistant-2a", "assistant-2b"])
  })

  it("keeps summaries standalone when the turn has no assistant response", () => {
    const view = deriveAssistantResponsePresentation(timeline([
      user("user-1", 1),
      summary("interrupted-summary", 2),
    ]))

    expect(view.summaryByAssistantId.size).toBe(0)
    expect(view.footerByAssistantId.size).toBe(0)
    expect(view.entries.flatMap((entry) =>
      entry.kind === "message" ? [entry.messageId] : []
    )).toEqual(["user-1", "interrupted-summary"])
  })

  it("does not attach a later turn's summary to an older assistant response", () => {
    const view = deriveAssistantResponsePresentation(timeline([
      user("user-1", 1),
      assistant("assistant-1", 2),
      user("user-2", 3),
      summary("summary-2", 4),
    ]))

    expect(view.summaryByAssistantId.size).toBe(0)
    expect(view.footerByAssistantId.size).toBe(0)
  })

  it("does not move a footer between assistant messages while the turn is active", () => {
    const first = deriveAssistantResponsePresentation(timeline([
      user("user-1", 1),
      assistant("assistant-1", 2),
    ]))
    const second = deriveAssistantResponsePresentation(timeline([
      user("user-1", 1),
      assistant("assistant-1", 2),
      assistant("assistant-2", 3),
    ]))

    expect(first.footerByAssistantId.size).toBe(0)
    expect(second.footerByAssistantId.size).toBe(0)
  })
})
