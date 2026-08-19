import { describe, expect, it } from "vitest"
import type { TimelineStatus } from "@magnitudedev/client-common"
import { emptyTimeline } from "@magnitudedev/client-common"
import { chatTimelinePlaceholderState } from "./chat-timeline-state"

const pending: TimelineStatus = { _tag: "pending", forkId: null }
const empty: TimelineStatus = { _tag: "empty", forkId: null, timeline: emptyTimeline() }
const none: TimelineStatus = { _tag: "none" }

describe("chatTimelinePlaceholderState", () => {
  it("shows the new-chat state when there is no session or entry", () => {
    expect(chatTimelinePlaceholderState(null, none, 0)).toEqual({
      isSessionLoading: false,
      isEmpty: true,
    })
  })

  it("shows loading for a selected session without accepted entries", () => {
    expect(chatTimelinePlaceholderState("session-1", pending, 0)).toEqual({
      isSessionLoading: true,
      isEmpty: false,
    })
  })

  it("shows an accepted empty session as empty", () => {
    expect(chatTimelinePlaceholderState("session-1", empty, 0)).toEqual({
      isSessionLoading: false,
      isEmpty: true,
    })
  })

  it.each([
    [null, none],
    ["session-1", pending],
    ["session-1", empty],
  ] as const)("lets visible entries outrank placeholder state", (sessionId, status) => {
    expect(chatTimelinePlaceholderState(sessionId, status, 1)).toEqual({
      isSessionLoading: false,
      isEmpty: false,
    })
  })
})
