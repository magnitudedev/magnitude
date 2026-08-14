import { describe, expect, it } from "vitest"
import type { DisplayMessage, DisplayViewShape } from "../index"
import { makeDisplayViewSnapshotFixture } from "./display-view-fixture"

const shape: DisplayViewShape = {
  timelines: {
    root: { kind: "tail", limit: 10, live: true, presentation: "default" },
  },
}

const messages: readonly DisplayMessage[] = [
  {
    id: "assistant-1",
    type: "assistant_message",
    content: "done",
    timestamp: 2,
  },
  {
    id: "user-1",
    type: "user_message",
    content: "start",
    timestamp: 1,
    taskMode: false,
    attachments: [],
  },
]

describe("makeDisplayViewSnapshotFixture", () => {
  it("builds one complete display state while preserving authoritative message order", () => {
    const snapshot = makeDisplayViewSnapshotFixture({
      shape,
      session: { sessionId: "session-1", title: "Fixture", cwd: "/repo" },
      status: { _tag: "Worked", lastProductiveMs: 20 },
      messages,
      messageOrder: ["user-1", "assistant-1"],
      entries: [],
    })

    expect(snapshot.state.timelines.root?.messages.order).toEqual(["user-1", "assistant-1"])
    expect(snapshot.state.timelines.root?.messages.byId["assistant-1"]).toEqual(messages[0])
    expect(snapshot.state.actors.root?.status).toEqual({ _tag: "Worked", lastProductiveMs: 20 })
    expect(snapshot.state.tasks.summary).toEqual({
      totalCount: 0,
      completedCount: 0,
      incompleteCount: 0,
    })
  })
})
