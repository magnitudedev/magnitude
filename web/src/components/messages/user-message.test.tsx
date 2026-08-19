import { describe, expect, it } from "vitest"
import { renderToStaticMarkup } from "react-dom/server"
import type { QueuedUserMessage as QueuedUserMessageType, UserMessage as UserMessageType } from "@magnitudedev/sdk"
import { QueuedUserMessage } from "./queued-user-message"
import { UserMessage } from "./user-message"

const attachment = { type: "mention_file" as const, path: "$M/attachments/notes.md" }

describe("attachment-only user messages", () => {
  it("renders an accepted attachment without an empty text bubble or copy action", () => {
    const message: UserMessageType = {
      id: "message-1",
      type: "user_message",
      content: "",
      timestamp: 1,
      taskMode: false,
      attachments: [attachment],
    }
    const html = renderToStaticMarkup(<UserMessage message={message} />)

    expect(html).toContain("notes.md")
    expect(html).not.toContain("data-user-message-content")
    expect(html).not.toContain("Copy")
  })

  it("renders a queued attachment without an empty text bubble or copy action", () => {
    const message: QueuedUserMessageType = {
      id: "message-1",
      type: "queued_user_message",
      content: "  ",
      timestamp: 1,
      taskMode: false,
      attachments: [attachment],
    }
    const html = renderToStaticMarkup(<QueuedUserMessage message={message} />)

    expect(html).toContain("notes.md")
    expect(html).toContain("Queued")
    expect(html).not.toContain("data-user-message-content")
    expect(html).not.toContain("Copy")
  })
})
