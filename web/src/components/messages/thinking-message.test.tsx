import { renderToStaticMarkup } from "react-dom/server"
import { Option } from "effect"
import { describe, expect, it } from "vitest"
import type { ThinkingMessage as ThinkingMessageType } from "@magnitudedev/sdk"
import { ThinkingMessage } from "./thinking-message"

const message = (
  overrides: Partial<ThinkingMessageType> = {},
): ThinkingMessageType => ({
  id: "thinking:turn-1:1",
  type: "thinking",
  content: "Consider the implementation boundaries.",
  label: Option.none(),
  phase: "completed",
  completedAt: Option.some(6_500),
  timestamp: 1_000,
  ...overrides,
})

describe("ThinkingMessage", () => {
  it("renders completed thinking as a collapsed disclosure with server timing", () => {
    const html = renderToStaticMarkup(<ThinkingMessage message={message()} />)
    expect(html).toContain("Thought for 5 seconds")
    expect(html).toContain('aria-expanded="false"')
    expect(html).not.toContain("Consider the implementation boundaries.")
    expect(html).not.toContain("Brain")
  })

  it("renders active thinking with shimmer and no invented completion", () => {
    const html = renderToStaticMarkup(
      <ThinkingMessage
        message={message({ phase: "active", completedAt: Option.none() })}
      />
    )
    expect(html).toContain("Thinking")
    expect(html).toContain("animate-shimmer")
    expect(html).not.toContain("Thought for")
  })
})
