import { describe, expect, it } from "vitest"
import { renderToStaticMarkup } from "react-dom/server"
import { Option } from "effect"
import type {
  AssistantMessage as AssistantMessageType,
  WorkSummaryMessage,
} from "@magnitudedev/sdk"
import { AssistantMessage } from "./assistant-message"

const message: AssistantMessageType = {
  id: "assistant-1",
  type: "assistant_message",
  content: "Finished **everything**.",
  timestamp: Date.now() - 3 * 60_000,
}

const summary: WorkSummaryMessage = {
  id: "summary-1",
  type: "work_summary",
  chainId: "chain-1",
  durationMs: 12_000,
  phase: "worked",
  performance: Option.none(),
  timestamp: Date.now(),
}

describe("AssistantMessage metadata", () => {
  it("places work status before left-aligned icon-only metadata", () => {
    const html = renderToStaticMarkup(
      <AssistantMessage message={message} workSummary={summary} isLatest />,
    )

    const response = html.indexOf("Finished")
    const worked = html.indexOf("Worked for 12 seconds")
    const copy = html.indexOf('aria-label="Copy response"')
    const time = html.indexOf("minutes ago")
    expect(response).toBeGreaterThan(-1)
    expect(worked).toBeGreaterThan(response)
    expect(copy).toBeGreaterThan(worked)
    expect(time).toBeGreaterThan(copy)
    expect(html).toContain("gap-2")
    expect(html).toContain("opacity-100")
    expect(html).not.toContain(">Copy response</button>")
  })

  it("keeps older metadata in the layout but reveals it only on hover or focus", () => {
    const html = renderToStaticMarkup(
      <AssistantMessage message={message} workSummary={summary} />,
    )

    expect(html).toContain("opacity-0")
    expect(html).toContain("group-hover/assistant:opacity-100")
    expect(html).toContain("group-focus-within/assistant:opacity-100")
    expect(html).toContain("Worked for 12 seconds")
  })
})
