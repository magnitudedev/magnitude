import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { renderToStaticMarkup } from "react-dom/server"
import type { WorkSummaryMessage } from "@magnitudedev/sdk"
import { workSummaryLabel } from "./work-summary"
import { MessageDispatch } from "./index"

const summary = (
  overrides: Partial<WorkSummaryMessage> = {}
): WorkSummaryMessage => ({
  id: "work_summary:chain-1",
  type: "work_summary",
  chainId: "chain-1",
  durationMs: 5_500,
  phase: "worked",
  performance: Option.none(),
  timestamp: 1,
  ...overrides,
})

describe("workSummaryLabel", () => {
  it("shows duration when performance data is unavailable", () => {
    expect(workSummaryLabel(summary())).toBe("Worked for 5 seconds")
  })

  it("shows the model, duration, and measured decode rate", () => {
    expect(
      workSummaryLabel(
        summary({
          durationMs: 65_000,
          performance: Option.some({
            modelDisplayName: "Qwen3 Coder",
            decodeTokensPerSecond: Option.some(20.45),
          }),
        })
      )
    ).toBe("Qwen3 Coder worked for 1:05 20.4 tok/s")
  })

  it("keeps model and duration when decode rate is unavailable", () => {
    expect(
      workSummaryLabel(
        summary({
          durationMs: 1_000,
          performance: Option.some({
            modelDisplayName: "DeepSeek V4 Flash",
            decodeTokensPerSecond: Option.none(),
          }),
        })
      )
    ).toBe("DeepSeek V4 Flash worked for 1 second")
  })

  it("is rendered by the web timeline message dispatcher", () => {
    const html = renderToStaticMarkup(<MessageDispatch message={summary()} />)
    expect(html).toContain('data-message-type="work-summary"')
    expect(html).toContain("Worked for 5 seconds")
  })
})
