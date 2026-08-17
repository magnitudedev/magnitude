import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import {
  ContextUsageIndicator,
  contextUsageStrokeClass,
  contextUsageTooltipLines,
} from "./context-usage-indicator"

describe("context usage presentation", () => {
  it("keeps the tooltip at three lines and reports only percent remaining", () => {
    expect(
      contextUsageTooltipLines(
        { tokenEstimate: 15_800, isCompacting: false },
        200_000
      )
    ).toEqual(["Context", "15.8k / 200k tokens", "92% remaining"])

    expect(
      contextUsageTooltipLines(
        { tokenEstimate: 15_800, isCompacting: true },
        200_000
      )
    ).toEqual(["Compacting...", "15.8k / 200k tokens", "92% remaining"])

    expect(contextUsageTooltipLines(null, 200_000)).toEqual([
      "Context",
      "- / 200k tokens",
      "100% remaining",
    ])
  })

  it("uses the approved usage thresholds and compaction override", () => {
    expect(contextUsageStrokeClass(69.99, false)).toContain("stroke-blue-700")
    expect(contextUsageStrokeClass(70, false)).toContain("stroke-orange-700")
    expect(contextUsageStrokeClass(89.99, false)).toContain("stroke-orange-700")
    expect(contextUsageStrokeClass(90, false)).toContain("stroke-red-600")
    expect(contextUsageStrokeClass(95, true)).toContain("stroke-violet-700")
  })

  it("rotates the authoritative arc only while compaction is projected", () => {
    const compacting = renderToStaticMarkup(
      <ContextUsageIndicator
        context={{ tokenEstimate: 150_000, isCompacting: true }}
        tokenCap={200_000}
      />
    )
    const idle = renderToStaticMarkup(
      <ContextUsageIndicator
        context={{ tokenEstimate: 150_000, isCompacting: false }}
        tokenCap={200_000}
      />
    )

    expect(compacting).toContain("Compacting...")
    expect(compacting).toContain("context-rewind")
    expect(compacting).toContain("stroke-violet-700")
    expect(idle).not.toContain("context-rewind")
    expect(idle).toContain("stroke-orange-700")
  })
})
