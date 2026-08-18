import { describe, expect, it } from "vitest"
import { ReasoningEffortSchema } from "@magnitudedev/sdk"
import { formatReasoningEffort } from "./model-properties"

describe("reasoning effort presentation", () => {
  it("uses the shared label for every canonical effort", () => {
    const cases = {
      none: "None",
      minimal: "Minimal",
      low: "Low",
      medium: "Medium",
      high: "High",
      xhigh: "XHigh",
      max: "Max",
      adaptive: "Adaptive",
    }

    for (const [effort, label] of Object.entries(cases)) {
      expect(formatReasoningEffort(ReasoningEffortSchema.make(effort))).toBe(label)
    }
  })
})
