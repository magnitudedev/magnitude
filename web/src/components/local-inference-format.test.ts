import { describe, expect, it } from "vitest"
import { Option } from "effect"
import {
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
} from "@magnitudedev/sdk"
import {
  formatBytes,
  formatContext,
  slotStatus,
  transferLabel,
  transferProgress,
} from "./local-inference-format"

describe("local inference formatting", () => {
  it("formats byte and context scales", () => {
    expect(formatBytes(24_000_000_000)).toBe("24 GB")
    expect(formatBytes(1_500_000_000)).toBe("1.5 GB")
    expect(formatContext(100_000)).toBe("100K")
    expect(formatContext(1_500_000)).toBe("1.5M")
  })

  it("bounds authoritative transfer progress", () => {
    expect(transferProgress({ completedBytes: 5, totalBytes: 10 })).toBe(50)
    expect(transferProgress({ completedBytes: 12, totalBytes: 10 })).toBe(100)
    expect(
      transferLabel({ stage: "downloading", completedBytes: 5, totalBytes: 10 })
    ).toContain("50%")
  })

  it("does not claim runtime readiness from selection alone", () => {
    const providerId = ProviderIdSchema.make("local")
    const providerModelId = ProviderModelIdSchema.make("model")
    const status = slotStatus({
      _tag: "ConfiguredLocal",
      slotId: PRIMARY_SLOT_ID,
      selection: {
        providerId,
        providerModelId,
        reasoningEffort: ReasoningEffortSchema.make("none"),
      },
      descriptor: {
        providerId,
        providerModelId,
        displayName: "Model",
        variantLabel: Option.none(),
      },
      availability: { _tag: "Available" },
      residency: { _tag: "Unloaded" },
      actions: ["Load"],
    })
    expect(status.label).toBe("Not loaded")
  })
})
