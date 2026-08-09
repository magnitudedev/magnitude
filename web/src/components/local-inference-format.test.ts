import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { PRIMARY_SLOT_ID, ProviderIdSchema, ProviderModelIdSchema, ReasoningEffortSchema } from "@magnitudedev/sdk"
import { downloadLabel, downloadProgress, formatBytes, formatContext, slotStatus } from "./local-inference-format"

describe("local inference formatting", () => {
  it("formats byte and context scales", () => {
    expect(formatBytes(24_000_000_000)).toBe("24 GB")
    expect(formatBytes(1_500_000_000)).toBe("1.5 GB")
    expect(formatContext(100_000)).toBe("100K")
    expect(formatContext(1_500_000)).toBe("1.5M")
  })

  it("bounds authoritative download progress", () => {
    expect(downloadProgress({ _tag: "Downloading", attemptIds: ["attempt" as never], stage: "downloading", completedBytes: 5, totalBytes: 10, bytesPerSecond: Option.none() })).toBe(50)
    expect(downloadLabel({ _tag: "Failed", attemptIds: ["attempt" as never], completedBytes: 2, totalBytes: 10, failure: { code: "failed", message: "network failed", retryable: true } })).toBe("network failed")
  })

  it("does not claim runtime readiness from selection alone", () => {
    const providerId = ProviderIdSchema.make("local")
    const providerModelId = ProviderModelIdSchema.make("model")
    const status = slotStatus({
      _tag: "ConfiguredLocal",
      slotId: PRIMARY_SLOT_ID,
      selection: { providerId, providerModelId, reasoningEffort: ReasoningEffortSchema.make("none") },
      descriptor: { providerId, providerModelId, displayName: "Model" },
      availability: { _tag: "Available" },
      instance: Option.none(),
      actions: ["Load"],
    })
    expect(status.label).toBe("Not loaded")
  })
})
