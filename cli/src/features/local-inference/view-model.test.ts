import { describe, expect, it } from "vitest"
import { Option } from "effect"
import {
  LocalInferenceAcceleratorIdSchema,
  LocalInferenceMemoryDomainIdSchema,
  LocalModelAvailableForDownload,
} from "@magnitudedev/sdk"
import {
  buildLocalInferenceSelections,
  describeLocalHardware,
  formatModelLoadProgress,
  selectionCapacityWarning,
  selectionMetadata,
} from "./view-model"
import { GIB, makeHardware, makeModel, makeView } from "./test-fixtures"

describe("local inference selection view model", () => {
  it("uses honest indeterminate and finishing labels around native fractional progress", () => {
    expect(formatModelLoadProgress(0)).toBe("Loading 0%")
    expect(formatModelLoadProgress(42)).toBe("Loading 42%")
    expect(formatModelLoadProgress(100)).toBe("Loading 100%")
  })

  it("presents unified memory from the hardware contract", () => {
    const memoryDomainId = LocalInferenceMemoryDomainIdSchema.make("unified")
    const hardware = makeHardware({
      platform: "MacOS",
      architecture: "Arm64",
      processor: Option.some("Apple M4 Max"),
      totalSystemMemoryBytes: 64 * GIB,
      accelerators: [{
        acceleratorId: LocalInferenceAcceleratorIdSchema.make("metal"),
        name: "Apple M4 Max",
        backend: "Metal",
        memoryDomainId,
      }],
      memoryDomains: [{
        memoryDomainId,
        kind: "UnifiedMemory",
        totalBytes: 64 * GIB,
        stableCapacityBytes: 52 * GIB,
        availableBytes: Option.none(),
        sharesSystemMemory: true,
      }],
    })

    expect(describeLocalHardware(hardware)).toEqual({
      system: {
        name: "Apple M4 Max",
        details: [
          "macOS · ARM64 · 16 logical CPU cores",
          "64.0 GiB unified memory · Metal GPU acceleration",
        ],
      },
      accelerators: [],
    })
  })

  it("classifies the downloaded model selected by a ready slot as running", () => {
    expect(buildLocalInferenceSelections(makeView())[0]?.kind).toBe("running")
  })

  it("keeps downloadable inventory entries actionable without duplicating state", () => {
    const entry = new LocalModelAvailableForDownload({ model: makeModel() })
    const selections = buildLocalInferenceSelections(makeView({ entries: [entry], ready: false }))
    expect(selections).toHaveLength(1)
    expect(selections[0]?.kind).toBe("recommendation")
    expect(selectionMetadata(selections[0]!)).toContain("Q4_K_M")
  })

  it("exposes a capacity warning from the inventory entry fit state", () => {
    const model = makeModel({
      fit: {
        _tag: "DoesNotFit",
        requiredBytes: 24 * GIB,
        availableBytes: 20 * GIB,
        limitingResource: "memory",
        memoryDomainIds: [],
      },
    })
    const selection = buildLocalInferenceSelections(makeView({
      entries: [new LocalModelAvailableForDownload({ model })],
      ready: false,
    }))[0]!
    expect(selectionCapacityWarning(selection)).toContain("estimated 24.0 GiB")
  })
})
