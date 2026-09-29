import { describe, expect, it } from "vitest"
import { Option } from "effect"
import type { LocalInferenceHardware, ModelLoadDevice, ModelOptimizationProgress } from "@magnitudedev/sdk"
import { describeModelOptimization, formatModelOptimization, modelOptimizationFraction } from "./model-load"

const device = { deviceId: "metal:0", backend: "metal" } as unknown as ModelLoadDevice
const hardware = Option.some({
  accelerators: [{ acceleratorId: "metal:0", name: "Apple M3 Max", backend: "metal", memoryDomainId: "unified" }],
} as unknown as LocalInferenceHardware)

describe("model optimization wording", () => {
  it("names the accelerator while tuning and has no share while preparing", () => {
    const preparing: ModelOptimizationProgress = { stage: "preparing", completed: 0, total: 0, device: Option.some(device) }
    const tuning: ModelOptimizationProgress = { stage: "tuning", completed: 42, total: 100, device: Option.some(device) }

    expect(describeModelOptimization(preparing, hardware)).toBe("Preparing to optimize…")
    expect(describeModelOptimization(tuning, hardware)).toBe("Optimizing for Apple M3 Max…")
    expect(describeModelOptimization(tuning, Option.none())).toBe("Optimizing…")
    expect(modelOptimizationFraction(preparing)).toEqual(Option.none())
    expect(formatModelOptimization(preparing)).toBe("Optimizing")
    expect(formatModelOptimization(tuning)).toBe("Optimizing 42%")
    expect(formatModelOptimization({ ...tuning, completed: 232, total: 400 })).toBe("Optimizing 58%")
  })
})
