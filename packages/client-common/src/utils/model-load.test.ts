import { describe, expect, it } from "vitest"
import { Option } from "effect"
import type { LocalInferenceHardware, ModelLoadDevice, ModelOptimizationProgress } from "@magnitudedev/sdk"
import {
  MEMORY_SHORTAGE_TITLE,
  describeMemoryShortage,
  describeModelLoadFailure,
  describeModelOptimization,
  formatModelOptimization,
  modelOptimizationFraction,
  modelStoppedForMemory,
} from "./model-load"

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

describe("memory shortage and stop wording", () => {
  it("states a blocked shortage with its two figures and a load under pressure with none", () => {
    const blocked = describeMemoryShortage({ _tag: "Blocked", requiredBytes: 30 * 2 ** 30, availableBytes: 24 * 2 ** 30 })
    expect(blocked).toBe("This model needs 30 GB and 24 GB is available. Quit apps you aren’t using, or choose a smaller model.")
    expect(describeMemoryShortage({ _tag: "UnderPressure" })).not.toMatch(/\d/)
  })

  it("describes a shortage in the user's terms and passes any other failure's message through", () => {
    expect(describeModelLoadFailure({
      _tag: "MemoryShortage", code: "memory_shortage", message: "private bytes", retryable: true,
      shortage: { _tag: "UnderPressure" },
    })).toBe(`${MEMORY_SHORTAGE_TITLE}. Loading stopped because your computer ran low on memory. Quit apps you aren’t using, or choose a smaller model.`)
    expect(describeModelLoadFailure({ code: "worker_exited", message: "the worker exited", retryable: true }))
      .toBe("the worker exited")
  })

  it("recognizes only a stop for memory pressure", () => {
    expect(modelStoppedForMemory({ _tag: "Stopped", reason: "memory_pressure" })).toBe(true)
    expect(modelStoppedForMemory({ _tag: "Stopped", reason: "user_stop" })).toBe(false)
    expect(modelStoppedForMemory({ _tag: "Unloaded" })).toBe(false)
  })
})
