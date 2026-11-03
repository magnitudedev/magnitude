import { describe, expect, it } from "vitest"
import { Option } from "effect"
import type { LocalInferenceHardware, ModelLoadDevice, ModelOptimizationProgress, ModelResidency } from "@magnitudedev/sdk"
import {
  modelOptimizationPendingIdle,
  MEMORY_PRESSURE_TITLE,
  describeMemoryShortage,
  describeModelLoadFailure,
  describeModelOptimization,
  formatModelOptimization,
  modelOptimizationFraction,
  modelStoppedForMemory,
} from "./model-load"

const device = { deviceId: "metal:0", backend: "metal" } as unknown as ModelLoadDevice
const hardware = Option.some({
  accelerators: [{ acceleratorId: "apple", name: "Apple M3 Max", executors: [{ deviceId: "metal:0", backend: "metal", memoryDomainIds: ["unified"] }] }],
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

  it("reads a pending optimization as pending only while nothing of the model runs", () => {
    const installation = { _tag: "Resolved", installedBytes: 1, primaryPath: "/m.gguf", ownership: "Magnitude" } as never
    const pending = (residencyState: ModelResidency) => ({ _tag: "OptimizationPending" as const, installation, residencyState })

    expect(modelOptimizationPendingIdle(pending({ _tag: "Unloaded" }))).toBe(true)
    expect(modelOptimizationPendingIdle(pending({ _tag: "Stopped", reason: "user_stop" }))).toBe(true)
    expect(modelOptimizationPendingIdle(pending({ _tag: "Ready" } as never))).toBe(false)
    expect(modelOptimizationPendingIdle(pending({ _tag: "Requested" }))).toBe(false)
    expect(modelOptimizationPendingIdle({ _tag: "Installed", installation, residencyState: { _tag: "Unloaded" } })).toBe(false)
    expect(modelOptimizationPendingIdle({
      _tag: "Optimizing", installation, residencyState: { _tag: "Unloaded" },
      progress: { stage: "tuning", completed: 1, total: 2, device: Option.none() },
    })).toBe(false)
  })
})

describe("memory shortage and stop wording", () => {
  it("states a blocked shortage with its two figures and a load under pressure with none", () => {
    const blocked = describeMemoryShortage({ _tag: "Blocked", requiredBytes: 30 * 2 ** 30, availableBytes: 24 * 2 ** 30 })
    expect(blocked).toBe("This model needs 32.3 GB and 25.8 GB is available. Quit apps you aren’t using, or choose a smaller model.")
    expect(describeMemoryShortage({ _tag: "UnderPressure" })).not.toMatch(/\d/)
  })

  it("describes a shortage in the user's terms and passes any other failure's message through", () => {
    expect(describeModelLoadFailure({
      _tag: "MemoryShortage", code: "memory_shortage", message: "private bytes", retryable: true,
      shortage: { _tag: "UnderPressure" },
    })).toBe(`${MEMORY_PRESSURE_TITLE}. Loading stopped to keep other apps responsive. Quit apps you aren’t using, or try again in a moment.`)
    expect(describeModelLoadFailure({ code: "worker_exited", message: "the worker exited", retryable: true }))
      .toBe("the worker exited")
  })

  it("recognizes only a stop for memory pressure", () => {
    expect(modelStoppedForMemory({ _tag: "Stopped", reason: "memory_pressure" })).toBe(true)
    expect(modelStoppedForMemory({ _tag: "Stopped", reason: "user_stop" })).toBe(false)
    expect(modelStoppedForMemory({ _tag: "Unloaded" })).toBe(false)
  })
})
