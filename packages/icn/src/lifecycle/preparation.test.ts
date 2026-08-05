import { describe, expect, it } from "vitest"
import { icnPreparationBackend } from "./preparation"

describe("icnPreparationBackend", () => {
  it.each([
    [{ type: "cpu", hardwareLabel: "AMD Ryzen" }, { _tag: "Cpu", hardwareLabel: "AMD Ryzen" }],
    [{ type: "metal", hardwareLabel: "Apple M4 Max" }, { _tag: "Metal", hardwareLabel: "Apple M4 Max" }],
    [{ type: "cuda", hardwareLabel: "NVIDIA RTX 5090" }, { _tag: "Cuda", hardwareLabel: "NVIDIA RTX 5090" }],
    [{ type: "vulkan", hardwareLabel: "AMD Radeon" }, { _tag: "Vulkan", hardwareLabel: "AMD Radeon" }],
  ] as const)("preserves the %s startup backend", (backend, expected) => {
    expect(icnPreparationBackend(backend)).toEqual(expected)
  })
})
