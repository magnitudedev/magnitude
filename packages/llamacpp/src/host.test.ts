import { describe, expect, it } from "vitest"
import type { LlamaCppHostProfile, LlamaCppMemoryDomain, ModelFitRequest } from "./contracts"
import { planModelForProfile } from "./host"

const GIB = 1024 ** 3

const profile = (memoryDomains: readonly LlamaCppMemoryDomain[]): LlamaCppHostProfile => ({
  system: { totalMemoryBytes: 64 * GIB, cpuModel: "test", logicalCores: 8 },
  memoryDomains,
  runtimeProbe: "complete",
  warnings: [],
})

const request = (overrides: Partial<ModelFitRequest> = {}): ModelFitRequest => ({
  modelBytes: 6 * GIB,
  contextBytesPerSlot: 2 * GIB,
  parallelSlots: 1,
  modelLayerCount: 40,
  ...overrides,
})

const system = (capacity: number): LlamaCppMemoryDomain => ({
  id: "system",
  kind: "system",
  stableCapacityBytes: capacity,
  currentFreeBytes: null,
  sharesSystemMemory: false,
  devices: [],
  splitGroupId: null,
})

const device = (id: string, capacity: number, splitGroupId: string | null): LlamaCppMemoryDomain => ({
  id,
  kind: "physical_device",
  stableCapacityBytes: capacity,
  currentFreeBytes: null,
  sharesSystemMemory: false,
  devices: [{ backend: "cuda", name: id }],
  splitGroupId,
})

describe("llama.cpp host fit planning", () => {
  it("uses a unified working set as one capacity domain", () => {
    const unified: LlamaCppMemoryDomain = {
      id: "unified",
      kind: "unified_working_set",
      stableCapacityBytes: 10 * GIB,
      currentFreeBytes: null,
      sharesSystemMemory: true,
      devices: [{ backend: "metal", name: "Apple GPU" }],
      splitGroupId: null,
    }
    expect(planModelForProfile(request(), profile([unified]))).toMatchObject({
      fits: true,
      gpuLayers: -1,
      splitMode: "none",
      stableCapacityBytes: 10 * GIB,
    })
  })

  it("returns a concrete partial layer plan for viable CPU/GPU placement", () => {
    const plan = planModelForProfile(request(), profile([
      system(8 * GIB),
      device("gpu-0", 4 * GIB, null),
    ]))
    expect(plan.fits).toBe(true)
    expect(plan.gpuLayers).toBe(13)
    expect(plan.splitMode).toBe("layer")
  })

  it("does not combine unrelated discrete memory domains", () => {
    const plan = planModelForProfile(request({ modelBytes: 9 * GIB }), profile([
      system(4 * GIB),
      device("gpu-0", 4 * GIB, null),
      device("gpu-1", 4 * GIB, null),
    ]))
    expect(plan.stableCapacityBytes).toBe(8 * GIB)
    expect(plan.fits).toBe(false)
  })

  it("combines only devices in an explicit split group", () => {
    const plan = planModelForProfile(request({ modelBytes: 9 * GIB }), profile([
      system(4 * GIB),
      device("gpu-0", 4 * GIB, "cuda-split"),
      device("gpu-1", 4 * GIB, "cuda-split"),
    ]))
    expect(plan.stableCapacityBytes).toBe(12 * GIB)
    expect(plan.fits).toBe(true)
    expect(plan.gpuLayers).toBeGreaterThan(0)
  })

  it("does not claim hybrid viability when layer count is unknown", () => {
    const plan = planModelForProfile(request({ modelLayerCount: null }), profile([
      system(4 * GIB),
      device("gpu-0", 4 * GIB, null),
    ]))
    expect(plan.gpuLayers).toBe(0)
    expect(plan.fits).toBe(false)
  })
})
