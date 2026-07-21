import { describe, expect, it } from "vitest"
import type { LocalInferenceHostProfile } from "@magnitudedev/sdk"
import { deriveHardwareMemoryView } from "./hardware-memory"

const gib = 1024 ** 3

const host = (overrides: Partial<LocalInferenceHostProfile> = {}): LocalInferenceHostProfile => ({
  platform: "linux",
  architecture: "x86_64",
  topologyFingerprint: "test",
  systemMemoryBytes: 64 * gib,
  cpuModel: "Example CPU",
  logicalCores: 16,
  memoryDomains: [],
  residentMemory: null,
  ...overrides,
})

describe("hardware memory view", () => {
  it("groups model, compute, and auxiliary as fixed cost", () => {
    const view = deriveHardwareMemoryView(host({
      memoryDomains: [{
        id: "system", kind: "unified_memory", totalCapacityBytes: 64 * gib,
        stableCapacityBytes: 60 * gib, currentFreeBytes: 12 * gib,
        sharesSystemMemory: true, backendNames: ["Metal"], deviceNames: ["Apple M4 Max"],
        splitGroupId: null,
      }],
      residentMemory: { modelId: "model", runtimeGeneration: 2, domains: [{
        memoryDomainId: "system", modelBytes: 27 * gib, computeBytes: 1.5 * gib,
        auxiliaryBytes: 0.5 * gib, contextBytes: 6 * gib,
      }] },
    }))
    expect(view.domains[0]).toMatchObject({
      label: "Apple M4 Max · Unified memory",
      fixedBytes: 29 * gib,
      kvCacheBytes: 6 * gib,
      systemAndAppsBytes: 17 * gib,
      freeBytes: 12 * gib,
      status: "complete",
    })
    expect(view.compact).toEqual({ usedBytes: 52 * gib, totalBytes: 64 * gib })
  })

  it("aggregates two participating GPUs without merging their detail blocks", () => {
    const domains = [0, 1].map((index) => ({
      id: `gpu-${index}`, kind: "physical_device" as const, totalCapacityBytes: 24 * gib,
      stableCapacityBytes: 22 * gib, currentFreeBytes: (6 - index * 2) * gib,
      sharesSystemMemory: false, backendNames: ["CUDA"], deviceNames: ["NVIDIA RTX 4090"],
      splitGroupId: null,
    }))
    const view = deriveHardwareMemoryView(host({
      memoryDomains: domains,
      residentMemory: { modelId: "model", runtimeGeneration: 1, domains: [
        { memoryDomainId: "gpu-0", modelBytes: 13 * gib, computeBytes: 0.5 * gib, auxiliaryBytes: 0, contextBytes: 2 * gib },
        { memoryDomainId: "gpu-1", modelBytes: 13 * gib, computeBytes: 1 * gib, auxiliaryBytes: 0, contextBytes: 4 * gib },
      ] },
    }))
    expect(view.domains.map((domain) => domain.label)).toEqual([
      "NVIDIA RTX 4090 · GPU 1",
      "NVIDIA RTX 4090 · GPU 2",
    ])
    expect(view.compact).toEqual({ usedBytes: 38 * gib, totalBytes: 48 * gib })
  })

  it("refuses to fabricate categories when allocation exceeds observed use", () => {
    const view = deriveHardwareMemoryView(host({
      memoryDomains: [{
        id: "gpu", kind: "physical_device", totalCapacityBytes: 24 * gib,
        stableCapacityBytes: 22 * gib, currentFreeBytes: 20 * gib,
        sharesSystemMemory: false, backendNames: ["CUDA"], deviceNames: ["GPU"], splitGroupId: null,
      }],
      residentMemory: { modelId: "model", runtimeGeneration: 1, domains: [{
        memoryDomainId: "gpu", modelBytes: 10 * gib, computeBytes: 0, auxiliaryBytes: 0, contextBytes: 1 * gib,
      }] },
    }))
    expect(view.domains[0]).toMatchObject({
      status: "inconsistent",
      usedBytes: 4 * gib,
      fixedBytes: null,
      systemAndAppsBytes: null,
    })
  })

  it("keeps a high-level accelerator aggregate when attribution is unavailable", () => {
    const view = deriveHardwareMemoryView(host({
      memoryDomains: [{
        id: "gpu", kind: "physical_device", totalCapacityBytes: 24 * gib,
        stableCapacityBytes: 22 * gib, currentFreeBytes: 6 * gib,
        sharesSystemMemory: false, backendNames: ["CUDA"], deviceNames: ["GPU"], splitGroupId: null,
      }, {
        id: "system", kind: "system", totalCapacityBytes: 64 * gib,
        stableCapacityBytes: 60 * gib, currentFreeBytes: 40 * gib,
        sharesSystemMemory: true, backendNames: [], deviceNames: [], splitGroupId: null,
      }],
      residentMemory: null,
    }))
    expect(view.compact).toEqual({ usedBytes: 18 * gib, totalBytes: 24 * gib })
    expect(view.domains[0]?.status).toBe("unavailable")
  })
})
