import { describe, expect, it } from "vitest"
import { Schema } from "effect"
import { LocalInferenceHardwareSchema, type LocalInferenceHardware } from "@magnitudedev/acn-protocol"
import { acceleratorDisplayName, sameEnvironmentForClients } from "./local-inference-hardware"

describe("local inference hardware projection", () => {
  it("uses native device descriptions instead of generic backend ordinals", () => {
    expect(acceleratorDisplayName({
      name: "CUDA0",
      description: "NVIDIA GB10",
    })).toBe("NVIDIA GB10")
    expect(acceleratorDisplayName({
      name: "MTL0",
      description: "Apple M4 Max",
    })).toBe("Apple M4 Max")
  })

  it("preserves a native device name that already identifies the hardware", () => {
    expect(acceleratorDisplayName({
      name: "NVIDIA RTX 5090",
      description: "NVIDIA RTX 5090",
    })).toBe("NVIDIA RTX 5090")
  })
})

const GiB = 1024 ** 3
const snapshot = (freeMemory: number, freeDisk: number, logicalCpus = 8): LocalInferenceHardware =>
  Schema.decodeUnknownSync(LocalInferenceHardwareSchema)({
    platform: "Linux",
    architecture: "X64",
    chipCores: 4,
    logicalCpus,
    cpuPoolCores: 4,
    totalSystemMemoryBytes: 32 * GiB,
    accelerators: [{ acceleratorId: "gpu", name: "Test GPU", executors: [{ deviceId: "primary", backend: "cuda", memoryDomainIds: ["cuda-0"] }] }],
    activeDeviceId: "primary",
    memoryDomains: [
      { memoryDomainId: "system", kind: "System", totalBytes: 32 * GiB, availableBytes: freeMemory },
      { memoryDomainId: "cuda-0", kind: "PhysicalDevice", totalBytes: 48 * GiB, availableBytes: 40 * GiB },
    ],
    modelStorage: { freeBytes: freeDisk },
  })

describe("hardware changes told to clients", () => {
  it("are no change when only free memory moved, as it does with every snapshot", () => {
    expect(sameEnvironmentForClients(snapshot(12 * GiB, 70 * GiB), snapshot(12 * GiB + 4096, 70 * GiB))).toBe(true)
  })

  it("are a change when anything else moved", () => {
    expect(sameEnvironmentForClients(snapshot(12 * GiB, 70 * GiB), snapshot(12 * GiB, 60 * GiB))).toBe(false)
    expect(sameEnvironmentForClients(snapshot(12 * GiB, 70 * GiB), snapshot(12 * GiB, 70 * GiB, 16))).toBe(false)
  })
})
