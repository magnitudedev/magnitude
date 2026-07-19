import { Option } from "effect"
import { describe, expect, test } from "vitest"
import { LlamaCpp } from "@magnitudedev/local-inference"
import type { HostHardwareSnapshot } from "@magnitudedev/local-inference/hardware"
import { detectedInferenceCapacity, hostToWire } from "./service"

const GIB = 1024 ** 3

const host = (
  platform: HostHardwareSnapshot["platform"],
  architecture: string,
  memoryGiB: number,
): HostHardwareSnapshot => ({
  capturedAt: new Date(0),
  platform,
  processArchitecture: architecture,
  nativeArchitecture: architecture,
  cpuModel: Option.none(),
  logicalCores: 12,
  totalMemoryBytes: memoryGiB * GIB,
  availableMemoryBytes: GIB,
})

const device = (input: {
  readonly id: string
  readonly backend: string
  readonly type: string
  readonly totalGiB: number
  readonly physicalId?: string
}): LlamaCpp.LlamaDevice => ({
  id: LlamaCpp.LlamaDeviceId.make(input.id),
  name: Option.some(input.id),
  backend: Option.some(input.backend),
  type: Option.some(input.type),
  physicalId: Option.fromNullable(input.physicalId),
  totalMemoryBytes: Option.some(input.totalGiB * GIB),
  freeMemoryBytes: Option.some(GIB),
})

describe("llama.cpp hardware topology", () => {
  test("aliases Apple Metal capacity to the unified system-memory domain", () => {
    const machine = host("darwin", "arm64", 64)
    const devices = [
      device({ id: "Metal0", backend: "Metal", type: "IGPU", totalGiB: 48 }),
      device({ id: "Accelerate", backend: "BLAS", type: "ACCEL", totalGiB: 0 }),
    ]
    const capacity = detectedInferenceCapacity(machine, devices)

    expect(capacity.systemMemoryBytes).toBe(64 * GIB)
    expect(capacity.acceleratorDomains).toEqual([{
      memoryDomainId: "unified:Metal:Metal0",
      capacityBytes: expect.closeTo(43.2 * GIB, 0),
      sharesSystemMemory: true,
      preferredBackend: "Metal",
    }])
    expect(hostToWire(machine, devices).memoryDomains[1]).toMatchObject({
      kind: "unified_working_set",
      totalCapacityBytes: 48 * GIB,
      sharesSystemMemory: true,
      deviceNames: ["Metal0"],
    })
    expect(hostToWire(machine, devices).memoryDomains).toHaveLength(2)
  })

  test("keeps discrete CUDA memory separate from Linux system memory", () => {
    const machine = host("linux", "x64", 64)
    const devices = [device({
      id: "CUDA0",
      backend: "CUDA",
      type: "GPU",
      totalGiB: 24,
      physicalId: "0000:01:00.0",
    })]
    const capacity = detectedInferenceCapacity(machine, devices)

    expect(capacity.acceleratorDomains).toEqual([{
      memoryDomainId: "physical:0000:01:00.0",
      capacityBytes: expect.closeTo(21.6 * GIB, 0),
      sharesSystemMemory: false,
      preferredBackend: "CUDA",
    }])
  })

  test("marks multiple devices from one llama.cpp backend as a supported split group", () => {
    const machine = host("linux", "x64", 128)
    const devices = [
      device({ id: "CUDA0", backend: "CUDA", type: "GPU", totalGiB: 24, physicalId: "0000:01:00.0" }),
      device({ id: "CUDA1", backend: "CUDA", type: "GPU", totalGiB: 24, physicalId: "0000:02:00.0" }),
    ]

    expect(detectedInferenceCapacity(machine, devices).acceleratorDomains).toEqual([
      expect.objectContaining({ modelSplitGroupId: "llamacpp:cuda" }),
      expect.objectContaining({ modelSplitGroupId: "llamacpp:cuda" }),
    ])
  })

  test("does not double count one GPU exposed through CUDA and Vulkan", () => {
    const machine = host("linux", "x64", 64)
    const devices = [
      device({ id: "CUDA0", backend: "CUDA", type: "GPU", totalGiB: 24 }),
      device({ id: "Vulkan0", backend: "Vulkan", type: "GPU", totalGiB: 24 }),
    ].map((visible) => ({ ...visible, name: Option.some("NVIDIA RTX 4090") }))

    expect(detectedInferenceCapacity(machine, devices).acceleratorDomains).toEqual([{
      memoryDomainId: "physical:CUDA:CUDA0",
      capacityBytes: expect.closeTo(21.6 * GIB, 0),
      sharesSystemMemory: false,
      preferredBackend: "CUDA",
    }])
  })
})
