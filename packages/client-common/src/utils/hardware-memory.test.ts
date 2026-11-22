import { Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { LocalInferenceHardwareSchema, ModelInstanceAllocationSchema } from "@magnitudedev/sdk"
import { deriveHardwareMemoryView, deriveHardwareRoles, memoryDomainBarLayout, participatingMemoryDomains } from "./hardware-memory"

const GiB = 1024 ** 3
const hardware = (available: number | null = 8 * GiB, discrete = false, bothBackends = false) => Schema.decodeUnknownSync(LocalInferenceHardwareSchema)({
  platform: discrete ? "Linux" : "MacOS", architecture: discrete ? "X64" : "Arm64",
  identity: { _tag: "Unavailable", formFactor: "Unknown" }, chipCores: 4, logicalCpus: 8, cpuPoolCores: 4, totalSystemMemoryBytes: 16 * GiB,
  accelerators: [
    { acceleratorId: "gpu", name: "Test GPU", executors: [
      { deviceId: "primary", backend: discrete ? "cuda" : "metal", memoryDomainIds: [discrete ? "gpu" : "system"] },
      ...(bothBackends ? [{ deviceId: "vulkan", backend: "vulkan", memoryDomainIds: ["gpu"] }] : []),
    ] },
  ],
  activeDeviceId: "primary",
  memoryDomains: [
    { memoryDomainId: "system", kind: discrete ? "System" : "UnifiedMemory", totalBytes: 16 * GiB, ...(available === null ? {} : { availableBytes: available }) },
    ...(discrete ? [{ memoryDomainId: "gpu", kind: "PhysicalDevice", totalBytes: 16 * GiB, availableBytes: 8 * GiB }] : []),
  ],
})
const allocation = (domain = "system", modelBytes = 3 * GiB) => Option.some(Schema.decodeUnknownSync(ModelInstanceAllocationSchema)({
  contextWindowTokens: 4096,
  memoryDomains: [{ memoryDomainId: domain, modelBytes, contextBytes: 2 * GiB, computeBytes: GiB / 2, auxiliaryBytes: GiB / 2 }],
  promptGroupPlaces: { held: 1, capacity: 1 },
}))

describe("hardware memory breakdown", () => {
  it("labels a physical GPU once when CUDA and Vulkan share its memory domain", () => {
    const domains = deriveHardwareMemoryView(hardware(8 * GiB, true, true), Option.none()).domains
    expect(domains[1]?.label).toBe("Test GPU · GPU 1")
  })
  it("separates weights from engine buffers without changing the allocation total", () => {
    const domain = deriveHardwareMemoryView(hardware(), allocation()).domains[0]!
    expect(domain).toMatchObject({ weightsBytes: 3.5 * GiB, overheadBytes: GiB / 2, kvCacheBytes: 2 * GiB, systemAndAppsBytes: 2 * GiB, freeBytes: 8 * GiB, status: "complete" })
    expect(domain.weightsBytes! + domain.overheadBytes! + domain.kvCacheBytes! + domain.systemAndAppsBytes! + domain.freeBytes!).toBe(domain.totalBytes)
  })
  it("shows released allocations as zero after stopping", () => {
    expect(deriveHardwareMemoryView(hardware(), Option.none()).domains[0]).toMatchObject({ weightsBytes: 0, overheadBytes: 0, kvCacheBytes: 0, systemAndAppsBytes: 8 * GiB })
  })
  it("does not subtract GPU allocations from system RAM", () => {
    const domains = deriveHardwareMemoryView(hardware(8 * GiB, true), allocation("gpu")).domains
    expect(domains[0]).toMatchObject({ weightsBytes: 0, systemAndAppsBytes: 8 * GiB })
    expect(domains[1]).toMatchObject({ weightsBytes: 3.5 * GiB, systemAndAppsBytes: 2 * GiB })
  })
  it("retains known allocations when free memory cannot be observed", () => {
    expect(deriveHardwareMemoryView(hardware(null), allocation()).domains[0]).toMatchObject({ weightsBytes: 3.5 * GiB, overheadBytes: GiB / 2, kvCacheBytes: 2 * GiB, systemAndAppsBytes: null, freeBytes: null, status: "missing_free" })
  })
  it("does not display an impossible breakdown when observations disagree", () => {
    expect(deriveHardwareMemoryView(hardware(), allocation("system", 12 * GiB)).domains[0]).toMatchObject({ weightsBytes: null, overheadBytes: null, kvCacheBytes: null, systemAndAppsBytes: null, status: "inconsistent" })
  })
  it("keeps the total exact when concurrent samples differ within rounding tolerance", () => {
    const domain = deriveHardwareMemoryView(hardware(10 * GiB + 1024), allocation()).domains[0]!
    expect(domain.status).toBe("rounding_adjusted")
    expect(domain.systemAndAppsBytes).toBe(0)
    expect(domain.usedBytes! + domain.freeBytes!).toBe(16 * GiB)
  })
})

describe("memory domain bar layout", () => {
  it("scales left model segments and right other-apps segment to total capacity with free headroom in the middle", () => {
    const domain = deriveHardwareMemoryView(hardware(), allocation()).domains[0]!
    const layout = memoryDomainBarLayout(domain)
    expect(layout.modelBytes).toBe(6 * GiB)
    expect(layout.leftSegments).toEqual([
      { key: "weights", label: "Weights", bytes: 3.5 * GiB, widthPercent: (3.5 / 16) * 100 },
      { key: "cache", label: "Context", bytes: 2 * GiB, widthPercent: (2 / 16) * 100 },
      { key: "runtime", label: "Overhead", bytes: 0.5 * GiB, widthPercent: (0.5 / 16) * 100 },
    ])
    expect(layout.rightSegment).toEqual({
      key: "other", label: "Other apps", bytes: 2 * GiB, widthPercent: (2 / 16) * 100,
    })
    expect(layout.freeBytes).toBe(8 * GiB)
    const leftPct = layout.leftSegments.reduce((sum, segment) => sum + segment.widthPercent, 0)
    expect(leftPct + (layout.freeBytes! / domain.totalBytes) * 100 + layout.rightSegment!.widthPercent).toBeCloseTo(100)
  })

  it("orders discrete device memory ahead of system memory for participating domains", () => {
    const splitAllocation = Option.some(Schema.decodeUnknownSync(ModelInstanceAllocationSchema)({
      contextWindowTokens: 4096,
      memoryDomains: [
        { memoryDomainId: "system", modelBytes: GiB, contextBytes: 0, computeBytes: 0, auxiliaryBytes: 0 },
        { memoryDomainId: "gpu", modelBytes: 3 * GiB, contextBytes: 2 * GiB, computeBytes: GiB / 2, auxiliaryBytes: GiB / 2 },
      ],
      promptGroupPlaces: { held: 1, capacity: 1 },
    }))
    const view = deriveHardwareMemoryView(hardware(8 * GiB, true), splitAllocation)
    expect(participatingMemoryDomains(view).map(({ id }) => id)).toEqual(["gpu", "system"])
    const gpuOnlyView = deriveHardwareMemoryView(hardware(8 * GiB, true), allocation("gpu"))
    expect(participatingMemoryDomains(gpuOnlyView).map(({ id }) => id)).toEqual(["gpu"])
  })

  it("keeps model segments and omits free/other apps when free memory is unobserved", () => {
    const domain = deriveHardwareMemoryView(hardware(null), allocation()).domains[0]!
    const layout = memoryDomainBarLayout(domain)
    expect(layout.modelBytes).toBe(6 * GiB)
    expect(layout.leftSegments).toHaveLength(3)
    expect(layout.rightSegment).toBeNull()
    expect(layout.freeBytes).toBeNull()
  })

  it("omits per-category segments when observations are inconsistent", () => {
    const domain = deriveHardwareMemoryView(hardware(), allocation("system", 12 * GiB)).domains[0]!
    const layout = memoryDomainBarLayout(domain)
    expect(layout.modelBytes).toBeNull()
    expect(layout.leftSegments).toEqual([])
    expect(layout.rightSegment).toBeNull()
  })
})

describe("hardware roles", () => {
  // An iGPU that can place in host RAM, and a discrete GPU serving. Host RAM is the iGPU's memory, not the serving device's.
  const hybrid = (overrides: Record<string, unknown> = {}) => Schema.decodeUnknownSync(LocalInferenceHardwareSchema)({
    platform: "Windows", architecture: "X64", productName: "Zenbook Pro", processor: "AMD Ryzen 9 7940HS",
    identity: { _tag: "Unavailable", formFactor: "Unknown" }, chipCores: 8, logicalCpus: 16, cpuPoolCores: 8, totalSystemMemoryBytes: 32 * GiB,
    accelerators: [
      { acceleratorId: "igpu", name: "AMD Radeon(TM) Graphics", executors: [{ deviceId: "vulkan:0", backend: "vulkan", memoryDomainIds: ["system"] }] },
      { acceleratorId: "dgpu", name: "NVIDIA GeForce RTX 4060", executors: [
        { deviceId: "cuda:0", backend: "cuda", memoryDomainIds: ["vram"] },
        { deviceId: "vulkan:1", backend: "vulkan", memoryDomainIds: ["vram", "system"] },
      ] },
    ],
    activeDeviceId: "cuda:0",
    memoryDomains: [
      { memoryDomainId: "system", kind: "System", totalBytes: 32 * GiB, availableBytes: 16 * GiB },
      { memoryDomainId: "vram", kind: "PhysicalDevice", totalBytes: 8 * GiB },
    ],
    ...overrides,
  })

  it("titles a hybrid machine by its product and CPU, leads with the serving dGPU, and keeps host RAM as system memory", () => {
    const roles = deriveHardwareRoles(hybrid())
    expect(roles.machineName).toBe("Zenbook Pro · AMD Ryzen 9 7940HS")
    expect(roles.unifiedMemory).toBe(false)
    expect(roles.devices.map(({ accelerator, role }) => [accelerator.acceleratorId, role])).toEqual([
      ["dgpu", { _tag: "InUse", backend: "cuda" }],
      ["igpu", { _tag: "Available" }],
    ])
    expect(roles.devices.map(({ dedicatedMemoryBytes, placesOnHost }) => [dedicatedMemoryBytes, placesOnHost]))
      .toEqual([[Option.some(8 * GiB), true], [Option.none(), true]])
    expect(deriveHardwareMemoryView(hybrid(), Option.none()).domains.map(({ label }) => label))
      .toEqual(["System memory", "NVIDIA GeForce RTX 4060 · GPU 1"])
  })
  it("titles a machine without a product name by its CPU, never an accelerator", () => {
    expect(deriveHardwareRoles({ ...hybrid(), productName: Option.none() }).machineName).toBe("AMD Ryzen 9 7940HS")
  })
  it("marks every device available while the CPU serves", () => {
    expect(deriveHardwareRoles(hybrid({ activeDeviceId: "host-cpu" })).devices.map(({ role }) => role._tag))
      .toEqual(["Available", "Available"])
  })
  it("keeps Apple Silicon's host RAM as unified memory", () => {
    const roles = deriveHardwareRoles(hardware())
    expect(roles.unifiedMemory).toBe(true)
    expect(roles.devices.map(({ role }) => role)).toEqual([{ _tag: "InUse", backend: "metal" }])
    expect(deriveHardwareMemoryView(hardware(), Option.none()).domains[0]?.label).toBe("Unified memory")
  })
})
