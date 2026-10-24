import { Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { LocalInferenceHardwareSchema, ModelInstanceAllocationSchema } from "@magnitudedev/sdk"
import { deriveHardwareMemoryView, deriveHardwareRoles } from "./hardware-memory"

const GiB = 1024 ** 3
const hardware = (available: number | null = 8 * GiB, discrete = false, bothBackends = false) => Schema.decodeUnknownSync(LocalInferenceHardwareSchema)({
  platform: discrete ? "Linux" : "MacOS", architecture: discrete ? "X64" : "Arm64",
  logicalCores: 8, totalSystemMemoryBytes: 16 * GiB,
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
}))

describe("hardware memory breakdown", () => {
  it("labels a physical GPU once when CUDA and Vulkan share its memory domain", () => {
    const domains = deriveHardwareMemoryView(hardware(8 * GiB, true, true), Option.none()).domains
    expect(domains[1]?.label).toBe("Test GPU · GPU 1")
  })
  it("separates weights from engine buffers without changing the allocation total", () => {
    const domain = deriveHardwareMemoryView(hardware(), allocation()).domains[0]!
    expect(domain).toMatchObject({ modelBytes: 3 * GiB, overheadBytes: GiB, fixedBytes: 4 * GiB, kvCacheBytes: 2 * GiB, systemAndAppsBytes: 2 * GiB, freeBytes: 8 * GiB, status: "complete" })
    expect(domain.modelBytes! + domain.overheadBytes! + domain.kvCacheBytes! + domain.systemAndAppsBytes! + domain.freeBytes!).toBe(domain.totalBytes)
  })
  it("shows released allocations as zero after stopping", () => {
    expect(deriveHardwareMemoryView(hardware(), Option.none()).domains[0]).toMatchObject({ modelBytes: 0, overheadBytes: 0, kvCacheBytes: 0, systemAndAppsBytes: 8 * GiB })
  })
  it("does not subtract GPU allocations from system RAM", () => {
    const domains = deriveHardwareMemoryView(hardware(8 * GiB, true), allocation("gpu")).domains
    expect(domains[0]).toMatchObject({ modelBytes: 0, systemAndAppsBytes: 8 * GiB })
    expect(domains[1]).toMatchObject({ modelBytes: 3 * GiB, systemAndAppsBytes: 2 * GiB })
  })
  it("retains known allocations when free memory cannot be observed", () => {
    expect(deriveHardwareMemoryView(hardware(null), allocation()).domains[0]).toMatchObject({ modelBytes: 3 * GiB, overheadBytes: GiB, kvCacheBytes: 2 * GiB, systemAndAppsBytes: null, freeBytes: null, status: "missing_free" })
  })
  it("does not display an impossible breakdown when observations disagree", () => {
    expect(deriveHardwareMemoryView(hardware(), allocation("system", 12 * GiB)).domains[0]).toMatchObject({ modelBytes: null, overheadBytes: null, kvCacheBytes: null, systemAndAppsBytes: null, status: "inconsistent" })
  })
  it("keeps the total exact when concurrent samples differ within rounding tolerance", () => {
    const domain = deriveHardwareMemoryView(hardware(10 * GiB + 1024), allocation()).domains[0]!
    expect(domain.status).toBe("rounding_adjusted")
    expect(domain.systemAndAppsBytes).toBe(0)
    expect(domain.usedBytes! + domain.freeBytes!).toBe(16 * GiB)
  })
})

describe("hardware roles", () => {
  // An iGPU that can place in host RAM, and a discrete GPU serving. Host RAM is the iGPU's memory, not the serving device's.
  const hybrid = (overrides: Record<string, unknown> = {}) => Schema.decodeUnknownSync(LocalInferenceHardwareSchema)({
    platform: "Windows", architecture: "X64", productName: "Zenbook Pro", processor: "AMD Ryzen 9 7940HS",
    logicalCores: 16, totalSystemMemoryBytes: 32 * GiB,
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
