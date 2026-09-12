import { Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { LocalInferenceHardwareSchema, ModelInstanceAllocationSchema } from "@magnitudedev/sdk"
import { deriveHardwareMemoryView } from "./hardware-memory"

const GiB = 1024 ** 3
const hardware = (available: number | null = 8 * GiB, discrete = false) => Schema.decodeUnknownSync(LocalInferenceHardwareSchema)({
  platform: discrete ? "Linux" : "MacOS", architecture: discrete ? "X64" : "Arm64",
  logicalCores: 8, totalSystemMemoryBytes: 16 * GiB, availableSystemMemoryBytes: 8 * GiB,
  systemAllocationCapacityBytes: 14 * GiB, systemAllocationHeadroomBytes: 6 * GiB, abortReserveBytes: GiB,
  accelerators: [{ acceleratorId: "gpu", name: "Test GPU", backend: discrete ? "CUDA" : "Metal", memoryDomainId: discrete ? "gpu" : "system" }],
  memoryDomains: [
    { memoryDomainId: "system", kind: discrete ? "System" : "UnifiedMemory", totalBytes: 16 * GiB, stableCapacityBytes: 14 * GiB, sharesSystemMemory: true, ...(available === null ? {} : { availableBytes: available }) },
    ...(discrete ? [{ memoryDomainId: "gpu", kind: "PhysicalDevice", totalBytes: 16 * GiB, stableCapacityBytes: 16 * GiB, availableBytes: 8 * GiB, sharesSystemMemory: false }] : []),
  ],
})
const allocation = (domain = "system", modelBytes = 3 * GiB) => Option.some(Schema.decodeUnknownSync(ModelInstanceAllocationSchema)({
  contextWindowTokens: 4096, parallelSequences: 1, physicalContextTokens: 4096,
  memoryDomains: [{ memoryDomainId: domain, modelBytes, contextBytes: 2 * GiB, computeBytes: GiB / 2, auxiliaryBytes: GiB / 2 }],
}))

describe("hardware memory breakdown", () => {
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
