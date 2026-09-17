import { Schema } from "effect"
import { LocalInferenceHardwareSchema } from "@magnitudedev/sdk"
import { MachineIdentityObservation } from "@magnitudedev/sdk/desktop-host"
const GiB = 1024 ** 3
export const identity = (manufacturer: string, model: string, formFactor = "Unknown") => Schema.decodeUnknownSync(MachineIdentityObservation)({ _tag: "Identified", manufacturer, model, formFactor })
export const hardware = (processor: string, memory: number, cores: number, accelerators: readonly { name: string; memory: number; shared?: boolean }[] = [], physicalCores?: number) => Schema.decodeUnknownSync(LocalInferenceHardwareSchema)({
  platform: processor.startsWith("Apple") ? "MacOS" : "Linux", architecture: processor.startsWith("Apple") || processor === "NVIDIA GB10" ? "Arm64" : "X64", processor,
  ...(physicalCores === undefined ? {} : { physicalCores }), logicalCores: cores, totalSystemMemoryBytes: memory * GiB, availableSystemMemoryBytes: memory * GiB / 2,
  systemAllocationCapacityBytes: memory * GiB, systemAllocationHeadroomBytes: memory * GiB / 2, abortReserveBytes: GiB,
  accelerators: accelerators.map((gpu, i) => ({ acceleratorId: `gpu-${i}`, name: gpu.name, backend: processor.startsWith("Apple") ? "Metal" : "CUDA", memoryDomainId: gpu.shared ? "system" : `gpu-${i}` })),
  memoryDomains: [{ memoryDomainId: "system", kind: accelerators.some(gpu => gpu.shared) ? "UnifiedMemory" : "System", totalBytes: memory * GiB, stableCapacityBytes: memory * GiB, sharesSystemMemory: true },
    ...accelerators.flatMap((gpu, i) => gpu.shared ? [] : [{ memoryDomainId: `gpu-${i}`, kind: "PhysicalDevice", totalBytes: gpu.memory * GiB, stableCapacityBytes: gpu.memory * GiB, sharesSystemMemory: false }])],
})
export const hardwareScenarios = [
  { label: "CPU inference laptop", identity: identity("Dell Inc.", "XPS 15 9530", "Portable"), value: hardware("13th Gen Intel(R) Core(TM) i7-13700H", 32, 20, [], 14) },
  { label: "Consumer HP laptop", identity: identity("HP", "HP Pavilion Plus Laptop 14-ey1xxx", "Portable"), value: hardware("AMD Ryzen 7 8845HS", 16, 16, [], 8) },
  { label: "Gaming laptop", identity: identity("Dell Inc.", "Dell G15 5530", "Portable"), value: hardware("Intel Core i7-13650HX", 32, 20, [{ name: "NVIDIA GeForce RTX 4060 Laptop GPU", memory: 8 }], 14) },
  { label: "Apple unified memory", identity: identity("Apple Inc.", "Mac16,8"), value: hardware("Apple M4 Pro", 48, 14, [{ name: "Apple M4 Pro", memory: 48, shared: true }], 14) },
  { label: "Two GPU workstation", identity: identity("Custom", "Workstation", "Desktop"), value: hardware("AMD Ryzen 9 9950X", 128, 32, [{ name: "NVIDIA GeForce RTX 5090", memory: 32 }, { name: "NVIDIA GeForce RTX 5090", memory: 32 }]) },
  { label: "DGX Spark", identity: identity("NVIDIA", "NVIDIA_DGX_Spark", "MiniPc"), value: hardware("NVIDIA GB10", 128, 20, [{ name: "NVIDIA GB10", memory: 128, shared: true }], 20) },
  { label: "Strix Halo mini PC", identity: identity("HP", "HP Z2 Mini G1a Workstation Desktop PC", "MiniPc"), value: hardware("AMD Ryzen AI Max+ PRO 395", 128, 32, [{ name: "AMD Radeon 8060S Graphics", memory: 128, shared: true }], 16) },
  { label: "Strix Halo mobile workstation", identity: identity("HP", "HP ZBook Ultra G1a 14 inch Mobile Workstation PC", "Portable"), value: hardware("AMD Ryzen AI Max+ PRO 395", 64, 32, [{ name: "AMD Radeon 8060S Graphics", memory: 64, shared: true }], 16) },
]
