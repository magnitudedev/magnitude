import { Schema } from "effect"
import { LocalInferenceHardwareSchema, type LocalInferenceHardware } from "@magnitudedev/sdk"
import { MachineIdentityObservation } from "@magnitudedev/sdk/desktop-host"
const GiB = 1024 ** 3
export const identity = (manufacturer: string, model: string, formFactor = "Unknown") => Schema.decodeUnknownSync(MachineIdentityObservation)({ _tag: "Identified", manufacturer, model, formFactor })
/** `value` on a machine its firmware names as `named`. */
export const identified = (value: LocalInferenceHardware, named: MachineIdentityObservation): LocalInferenceHardware => ({ ...value, identity: named })
export const hardware = (processor: string, memory: number, logicalCpus: number, accelerators: readonly { name: string; memory: number; shared?: boolean }[] = [], chipCores = logicalCpus) => Schema.decodeUnknownSync(LocalInferenceHardwareSchema)({
  platform: processor.startsWith("Apple") ? "MacOS" : "Linux", architecture: processor.startsWith("Apple") || processor === "NVIDIA GB10" ? "Arm64" : "X64", processor,
  identity: { _tag: "Unavailable", formFactor: "Unknown" }, chipCores, logicalCpus, cpuPoolCores: chipCores, totalSystemMemoryBytes: memory * GiB,
  accelerators: accelerators.map((gpu, i) => ({ acceleratorId: `gpu-${i}`, name: gpu.name, executors: [{ deviceId: `executor-${i}`, backend: processor.startsWith("Apple") ? "metal" : /nvidia|rtx|gb10/i.test(gpu.name) ? "cuda" : "vulkan", memoryDomainIds: [gpu.shared ? "system" : `gpu-${i}`] }] })),
  // The active device is the first GPU's executor, or the host CPU without one; host RAM is unified when it is that GPU's memory.
  activeDeviceId: accelerators.length > 0 ? "executor-0" : "host-cpu",
  memoryDomains: [{ memoryDomainId: "system", kind: accelerators[0]?.shared ? "UnifiedMemory" : "System", totalBytes: memory * GiB },
    ...accelerators.flatMap((gpu, i) => gpu.shared ? [] : [{ memoryDomainId: `gpu-${i}`, kind: "PhysicalDevice", totalBytes: gpu.memory * GiB }])],
})
export const hardwareScenarios = [
  { label: "CPU inference laptop", value: identified(hardware("13th Gen Intel(R) Core(TM) i7-13700H", 32, 20, [], 14), identity("Dell Inc.", "XPS 15 9530", "Portable")) },
  { label: "Consumer HP laptop", value: identified(hardware("AMD Ryzen 7 8845HS", 16, 16, [], 8), identity("HP", "HP Pavilion Plus Laptop 14-ey1xxx", "Portable")) },
  { label: "Gaming laptop", value: identified(hardware("Intel Core i7-13650HX", 32, 20, [{ name: "NVIDIA GeForce RTX 4060 Laptop GPU", memory: 8 }], 14), identity("Dell Inc.", "Dell G15 5530", "Portable")) },
  { label: "Apple unified memory", value: identified(hardware("Apple M4 Pro", 48, 14, [{ name: "Apple M4 Pro", memory: 48, shared: true }], 14), identity("Apple Inc.", "Mac16,8")) },
  { label: "Two GPU workstation", value: identified(hardware("AMD Ryzen 9 9950X", 128, 32, [{ name: "NVIDIA GeForce RTX 5090", memory: 32 }, { name: "NVIDIA GeForce RTX 5090", memory: 32 }]), identity("Custom", "Workstation", "Desktop")) },
  { label: "DGX Spark", value: identified(hardware("NVIDIA GB10", 128, 20, [{ name: "NVIDIA GB10", memory: 128, shared: true }], 20), identity("NVIDIA", "NVIDIA_DGX_Spark", "MiniPc")) },
  { label: "Strix Halo mini PC", value: identified(hardware("AMD Ryzen AI Max+ PRO 395", 128, 32, [{ name: "AMD Radeon 8060S Graphics", memory: 128, shared: true }], 16), identity("HP", "HP Z2 Mini G1a Workstation Desktop PC", "MiniPc")) },
  { label: "Strix Halo mobile workstation", value: identified(hardware("AMD Ryzen AI Max+ PRO 395", 64, 32, [{ name: "AMD Radeon 8060S Graphics", memory: 64, shared: true }], 16), identity("HP", "HP ZBook Ultra G1a 14 inch Mobile Workstation PC", "Portable")) },
]
