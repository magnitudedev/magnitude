import { Data, Effect, Match, Option } from "effect"
import type * as InferenceSchema from "@magnitudedev/icn-protocol/schemas"
import {
  LocalInferenceAcceleratorIdSchema,
  LocalInferenceMemoryDomainIdSchema,
  type LocalInferenceHardware,
} from "./model-state"
import type { AcnHardwareSummary, StartupBackend } from "./acn-health"

export class InferenceHardwareProjectionError extends Data.TaggedError(
  "InferenceHardwareProjectionError",
)<{ readonly message: string }> {}

const genericBackendOrdinal = /^(?:CUDA|MTL|Vulkan|ROCm|SYCL|OpenCL)\d+$/i

export const inferenceAcceleratorDisplayName = (
  device: Pick<InferenceSchema.HardwareDevice, "name" | "description">,
): string => genericBackendOrdinal.test(device.name.trim()) && device.description.trim().length > 0
  ? device.description.trim()
  : device.name

export const projectInferenceHardware = (
  hardware: InferenceSchema.HardwareSnapshot,
): Effect.Effect<LocalInferenceHardware, InferenceHardwareProjectionError> => Effect.gen(function* () {
  const memoryDomains = hardware.memory_domains.map((domain) => ({
    memoryDomainId: LocalInferenceMemoryDomainIdSchema.make(domain.id),
    kind: Match.value(domain.kind).pipe(
      Match.when("unified_memory", () => "UnifiedMemory" as const),
      Match.when("physical_device", () => "PhysicalDevice" as const),
      Match.when("system", () => "System" as const),
      Match.exhaustive,
    ),
    totalBytes: domain.total_capacity_bytes,
    stableCapacityBytes: domain.stable_capacity_bytes,
    availableBytes: Option.flatMap(domain.current_free_bytes, Option.fromNullable),
    sharesSystemMemory: domain.shares_system_memory,
  }))
  // Accelerators are the GPUs a model can run on: a device below its backend's floor carries an
  // unavailable reason and never runs one, so it is not an accelerator.
  const accelerators = hardware.memory_domains.flatMap((domain) => domain.devices
    .filter((device) => device.kind !== "cpu" && Option.isNone(device.unavailable_reason))
    .map((device) => ({
      acceleratorId: LocalInferenceAcceleratorIdSchema.make(device.id),
      name: inferenceAcceleratorDisplayName(device),
      backend: device.backend,
      memoryDomainId: LocalInferenceMemoryDomainIdSchema.make(domain.id),
    })))
  const platform = hardware.platform === "macos"
    ? "MacOS"
    : hardware.platform === "windows"
      ? "Windows"
      : hardware.platform === "linux"
        ? "Linux"
        : yield* new InferenceHardwareProjectionError({ message: `Unsupported ICN platform ${hardware.platform}` })
  const architecture = hardware.architecture === "aarch64" || hardware.architecture === "arm64"
    ? "Arm64"
    : hardware.architecture === "x86_64" || hardware.architecture === "amd64" || hardware.architecture === "x64"
      ? "X64"
      : yield* new InferenceHardwareProjectionError({ message: `Unsupported ICN architecture ${hardware.architecture}` })
  return {
    platform,
    architecture,
    productName: Option.flatMap(hardware.system_product_name, Option.fromNullable),
    processor: Option.flatMap(hardware.cpu_model, Option.fromNullable),
    physicalCores: Option.filter(Option.flatMap(hardware.physical_cores, Option.fromNullable), count => count > 0),
    logicalCores: Math.max(1, hardware.logical_cores),
    totalSystemMemoryBytes: hardware.system_memory.physical_capacity_bytes,
    availableSystemMemoryBytes: hardware.system_memory.physical_available_bytes,
    systemAllocationCapacityBytes: hardware.system_memory.allocation_capacity_bytes,
    systemAllocationHeadroomBytes: hardware.system_memory.allocation_headroom_bytes,
    abortReserveBytes: hardware.system_memory.abort_reserve_bytes,
    accelerators,
    memoryDomains,
  }
})

type SnapshotDevice = { readonly device: InferenceSchema.HardwareDevice; readonly domain: InferenceSchema.HardwareMemoryDomain }
const deviceBytes = ({ device, domain }: SnapshotDevice): number => Option.match(Option.flatMap(device.memory_limit, Option.fromNullable), {
  onNone: () => domain.total_capacity_bytes,
  onSome: limit => limit.total_bytes,
})
const usable = ({ device }: SnapshotDevice): boolean => Option.isNone(device.unavailable_reason)
const startupBackendName = (backend: StartupBackend): AcnHardwareSummary["backend"] =>
  backend._tag === "Cpu" ? "cpu" : backend._tag === "Metal" ? "metal" : backend._tag === "Cuda" ? "cuda" : "vulkan"

/** The machine as one record: the backend the engine runs on, the GPU it runs on if any, and why a present GPU is not used. */
export const summarizeInferenceHardware = (
  loaded: Option.Option<StartupBackend>,
  hardware: InferenceSchema.HardwareSnapshot,
): AcnHardwareSummary => {
  const devices: SnapshotDevice[] = hardware.memory_domains.flatMap(domain => domain.devices
    .filter(device => device.kind !== "cpu")
    .map(device => ({ device, domain })))
  const largestFirst = [...devices].sort((a, b) => deviceBytes(b) - deviceBytes(a))
  const backend = Option.match(loaded, {
    onSome: startupBackendName,
    onNone: () => largestFirst.find(usable)?.device.backend ?? "cpu",
  })
  const used = backend === "cpu" ? undefined : largestFirst.find(entry => entry.device.backend === backend && usable(entry))
  const diagnostics = Option.getOrElse(hardware.backend_diagnostics, () => [])
  const unusedReason = (entry: SnapshotDevice): string => Option.getOrElse(entry.device.unavailable_reason, () => {
    const diagnostic = diagnostics.find(item => item.backend === entry.device.backend)
    if (diagnostic) return `${entry.device.backend}: ${diagnostic.message}`
    if (!hardware.enabled_backends.includes(entry.device.backend)) return `${entry.device.backend}: not available in this build`
    if (entry.device.backend === "vulkan" && entry.device.kind === "integrated_gpu") return "vulkan: integrated GPU is not selected automatically"
    return `${entry.device.backend}: not selected`
  })
  const unused = used ? undefined : largestFirst[0]
  const physical = new Set(devices.map(entry => `${entry.domain.id}|${entry.device.name}`))
  return {
    backend,
    gpuName: Option.fromNullable(used?.device.name).pipe(Option.filter(name => name.length > 0)),
    gpuKind: used ? Option.some(used.device.kind === "integrated_gpu" ? "integrated" : "discrete") : Option.none(),
    gpuMemoryBytes: used ? Option.some(deviceBytes(used)) : Option.none(),
    gpuCount: physical.size,
    gpuUnusedReason: unused ? Option.some(unusedReason(unused)) : Option.none(),
    cpuModel: Option.flatMap(hardware.cpu_model, Option.fromNullable).pipe(Option.filter(model => model.trim().length > 0)),
    cpuCores: Option.flatMap(hardware.physical_cores, Option.fromNullable).pipe(Option.filter(count => count > 0),
      Option.orElse(() => hardware.logical_cores > 0 ? Option.some(hardware.logical_cores) : Option.none())),
    memoryBytes: hardware.system_memory.physical_capacity_bytes,
  }
}
