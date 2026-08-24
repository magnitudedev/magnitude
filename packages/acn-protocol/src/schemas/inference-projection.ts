import { Data, Effect, Match, Option } from "effect"
import type * as InferenceSchema from "@magnitudedev/icn-protocol/schemas"
import {
  LocalInferenceAcceleratorIdSchema,
  LocalInferenceMemoryDomainIdSchema,
  type LocalInferenceHardware,
} from "./model-state"

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
  const accelerators = hardware.memory_domains.flatMap((domain) => domain.devices
    .filter((device) => device.kind !== "cpu")
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
