import { Data, Effect, Option } from "effect"
import type * as InferenceSchema from "@magnitudedev/icn-protocol/schemas"
import {
  LocalInferenceAcceleratorIdSchema,
  LocalInferenceDeviceIdSchema,
  LocalInferenceMemoryDomainIdSchema,
  type LocalInferenceHardware,
} from "./model-state"
import type { AcnHardwareSummary } from "./acn-health"

export class InferenceHardwareProjectionError extends Data.TaggedError(
  "InferenceHardwareProjectionError",
)<{ readonly message: string }> {}

const genericBackendOrdinal = /^(?:CUDA|MTL|Vulkan|ROCm|SYCL|OpenCL)\d+$/i

export const inferenceAcceleratorDisplayName = (
  device: Pick<InferenceSchema.HardwareExecutor, "name" | "description">,
): string => genericBackendOrdinal.test(device.name.trim()) && device.description.trim().length > 0
  ? device.description.trim()
  : device.name

type AccessPath = InferenceSchema.HardwareAccessPath

/**
 * Every executor's device placement, each index resolved to its access path. Device storage is
 * placed only in a memory domain (host RAM or a device's own memory), so a placement fails on an
 * index that names no path, a path of another executor, or a path to any other store.
 */
const resolvePlacements = (
  hardware: InferenceSchema.HardwareSnapshot,
): Effect.Effect<ReadonlyMap<string, readonly AccessPath[]>, InferenceHardwareProjectionError> => {
  const paths = new Map(hardware.access_paths.map(path => [path.id, path]))
  const memoryDomains = new Set([hardware.host_ram.store, ...hardware.device_memory.map(memory => memory.store)]
    .map(store => store.id))
  const executors = hardware.physical_devices.flatMap(physical => physical.executors)
  return Effect.forEach(executors, executor => Effect.forEach(executor.device_placement, index => {
    const path = paths.get(index)
    return path && path.executor === executor.id && memoryDomains.has(path.store)
      ? Effect.succeed(path)
      : Effect.fail(new InferenceHardwareProjectionError({
          message: `Executor ${executor.id} references invalid placement path ${index}`,
        }))
  }).pipe(Effect.map(placement => [executor.id, placement] as const))).pipe(Effect.map(entries => new Map(entries)))
}

type SnapshotExecutor = { readonly executor: InferenceSchema.HardwareExecutor; readonly physical: InferenceSchema.HardwarePhysicalDevice }

const snapshotExecutors = (hardware: InferenceSchema.HardwareSnapshot): readonly SnapshotExecutor[] =>
  hardware.physical_devices.flatMap(physical => physical.executors.map(executor => ({ executor, physical })))

/**
 * The active device: the device a resident (Ready) model runs on, and otherwise the service's
 * serving device. A device naming no reported executor breaks ICN's contract.
 */
const resolveActive = (
  hardware: InferenceSchema.HardwareSnapshot,
  instances: InferenceSchema.ModelInstancesSnapshot,
): Effect.Effect<SnapshotExecutor, InferenceHardwareProjectionError> => {
  const id = instances.instances.flatMap(({ lifecycle }) => lifecycle._tag === "Ready" ? [lifecycle.device.id] : [])[0]
    ?? hardware.selected_executor
  return Option.match(Option.fromNullable(snapshotExecutors(hardware).find(({ executor }) => executor.id === id)), {
    onNone: () => Effect.fail(new InferenceHardwareProjectionError({ message: `Active device ${id} is not a reported executor` })),
    onSome: Effect.succeed,
  })
}

export const projectInferenceHardware = (
  hardware: InferenceSchema.HardwareSnapshot,
  instances: InferenceSchema.ModelInstancesSnapshot,
): Effect.Effect<LocalInferenceHardware, InferenceHardwareProjectionError> => Effect.gen(function* () {
  const host = hardware.host_ram
  const placements = yield* resolvePlacements(hardware)
  const active = yield* resolveActive(hardware, instances)
  const accelerators = hardware.physical_devices.flatMap(physical => {
    const executors = physical.executors.filter(executor => executor.availability.type === "available")
    const [first] = executors
    return physical.kind === "cpu" || first === undefined ? [] : [{
      acceleratorId: LocalInferenceAcceleratorIdSchema.make(physical.id),
      name: inferenceAcceleratorDisplayName(first),
      executors: executors.map(executor => ({
        deviceId: LocalInferenceDeviceIdSchema.make(executor.id),
        backend: executor.backend,
        memoryDomainIds: [...new Set(placements.get(executor.id)!.map(path =>
          LocalInferenceMemoryDomainIdSchema.make(path.store)))],
      })),
    }]
  })
  const memoryDomains = [
    {
      memoryDomainId: LocalInferenceMemoryDomainIdSchema.make(host.store.id),
      // Host RAM is unified only when it is the active device's own memory: a GPU is integrated
      // exactly when its memory is the host's.
      kind: active.physical.kind === "integrated_gpu" ? "UnifiedMemory" as const : "System" as const,
      totalBytes: host.store.total_capacity_bytes,
      availableBytes: Option.some(host.current_free_bytes),
    },
    ...hardware.device_memory.map(memory => ({
      memoryDomainId: LocalInferenceMemoryDomainIdSchema.make(memory.store.id),
      kind: "PhysicalDevice" as const,
      totalBytes: memory.store.total_capacity_bytes,
      availableBytes: Option.flatMap(memory.current_free_bytes, Option.fromNullable),
    })),
  ]
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
    totalSystemMemoryBytes: host.store.total_capacity_bytes,
    accelerators,
    activeDeviceId: LocalInferenceDeviceIdSchema.make(active.executor.id),
    memoryDomains,
  }
})


/**
 * The memory a GPU has, as its driver states it: its own memory store's total, and for host RAM
 * its placement reaches, what the paths' grants let it place there (host RAM's total where a path
 * is ungranted). An executor without a placement (unavailable) has its physical device's own
 * memory, or host RAM's total for a GPU whose memory is the host's.
 */
const deviceBytes = (
  hardware: InferenceSchema.HardwareSnapshot,
  placements: ReadonlyMap<string, readonly AccessPath[]>,
) => {
  const host = hardware.host_ram.store
  return ({ executor, physical }: SnapshotExecutor): number => {
    const placed = placements.get(executor.id)!
    if (placed.length === 0) {
      return hardware.device_memory.find(memory => memory.physical_device_id === physical.id)?.store.total_capacity_bytes
        ?? host.total_capacity_bytes
    }
    const stores = new Set(placed.map(path => path.store))
    const own = hardware.device_memory.filter(memory => stores.has(memory.store.id))
      .reduce((total, memory) => total + memory.store.total_capacity_bytes, 0)
    const onHost = placed.filter(path => path.store === host.id).reduce((total, path) =>
      total + Option.match(Option.flatMap(path.grant, Option.fromNullable), {
        onNone: () => host.total_capacity_bytes,
        onSome: grant => grant.bytes,
      }), 0)
    return own + Math.min(onHost, host.total_capacity_bytes)
  }
}

/** The machine as one record: the active device's backend, its GPU if it is one, and why a present GPU is not used. */
export const summarizeInferenceHardware = (
  hardware: InferenceSchema.HardwareSnapshot,
  instances: InferenceSchema.ModelInstancesSnapshot,
): Effect.Effect<AcnHardwareSummary, InferenceHardwareProjectionError> => Effect.gen(function* () {
  const bytes = deviceBytes(hardware, yield* resolvePlacements(hardware))
  const active = yield* resolveActive(hardware, instances)
  const used = active.physical.kind === "cpu" ? undefined : active
  const largestFirst = snapshotExecutors(hardware).filter(({ physical }) => physical.kind !== "cpu")
    .sort((a, b) => bytes(b) - bytes(a))
  const diagnostics = Option.getOrElse(hardware.backend_diagnostics, () => [])
  const unusedReason = ({ executor }: SnapshotExecutor): string => {
    if (executor.availability.type === "unavailable") return executor.availability.reason
    const diagnostic = diagnostics.find(item => item.source.type === "backend" && item.source.backend === executor.backend)
    if (diagnostic) return `${executor.backend}: ${diagnostic.message}`
    return `${executor.backend}: not selected`
  }
  const unused = used ? undefined : largestFirst[0]
  return {
    backend: active.executor.backend,
    gpuName: Option.fromNullable(used?.executor.name).pipe(Option.filter(name => name.length > 0)),
    gpuKind: used ? Option.some(used.physical.kind === "integrated_gpu" ? "integrated" : "discrete") : Option.none(),
    gpuMemoryBytes: used ? Option.some(bytes(used)) : Option.none(),
    gpuCount: hardware.physical_devices.filter(physical => physical.kind !== "cpu").length,
    gpuUnusedReason: unused ? Option.some(unusedReason(unused)) : Option.none(),
    cpuModel: Option.flatMap(hardware.cpu_model, Option.fromNullable).pipe(Option.filter(model => model.trim().length > 0)),
    cpuCores: Option.flatMap(hardware.physical_cores, Option.fromNullable).pipe(Option.filter(count => count > 0),
      Option.orElse(() => hardware.logical_cores > 0 ? Option.some(hardware.logical_cores) : Option.none())),
    memoryBytes: hardware.host_ram.store.total_capacity_bytes,
  } satisfies AcnHardwareSummary
})
