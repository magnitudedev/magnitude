import { Context, Data, Effect, Layer, Match, Option, Schema, Scope, Stream } from "effect"
import {
  LocalInference,
  LocalInferenceAcceleratorIdSchema,
  LocalInferenceHardwareSchema,
  LocalInferenceMemoryDomainIdSchema,
  type LocalInferenceHardware as LocalInferenceHardwareState,
  type MirroredSnapshot,
} from "@magnitudedev/acn-protocol"
import type * as Generated from "@magnitudedev/icn-protocol/schemas"
import { IcnHardware, IcnInstances } from "@magnitudedev/icn"
import { AcnChanges } from "./changes"
import { makeMirroredState } from "./mirrored-state"

export class LocalInferenceHardwareProjectionFailure extends Data.TaggedError("LocalInferenceHardwareProjectionFailure")<{
  readonly message: string
}> {}

const genericBackendOrdinal = /^(?:CUDA|MTL|Vulkan|ROCm|SYCL|OpenCL)\d+$/i

export const acceleratorDisplayName = (
  device: Pick<Generated.HardwareDevice, "name" | "description">,
): string => genericBackendOrdinal.test(device.name.trim())
    && device.description.trim().length > 0
    ? device.description.trim()
    : device.name

export const projectLocalInferenceHardware = (
  hardware: Generated.HardwareSnapshot,
): Effect.Effect<LocalInferenceHardwareState, LocalInferenceHardwareProjectionFailure> => Effect.gen(function* () {
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
      name: acceleratorDisplayName(device),
      backend: device.backend,
      memoryDomainId: LocalInferenceMemoryDomainIdSchema.make(domain.id),
    })))
  const platform = hardware.platform === "macos"
      ? "MacOS"
      : hardware.platform === "windows"
        ? "Windows"
        : hardware.platform === "linux"
          ? "Linux"
          : yield* new LocalInferenceHardwareProjectionFailure({
              message: `Unsupported ICN platform ${hardware.platform}`,
            })
  const architecture = hardware.architecture === "aarch64" || hardware.architecture === "arm64"
      ? "Arm64"
      : hardware.architecture === "x86_64" || hardware.architecture === "amd64" || hardware.architecture === "x64"
        ? "X64"
        : yield* new LocalInferenceHardwareProjectionFailure({
            message: `Unsupported ICN architecture ${hardware.architecture}`,
          })
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

export interface LocalInferenceHardwareApi {
  readonly snapshot: Effect.Effect<MirroredSnapshot<LocalInferenceHardwareState>>
  readonly changes: Stream.Stream<MirroredSnapshot<LocalInferenceHardwareState>>
  readonly refresh: Effect.Effect<void>
}

export class LocalInferenceHardware extends Context.Tag("LocalInferenceHardware")<
  LocalInferenceHardware,
  LocalInferenceHardwareApi
>() {}

export const LocalInferenceHardwareLive: Layer.Layer<
  LocalInferenceHardware,
  LocalInferenceHardwareProjectionFailure,
  IcnHardware | IcnInstances | AcnChanges
> = Layer.scoped(LocalInferenceHardware, Effect.gen(function* () {
  const hardware = yield* IcnHardware
  const instances = yield* IcnInstances
  const scope = yield* Scope.Scope
  const mirror = yield* makeMirroredState(
    { name: LocalInference.GetLocalInferenceHardware.name },
    yield* projectLocalInferenceHardware((yield* hardware.get).state),
  )
  const rebuild = hardware.get.pipe(
    Effect.flatMap(({ state }) => projectLocalInferenceHardware(state)),
    Effect.flatMap((state) => mirror.setIfChanged(
      state,
      Schema.equivalence(LocalInferenceHardwareSchema),
    )),
  )
  yield* Effect.forkIn(hardware.changes.pipe(
    Stream.runForEach(({ state }) => projectLocalInferenceHardware(state).pipe(
      Effect.flatMap((projected) => mirror.setIfChanged(
        projected,
        Schema.equivalence(LocalInferenceHardwareSchema),
      )),
      Effect.catchAll((error) => Effect.logWarning("Unable to project local inference hardware").pipe(
        Effect.annotateLogs({ cause: error.message }),
      )),
      Effect.asVoid,
    )),
  ), scope)
  // Instance changes can immediately alter available memory. The hardware
  // projection owns invalidating and rebuilding its availability snapshot;
  // model-instance lifecycle consumers do not need to coordinate that concern.
  yield* Effect.forkIn(instances.changes.pipe(
    Stream.drop(1),
    Stream.runForEach(() => hardware.refresh.pipe(Effect.ignore)),
  ), scope)
  return LocalInferenceHardware.of({
    snapshot: mirror.get,
    changes: mirror.changes,
    refresh: hardware.refresh.pipe(
      Effect.zipRight(rebuild),
      Effect.asVoid,
      Effect.catchAll((error) => Effect.logWarning("Unable to refresh local inference hardware").pipe(
        Effect.annotateLogs({ cause: error instanceof Error ? error.message : String(error) }),
      )),
    ),
  })
}))
