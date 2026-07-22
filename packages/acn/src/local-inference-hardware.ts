import { Context, Data, Effect, Layer, Match, Option, Schema, Scope, Stream } from "effect"
import {
  LocalInferenceAcceleratorIdSchema,
  LocalInferenceHardwareMirror,
  LocalInferenceMemoryDomainIdSchema,
  type LocalInferenceHardware as LocalInferenceHardwareState,
  type MirroredSnapshot,
} from "@magnitudedev/protocol"
import { IcnHardware, type Generated } from "@magnitudedev/icn"
import { makeMirroredState, MirroredStateChanges } from "./mirrored-state"

export class LocalInferenceHardwareProjectionFailure extends Data.TaggedError("LocalInferenceHardwareProjectionFailure")<{
  readonly message: string
}> {}

export const projectLocalInferenceHardware = (
  hardware: Generated.HardwareSnapshotSchema,
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
      name: device.name,
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
    processor: Option.flatMap(hardware.cpu_model, Option.fromNullable),
    logicalCores: Math.max(1, hardware.logical_cores),
    totalSystemMemoryBytes: hardware.system_memory.total_bytes,
    availableSystemMemoryBytes: Option.flatMap(hardware.system_memory.current_available_bytes, Option.fromNullable),
    accelerators,
    memoryDomains,
    residentMemory: Option.flatMap(hardware.resident_memory, Option.fromNullable).pipe(
      Option.map((resident) => ({ domains: resident.domains.map((domain) => ({
        memoryDomainId: LocalInferenceMemoryDomainIdSchema.make(domain.memory_domain_id),
        modelBytes: domain.model_bytes,
        contextBytes: domain.context_bytes,
        computeBytes: domain.compute_bytes,
        auxiliaryBytes: domain.auxiliary_bytes,
      })) })),
    ),
  }
})

export interface LocalInferenceHardwareApi {
  readonly snapshot: Effect.Effect<MirroredSnapshot<LocalInferenceHardwareState>>
  readonly refresh: Effect.Effect<void>
}

export class LocalInferenceHardware extends Context.Tag("LocalInferenceHardware")<
  LocalInferenceHardware,
  LocalInferenceHardwareApi
>() {}

export const LocalInferenceHardwareLive: Layer.Layer<
  LocalInferenceHardware,
  LocalInferenceHardwareProjectionFailure,
  IcnHardware | MirroredStateChanges
> = Layer.scoped(LocalInferenceHardware, Effect.gen(function* () {
  const hardware = yield* IcnHardware
  const scope = yield* Scope.Scope
  const mirror = yield* makeMirroredState(
    LocalInferenceHardwareMirror,
    yield* projectLocalInferenceHardware((yield* hardware.get).state),
  )
  const rebuild = hardware.get.pipe(
    Effect.flatMap(({ state }) => projectLocalInferenceHardware(state)),
    Effect.flatMap((state) => mirror.setIfChanged(
      state,
      Schema.equivalence(LocalInferenceHardwareMirror.stateSchema),
    )),
  )
  yield* Effect.forkIn(hardware.changes.pipe(
    Stream.runForEach(({ state }) => projectLocalInferenceHardware(state).pipe(
      Effect.flatMap((projected) => mirror.setIfChanged(
        projected,
        Schema.equivalence(LocalInferenceHardwareMirror.stateSchema),
      )),
      Effect.catchAll((error) => Effect.logWarning("Unable to project local inference hardware").pipe(
        Effect.annotateLogs({ cause: error.message }),
      )),
      Effect.asVoid,
    )),
  ), scope)
  return LocalInferenceHardware.of({
    snapshot: mirror.get,
    refresh: hardware.refresh.pipe(
      Effect.zipRight(rebuild),
      Effect.asVoid,
      Effect.catchAll((error) => Effect.logWarning("Unable to refresh local inference hardware").pipe(
        Effect.annotateLogs({ cause: error instanceof Error ? error.message : String(error) }),
      )),
    ),
  })
}))
