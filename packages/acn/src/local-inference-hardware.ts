import { Context, Effect, Layer, Schema, Scope, Stream } from "effect"
import {
  InferenceHardwareProjectionError,
  LocalInferenceHardwareSchema,
  inferenceAcceleratorDisplayName,
  projectInferenceHardware,
  type LocalInferenceHardware as LocalInferenceHardwareState,
  type MirroredSnapshot,
} from "@magnitudedev/acn-protocol"
import { IcnHardware, IcnInstances } from "@magnitudedev/icn"
import { makeObservedState } from "./mirrored-state"

export const acceleratorDisplayName = inferenceAcceleratorDisplayName
export const projectLocalInferenceHardware = projectInferenceHardware

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
  InferenceHardwareProjectionError,
  IcnHardware | IcnInstances
> = Layer.scoped(LocalInferenceHardware, Effect.gen(function* () {
  const hardware = yield* IcnHardware
  const instances = yield* IcnInstances
  const scope = yield* Scope.Scope
  const mirror = yield* makeObservedState(
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
