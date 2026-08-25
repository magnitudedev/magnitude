import { Context, Effect, Layer, Scope, Stream } from "effect"
import {
  Models,
  inferenceAcceleratorDisplayName,
  projectInferenceHardware,
  type LocalInferenceHardware as LocalInferenceHardwareState,
} from "@magnitudedev/acn-protocol"
import { IcnHardware, IcnInstances } from "@magnitudedev/icn"
import { AcnChanges } from "./changes"

export const acceleratorDisplayName = inferenceAcceleratorDisplayName
export const projectLocalInferenceHardware = projectInferenceHardware

export interface LocalInferenceHardwareApi {
  readonly state: Effect.Effect<LocalInferenceHardwareState>
  readonly changes: Stream.Stream<LocalInferenceHardwareState>
  readonly refresh: Effect.Effect<void>
}

export class LocalInferenceHardware extends Context.Tag("LocalInferenceHardware")<
  LocalInferenceHardware,
  LocalInferenceHardwareApi
>() {}

export const LocalInferenceHardwareLive: Layer.Layer<
  LocalInferenceHardware,
  never,
  IcnHardware | IcnInstances | AcnChanges
> = Layer.scoped(LocalInferenceHardware, Effect.gen(function* () {
  const hardware = yield* IcnHardware
  const instances = yield* IcnInstances
  const changes = yield* AcnChanges
  const scope = yield* Scope.Scope
  const state = hardware.get.pipe(
    Effect.flatMap(({ state }) => projectLocalInferenceHardware(state)),
    Effect.orDie,
  )
  const projectedChanges = hardware.changes.pipe(
    Stream.mapEffect(({ state }) => projectLocalInferenceHardware(state).pipe(
      Effect.tapError((error) => Effect.logWarning("Unable to project local inference hardware").pipe(
        Effect.annotateLogs({ cause: error.message }),
      )),
      Effect.option,
    )),
    Stream.filterMap((state) => state),
  )
  yield* Effect.forkIn(projectedChanges.pipe(
    Stream.runForEach(() => changes.publish({ query: Models.GetLocalEnvironment.name })),
  ), scope)
  // Instance changes can immediately alter available memory. The hardware
  // projection owns invalidating and rebuilding its availability snapshot;
  // model-instance lifecycle consumers do not need to coordinate that concern.
  yield* Effect.forkIn(instances.changes.pipe(
    Stream.drop(1),
    Stream.runForEach(() => hardware.refresh.pipe(Effect.ignore)),
  ), scope)
  return LocalInferenceHardware.of({
    state,
    changes: projectedChanges,
    refresh: hardware.refresh.pipe(
      Effect.asVoid,
      Effect.catchAll((error) => Effect.logWarning("Unable to refresh local inference hardware").pipe(
        Effect.annotateLogs({ cause: error instanceof Error ? error.message : String(error) }),
      )),
    ),
  })
}))
