import { Cause, Context, Duration, Effect, Layer, Schema } from "effect"
import { IcnClient, type IcnClientService } from "../client.js"
import { HardwareSnapshot as HardwareSnapshotSchema } from "../generated/schemas.js"
import {
  makeIcnObservedState,
  type IcnObservedState,
} from "../observed-state.js"

type HardwareReadError = Effect.Effect.Error<ReturnType<IcnClientService["system"]["getHardware"]>>

export interface IcnHardwareService extends IcnObservedState<HardwareSnapshotSchema, HardwareReadError> {}

export class IcnHardware extends Context.Tag("@magnitudedev/icn/IcnHardware")<
  IcnHardware,
  IcnHardwareService
>() {}

export interface IcnHardwareOptions {
  readonly refreshInterval?: Duration.DurationInput
}

export const makeIcnHardware = (
  options: IcnHardwareOptions = {},
): Layer.Layer<IcnHardware, HardwareReadError, IcnClient> =>
  Layer.scoped(
    IcnHardware,
    Effect.gen(function* () {
      const client = yield* IcnClient
      const read = client.system.getHardware({})
      const initial = yield* read
      const observed = yield* makeIcnObservedState(
        initial,
        read,
        Schema.equivalence(HardwareSnapshotSchema),
      )

      yield* observed.refresh.pipe(
        Effect.tapError((error) => Effect.logWarning("Unable to refresh ICN hardware snapshot").pipe(
          Effect.annotateLogs({ cause: Cause.pretty(Cause.fail(error)) }),
        )),
        Effect.option,
        Effect.delay(options.refreshInterval ?? "2 seconds"),
        Effect.forever,
        Effect.forkScoped,
      )

      return IcnHardware.of(observed)
    }),
  )
