import { Context, Duration, Effect, Layer } from "effect"
import {
  IcnApiClient,
  type IcnApiClient as IcnApiClientService,
} from "../generated/client.js"
import type { HardwareSnapshotSchema } from "../generated/schemas.js"
import {
  makeIcnObservedState,
  type IcnObservedState,
} from "../observed-state.js"

type HardwareReadError = Effect.Effect.Error<ReturnType<IcnApiClientService["system"]["getHardware"]>>

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
): Layer.Layer<IcnHardware, HardwareReadError, IcnApiClient> =>
  Layer.scoped(
    IcnHardware,
    Effect.gen(function* () {
      const client = yield* IcnApiClient
      const read = client.system.getHardware({})
      const initial = yield* read
      const observed = yield* makeIcnObservedState(initial, read)

      yield* observed.refresh.pipe(
        Effect.catchAll((cause) => Effect.logWarning("Unable to refresh ICN hardware snapshot").pipe(
          Effect.annotateLogs({ cause: String(cause) }),
        )),
        Effect.delay(options.refreshInterval ?? "2 seconds"),
        Effect.forever,
        Effect.forkScoped,
      )

      return IcnHardware.of(observed)
    }),
  )
