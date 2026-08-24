import { Context, Duration, Effect, Layer, Schema } from "effect"
import { IcnClient, type IcnClientService } from "../client.js"
import {
  ModelDownloadsResponse as ModelDownloadsResponseSchema,
} from "@magnitudedev/icn-protocol/schemas"
import { makeIcnObservedState, type IcnObservedState } from "../observed-state.js"
import { IcnEvents, refreshOnIcnEvents } from "../events/index.js"

type DownloadsReadError = Effect.Effect.Error<
  ReturnType<IcnClientService["models"]["listModelDownloads"]>
>

export interface IcnDownloadsService
  extends IcnObservedState<ModelDownloadsResponseSchema, DownloadsReadError> {}

export class IcnDownloads extends Context.Tag("@magnitudedev/icn/IcnDownloads")<
  IcnDownloads,
  IcnDownloadsService
>() {}

export interface IcnDownloadsOptions {
  readonly retryInterval?: Duration.DurationInput
}

export const makeIcnDownloads = (
  options: IcnDownloadsOptions = {},
): Layer.Layer<IcnDownloads, DownloadsReadError, IcnClient | IcnEvents> =>
  Layer.scoped(
    IcnDownloads,
    Effect.gen(function* () {
      const client = yield* IcnClient
      const events = yield* IcnEvents
      const invalidations = yield* events.subscribe
      const read = client.models.listModelDownloads({})
      const initial = yield* read
      const observed = yield* makeIcnObservedState(
        initial,
        read,
        Schema.equivalence(ModelDownloadsResponseSchema),
        { initiallyInitialized: true },
      )
      yield* refreshOnIcnEvents(
        invalidations,
        new Set(["downloads"]),
        observed.refresh,
        "model downloads",
        options.retryInterval ?? "1 second",
      ).pipe(
        Effect.forkScoped,
      )
      return IcnDownloads.of(observed)
    }),
  )
