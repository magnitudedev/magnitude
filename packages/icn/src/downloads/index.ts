import { Cause, Context, Duration, Effect, Layer, Queue, Schema } from "effect"
import { IcnClient, type IcnClientService } from "../client.js"
import { ModelDownloadsResponse as ModelDownloadsResponseSchema } from "../generated/schemas.js"
import { makeIcnObservedState, type IcnObservedState } from "../observed-state.js"

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
  readonly refreshInterval?: Duration.DurationInput
}

export const makeIcnDownloads = (
  options: IcnDownloadsOptions = {},
): Layer.Layer<IcnDownloads, DownloadsReadError, IcnClient> =>
  Layer.scoped(
    IcnDownloads,
    Effect.gen(function* () {
      const client = yield* IcnClient
      const read = client.models.listModelDownloads({})
      const initial = yield* read
      const observed = yield* makeIcnObservedState(
        initial,
        read,
        Schema.equivalence(ModelDownloadsResponseSchema),
      )
      const wake = yield* Queue.sliding<void>(1)
      const hasActiveAttempt = observed.get.pipe(Effect.map(({ state }) =>
        state.attempts.some((attempt) =>
          attempt._tag === "Pending" || attempt._tag === "Downloading")))
      const poll = Effect.gen(function* () {
        yield* Queue.take(wake)
        while (yield* hasActiveAttempt) {
          yield* Effect.sleep(options.refreshInterval ?? "1 second")
          yield* observed.refresh.pipe(
            Effect.tapError((error) => Effect.logWarning("Unable to refresh model download attempts").pipe(
              Effect.annotateLogs({ cause: Cause.pretty(Cause.fail(error)) }),
            )),
            Effect.option,
          )
        }
      })
      yield* poll.pipe(
        Effect.forever,
        Effect.forkScoped,
      )
      if (yield* hasActiveAttempt) yield* Queue.offer(wake, undefined)
      const refresh = observed.refresh.pipe(
        Effect.tap(() => hasActiveAttempt.pipe(
          Effect.flatMap((active) => active ? Queue.offer(wake, undefined) : Effect.void),
        )),
      )
      return IcnDownloads.of({ ...observed, refresh })
    }),
  )
