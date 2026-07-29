import { Cause, Context, Duration, Effect, Layer, Schema } from "effect"
import { IcnClient, type IcnClientService } from "../client.js"
import {
  ModelDownloadsResponse as ModelDownloadsResponseSchema,
  type DownloadAttempt,
} from "@magnitudedev/icn-protocol/schemas"
import { makeIcnObservedState, type IcnObservedState } from "../observed-state.js"

type DownloadsReadError = Effect.Effect.Error<
  ReturnType<IcnClientService["models"]["listModelDownloads"]>
>

export interface IcnDownloadsService
  extends IcnObservedState<ModelDownloadsResponseSchema, DownloadsReadError> {
  readonly observeAttempt: (attempt: DownloadAttempt) => Effect.Effect<void>
}

export class IcnDownloads extends Context.Tag("@magnitudedev/icn/IcnDownloads")<
  IcnDownloads,
  IcnDownloadsService
>() {}

export interface IcnDownloadsOptions {
  readonly refreshInterval?: Duration.DurationInput
  readonly idleRefreshInterval?: Duration.DurationInput
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
      const hasActiveAttempt = observed.get.pipe(Effect.map(({ state }) =>
        state.attempts.some((attempt) =>
          attempt._tag === "Pending" || attempt._tag === "Downloading")))
      const poll = Effect.gen(function* () {
        const active = yield* hasActiveAttempt
        yield* Effect.sleep(active
          ? options.refreshInterval ?? "1 second"
          : options.idleRefreshInterval ?? "5 seconds")
        yield* observed.refresh.pipe(
          Effect.tapError((error) => Effect.logWarning("Unable to refresh model download attempts").pipe(
            Effect.annotateLogs({ cause: Cause.pretty(Cause.fail(error)) }),
          )),
          Effect.option,
        )
      })
      yield* poll.pipe(
        Effect.forever,
        Effect.forkScoped,
      )
      const observeAttempt = (attempt: DownloadAttempt) => observed.update((state) => {
        const existing = state.attempts.findIndex(({ id }) => id === attempt.id)
        if (existing === -1) {
          return { attempts: [...state.attempts, attempt] }
        }
        return {
          attempts: state.attempts.map((current, index) =>
            index === existing ? attempt : current),
        }
      })
      return IcnDownloads.of({
        get: observed.get,
        changes: observed.changes,
        initialized: observed.initialized,
        refresh: observed.refresh,
        observeAttempt,
      })
    }),
  )
