import { Cause, Context, Duration, Effect, Layer, Option, Schema } from "effect"
import { IcnClient, type IcnClientService } from "../client.js"
import {
  ModelDownloadsResponse as ModelDownloadsResponseSchema,
  type ModelDownload,
  type StartModelDownloadResponse,
} from "@magnitudedev/icn-protocol/schemas"
import { makeIcnObservedState, type IcnObservedState } from "../observed-state.js"

type DownloadsReadError = Effect.Effect.Error<
  ReturnType<IcnClientService["models"]["listModelDownloads"]>
>

export interface IcnDownloadsService
  extends IcnObservedState<ModelDownloadsResponseSchema, DownloadsReadError> {
  readonly observeDownload: (download: ModelDownload) => Effect.Effect<void>
  readonly observeAdmission: (admission: StartModelDownloadResponse) => Effect.Effect<void>
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
      const hasActiveDownload = observed.get.pipe(Effect.map(({ state }) =>
        state.downloads.some(({ state }) =>
          state._tag === "Pending" || state._tag === "Downloading")))
      const poll = Effect.gen(function* () {
        const active = yield* hasActiveDownload
        yield* Effect.sleep(active
          ? options.refreshInterval ?? "1 second"
          : options.idleRefreshInterval ?? "5 seconds")
        yield* observed.refresh.pipe(
          Effect.tapError((error) => Effect.logWarning("Unable to refresh model downloads").pipe(
            Effect.annotateLogs({ cause: Cause.pretty(Cause.fail(error)) }),
          )),
          Effect.option,
        )
      })
      yield* poll.pipe(
        Effect.forever,
        Effect.forkScoped,
      )
      const observeDownload = (download: ModelDownload) => observed.update((state) => {
        const existing = state.downloads.findIndex(({ id }) => id === download.id)
        if (existing === -1) {
          return { ...state, downloads: [...state.downloads, download] }
        }
        return {
          ...state,
          downloads: state.downloads.map((current, index) =>
            index === existing ? download : current),
        }
      })
      const observeAdmission = (admission: StartModelDownloadResponse) => Option.match(
        admission.download,
        {
          onNone: () => Effect.void,
          onSome: observeDownload,
        },
      )
      return IcnDownloads.of({
        get: observed.get,
        changes: observed.changes,
        initialized: observed.initialized,
        refresh: observed.refresh,
        observeDownload,
        observeAdmission,
      })
    }),
  )
