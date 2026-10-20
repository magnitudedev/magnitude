import { Context, Duration, Effect, Layer, Option } from "effect"
import { dataEquivalence } from "@magnitudedev/utils/data-equivalence"
import {
  ModelDownloadsResponse,
  type ModelDownload,
  type ModelId,
  type ModelPackage,
  type ServableModelBundle,
} from "@magnitudedev/icn-protocol/schemas"
import { IcnClient, type IcnClientService } from "../client.js"
import { makeIcnObservedState, type IcnObservedState } from "../observed-state.js"
import { IcnEvents, refreshOnIcnEvents } from "../events/index.js"

type DownloadsReadError = Effect.Effect.Error<ReturnType<IcnClientService["downloads"]["listModelDownloads"]>>
export type DownloadStartError = Effect.Effect.Error<ReturnType<IcnClientService["downloads"]["startModelDownload"]>>
export type DownloadCancelError = Effect.Effect.Error<ReturnType<IcnClientService["downloads"]["cancelModelDownload"]>>
export type DownloadAcknowledgeError = Effect.Effect.Error<
  ReturnType<IcnClientService["downloads"]["acknowledgeModelDownloadFailure"]>
>

/**
 * Non-catalog bundle downloads: added models and drafters fetched alone. `start` admits a download, or
 * none when every package of the bundle is already installed; ICN echoes `forModel` in the download so
 * its progress belongs to that model.
 */
export interface IcnDownloadsService extends IcnObservedState<typeof ModelDownloadsResponse.Type, DownloadsReadError> {
  readonly start: (
    bundle: ServableModelBundle,
    forModel: ModelId,
  ) => Effect.Effect<Option.Option<ModelDownload>, DownloadStartError>
  readonly cancel: (downloadId: string) => Effect.Effect<ModelDownload, DownloadCancelError>
  readonly acknowledgeFailure: (downloadId: string) => Effect.Effect<ModelDownload, DownloadAcknowledgeError>
}
export class IcnDownloads extends Context.Tag("@magnitudedev/icn/IcnDownloads")<IcnDownloads, IcnDownloadsService>() {}

export interface IcnDownloadsOptions { readonly retryInterval?: Duration.DurationInput }

export const makeIcnDownloads = (
  options: IcnDownloadsOptions = {},
): Layer.Layer<IcnDownloads, DownloadsReadError, IcnClient | IcnEvents> => Layer.scoped(IcnDownloads, Effect.gen(function* () {
  const client = yield* IcnClient
  const invalidations = yield* (yield* IcnEvents).subscribe
  const read = client.downloads.listModelDownloads({})
  const initial = yield* read
  const observed = yield* makeIcnObservedState(initial, read, dataEquivalence(), {
    initiallyInitialized: true,
  })
  yield* refreshOnIcnEvents(
    invalidations,
    new Set(["downloads"]),
    observed.refresh,
    "ICN downloads",
    options.retryInterval ?? "1 second",
  ).pipe(Effect.forkScoped)
  const settle = observed.refresh.pipe(Effect.ignore)
  return IcnDownloads.of({
    ...observed,
    start: (bundle, forModel) => client.downloads.startModelDownload({
      payload: { bundle, forModel: Option.some(forModel) },
    }).pipe(
      Effect.map(({ download }) => download),
      Effect.tap(() => settle),
    ),
    cancel: (downloadId) => client.downloads.cancelModelDownload({ path: { download_id: downloadId } }).pipe(
      Effect.tap(() => settle),
    ),
    acknowledgeFailure: (downloadId) =>
      client.downloads.acknowledgeModelDownloadFailure({ path: { download_id: downloadId } }).pipe(
        Effect.tap(() => settle),
      ),
  })
}))

/** Every package a bundle downloads, target first. */
export const bundlePackages = (bundle: ServableModelBundle): readonly ModelPackage[] => {
  if (bundle._tag === "Standalone") return [bundle.package]
  return bundle.draftSource._tag === "Separate" ? [bundle.target, bundle.draftSource.draft] : [bundle.target]
}

/** Whether a package holds this repository file at this commit. */
export const packageHoldsFile = (
  pkg: ModelPackage,
  repository: string,
  commit: string,
  path: string,
): boolean => pkg.source._tag === "HuggingFace"
  && pkg.source.repository === repository
  && pkg.source.revision === commit
  && pkg.files.some((file) => file.path === path)
