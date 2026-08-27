import { Context, Effect, Layer, Option, SubscriptionRef } from "effect"
import type {
  ModelBundleDownload,
  ModelDownloadId,
} from "@magnitudedev/acn-protocol"
import type { ProviderModelId } from "@magnitudedev/ai"
import { LocalModelPackages } from "./local-model-packages"

export interface LocalModelSyncsApi {
  /** Associates one client-visible model sync with the exact ICN occurrence it admitted. */
  readonly admitted: (
    modelId: ProviderModelId,
    downloadId: ModelDownloadId,
  ) => Effect.Effect<void>
  /** Removes obsolete occurrence correlation when ICN reports that the model is already current. */
  readonly current: (modelId: ProviderModelId) => Effect.Effect<void>
  /** Resolves the model's correlated occurrence from current ICN download facts. */
  readonly download: (modelId: ProviderModelId) => Effect.Effect<Option.Option<ModelBundleDownload>>
  /** Resolves every retained correlation against one current ICN download snapshot. */
  readonly downloads: Effect.Effect<ReadonlyMap<ProviderModelId, ModelBundleDownload>>
}

export class LocalModelSyncs extends Context.Tag("LocalModelSyncs")<
  LocalModelSyncs,
  LocalModelSyncsApi
>() {}

export const LocalModelSyncsLive: Layer.Layer<
  LocalModelSyncs,
  never,
  LocalModelPackages
> = Layer.effect(LocalModelSyncs, Effect.gen(function* () {
  const packages = yield* LocalModelPackages
  const correlations = yield* SubscriptionRef.make<ReadonlyMap<ProviderModelId, ModelDownloadId>>(
    new Map(),
  )

  return LocalModelSyncs.of({
    admitted: (modelId, downloadId) => SubscriptionRef.update(correlations, (current) => {
      const next = new Map(current)
      next.set(modelId, downloadId)
      return next
    }),
    current: (modelId) => SubscriptionRef.update(correlations, (current) => {
      if (!current.has(modelId)) return current
      const next = new Map(current)
      next.delete(modelId)
      return next
    }),
    download: (modelId) => Effect.all([
      SubscriptionRef.get(correlations),
      packages.state,
    ]).pipe(
      Effect.map(([current, packageState]) => Option.fromNullable(
        packageState.downloads.find(({ id }) => id === current.get(modelId)),
      )),
    ),
    downloads: Effect.all([
      SubscriptionRef.get(correlations),
      packages.state,
    ]).pipe(
      Effect.map(([current, packageState]) => {
        const downloadsById = new Map(packageState.downloads.map((download) => [download.id, download]))
        return new Map([...current].flatMap(([modelId, downloadId]) => {
          const correlated = downloadsById.get(downloadId)
          return correlated === undefined ? [] : [[modelId, correlated] as const]
        }))
      }),
    ),
  })
}))
