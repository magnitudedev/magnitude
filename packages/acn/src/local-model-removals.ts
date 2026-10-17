import { Context, Effect, Layer, Stream, SubscriptionRef } from "effect"
import {
  LocalModelMutationFailed,
  type LocalInferenceError,
  type ModelFailure,
  type CatalogFormModelId,
} from "@magnitudedev/acn-protocol"
import { IcnCatalog, IcnClient, IcnLocalModels } from "@magnitudedev/icn"
import type { CatalogInstallationRemoval } from "@magnitudedev/icn-protocol/schemas"
import { icnCommandFailure } from "./icn-command-failure"

export type LocalModelRemovalState =
  | { readonly _tag: "Removing" }
  | { readonly _tag: "RemoveFailed"; readonly failure: ModelFailure }

export interface LocalModelRemovalsApi {
  readonly state: Effect.Effect<ReadonlyMap<CatalogFormModelId, LocalModelRemovalState>>
  readonly changes: Stream.Stream<void>
  readonly remove: (modelId: CatalogFormModelId) => Effect.Effect<{}, LocalInferenceError>
  /** A new installation supersedes an earlier removal's failure. */
  readonly forget: (modelId: CatalogFormModelId) => Effect.Effect<void>
}

export class LocalModelRemovals extends Context.Tag("LocalModelRemovals")<
  LocalModelRemovals,
  LocalModelRemovalsApi
>() {}

/** ICN retains an installation it doesn't own outright: only another app's copy (hide it instead), or shared files. */
export const catalogRemovalOutcome = (modelId: CatalogFormModelId) => (
  result: CatalogInstallationRemoval,
): Effect.Effect<{}, LocalModelMutationFailed> => result._tag === "Removed"
  ? Effect.succeed({})
  : Effect.fail(new LocalModelMutationFailed({
      code: result.reason === "ExternalOwnership"
        ? "model_removal_retained_external"
        : "model_removal_retained_shared",
      message: result.reason === "ExternalOwnership"
        ? `${modelId} uses files another app manages, which Magnitude never deletes. Hide it with \`magnitude models hide ${modelId}\` instead.`
        : "This model uses files shared by another catalog model and was not removed.",
      retryable: false,
    }))

const CATALOG_MODEL_NOT_INSTALLED = "catalog_model_not_installed"

/** Removing a model that isn't installed changes nothing, so it is never a failed removal. */
export const notInstalledRemoval = (modelId: CatalogFormModelId) => new LocalModelMutationFailed({
  code: "model_not_installed",
  message: `${modelId} is not installed on this computer, so there is nothing to remove.`,
  retryable: false,
})

const updated = (
  current: ReadonlyMap<CatalogFormModelId, LocalModelRemovalState>,
  update: (next: Map<CatalogFormModelId, LocalModelRemovalState>) => void,
) => {
  const next = new Map(current)
  update(next)
  return next
}

export const LocalModelRemovalsLive: Layer.Layer<
  LocalModelRemovals,
  never,
  IcnClient | IcnCatalog | IcnLocalModels
> =
  Layer.effect(LocalModelRemovals, Effect.gen(function* () {
    const client = yield* IcnClient
    const catalog = yield* IcnCatalog
    const localModels = yield* IcnLocalModels
    const removals = yield* SubscriptionRef.make<ReadonlyMap<CatalogFormModelId, LocalModelRemovalState>>(new Map())
    const clear = (modelId: CatalogFormModelId) =>
      SubscriptionRef.update(removals, (current) => current.has(modelId)
        ? updated(current, (next) => next.delete(modelId))
        : current)
    const remove = (modelId: CatalogFormModelId) => Effect.uninterruptibleMask((restore) => Effect.gen(function* () {
      const admitted = yield* SubscriptionRef.modify(removals, (current) => current.get(modelId)?._tag === "Removing"
        ? [false, current]
        : [true, updated(current, (next) => next.set(modelId, { _tag: "Removing" }))])
      if (!admitted) {
        return yield* new LocalModelMutationFailed({
          code: "model_remove_in_progress",
          message: "This model is already being removed",
          retryable: true,
        })
      }
      yield* restore(client.catalog.removeCatalogModelInstallation({ path: { model_id: modelId } }).pipe(
        Effect.mapError((cause) => icnCommandFailure("remove", cause)),
        Effect.mapError((failure) => failure.code === CATALOG_MODEL_NOT_INSTALLED ? notInstalledRemoval(modelId) : failure),
        Effect.flatMap(catalogRemovalOutcome(modelId)),
        Effect.tap(() => catalog.refresh.pipe(Effect.ignore)),
        // The catalog model's drafter stays while another model uses it; otherwise it goes too.
        Effect.tap(() => localModels.pruneDrafters.pipe(Effect.ignore)),
        Effect.tapError((failure) => failure.code === "model_not_installed"
          ? clear(modelId)
          : SubscriptionRef.update(removals, (current) => updated(current, (next) => next.set(modelId, {
              _tag: "RemoveFailed",
              failure,
            })))),
        Effect.onInterrupt(() => clear(modelId)),
      ))
      yield* clear(modelId)
      return {}
    }))
    return LocalModelRemovals.of({
      state: SubscriptionRef.get(removals),
      changes: removals.changes.pipe(Stream.drop(1), Stream.map(() => undefined)),
      remove,
      forget: clear,
    })
  }))
