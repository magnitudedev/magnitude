import { Context, Effect, Layer, Stream, SubscriptionRef } from "effect"
import {
  LocalModelMutationFailed,
  type LocalInferenceError,
  type ModelFailure,
  type CatalogFormModelId,
} from "@magnitudedev/acn-protocol"
import { IcnCatalog, IcnClient } from "@magnitudedev/icn"
import type { CatalogInstallationRemoval } from "@magnitudedev/icn-protocol/schemas"
import { icnCommandFailure } from "./icn-command-failure"

export type LocalModelRemovalState =
  | { readonly _tag: "Removing" }
  | { readonly _tag: "RemoveFailed"; readonly failure: ModelFailure }

export interface LocalModelRemovalsApi {
  readonly state: Effect.Effect<ReadonlyMap<CatalogFormModelId, LocalModelRemovalState>>
  readonly changes: Stream.Stream<void>
  readonly remove: (modelId: CatalogFormModelId) => Effect.Effect<{}, LocalInferenceError>
}

export class LocalModelRemovals extends Context.Tag("LocalModelRemovals")<
  LocalModelRemovals,
  LocalModelRemovalsApi
>() {}

export const catalogRemovalOutcome = (
  result: CatalogInstallationRemoval,
): Effect.Effect<{}, LocalModelMutationFailed> => result._tag === "Removed"
  ? Effect.succeed({})
  : Effect.fail(new LocalModelMutationFailed({
      code: result.reason === "ExternalOwnership"
        ? "model_removal_retained_external"
        : "model_removal_retained_shared",
      message: result.reason === "ExternalOwnership"
        ? "This model is provided by an external Hugging Face cache and was not removed."
        : "This model uses files shared by another catalog model and was not removed.",
      retryable: false,
    }))

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
  IcnClient | IcnCatalog
> =
  Layer.effect(LocalModelRemovals, Effect.gen(function* () {
    const client = yield* IcnClient
    const catalog = yield* IcnCatalog
    const removals = yield* SubscriptionRef.make<ReadonlyMap<CatalogFormModelId, LocalModelRemovalState>>(new Map())
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
      const clear = SubscriptionRef.update(removals, (current) => updated(current, (next) => next.delete(modelId)))
      yield* restore(client.catalog.removeCatalogModelInstallation({ path: { model_id: modelId } }).pipe(
        Effect.mapError((cause) => icnCommandFailure("remove", cause)),
        Effect.flatMap(catalogRemovalOutcome),
        Effect.tap(() => catalog.refresh.pipe(Effect.ignore)),
        Effect.tapError((failure) => SubscriptionRef.update(removals, (current) => updated(current, (next) => next.set(modelId, {
          _tag: "RemoveFailed",
          failure,
        })))),
        Effect.onInterrupt(() => clear),
      ))
      yield* clear
      return {}
    }))
    return LocalModelRemovals.of({
      state: SubscriptionRef.get(removals),
      changes: removals.changes.pipe(Stream.drop(1), Stream.map(() => undefined)),
      remove,
    })
  }))
