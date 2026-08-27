import { Context, Effect, Layer, Option, Stream, SubscriptionRef } from "effect"
import type {
  ModelDownloadFailure,
  ModelDownloadId,
  ModelFailure,
} from "@magnitudedev/acn-protocol"
import type { ProviderModelId } from "@magnitudedev/ai"

export type LocalModelSyncState =
  | { readonly _tag: "Admitting"; readonly generation: number; readonly cancelRequested: boolean }
  | { readonly _tag: "Correlated"; readonly generation: number; readonly downloadId: ModelDownloadId }
  | { readonly _tag: "AdmissionFailed"; readonly generation: number; readonly failure: ModelDownloadFailure }

export type LocalModelRemovalState =
  | { readonly _tag: "Removing"; readonly generation: number }
  | { readonly _tag: "RemoveFailed"; readonly generation: number; readonly failure: ModelFailure }

export interface LocalModelAcquisitionCoordination {
  readonly nextGeneration: number
  readonly syncs: ReadonlyMap<ProviderModelId, LocalModelSyncState>
  readonly removals: ReadonlyMap<ProviderModelId, LocalModelRemovalState>
}

export interface LocalModelAcquisitionCoordinatorApi {
  readonly state: Effect.Effect<LocalModelAcquisitionCoordination>
  readonly changes: Stream.Stream<void>
  readonly admitSync: (
    modelId: ProviderModelId,
    replaceGeneration: Option.Option<number>,
  ) => Effect.Effect<Option.Option<number>>
  readonly correlateSync: (
    modelId: ProviderModelId,
    generation: number,
    downloadId: ModelDownloadId,
  ) => Effect.Effect<Option.Option<boolean>>
  readonly failSyncAdmission: (
    modelId: ProviderModelId,
    generation: number,
    failure: ModelDownloadFailure,
  ) => Effect.Effect<void>
  readonly finishSync: (modelId: ProviderModelId, generation: number) => Effect.Effect<void>
  readonly requestSyncCancellation: (
    modelId: ProviderModelId,
  ) => Effect.Effect<Option.Option<{
    readonly generation: number
    readonly downloadId: ModelDownloadId
  }>>
  readonly admitRemoval: (modelId: ProviderModelId) => Effect.Effect<Option.Option<number>>
  readonly failRemoval: (
    modelId: ProviderModelId,
    generation: number,
    failure: ModelFailure,
  ) => Effect.Effect<void>
  readonly finishRemoval: (modelId: ProviderModelId, generation: number) => Effect.Effect<void>
}

export class LocalModelAcquisitionCoordinator extends Context.Tag(
  "LocalModelAcquisitionCoordinator",
)<LocalModelAcquisitionCoordinator, LocalModelAcquisitionCoordinatorApi>() {}

const updateMap = <K, V>(
  source: ReadonlyMap<K, V>,
  update: (next: Map<K, V>) => void,
): ReadonlyMap<K, V> => {
  const next = new Map(source)
  update(next)
  return next
}

export const LocalModelAcquisitionCoordinatorLive: Layer.Layer<LocalModelAcquisitionCoordinator> =
  Layer.effect(LocalModelAcquisitionCoordinator, Effect.gen(function* () {
    const state = yield* SubscriptionRef.make<LocalModelAcquisitionCoordination>({
      nextGeneration: 1,
      syncs: new Map(),
      removals: new Map(),
    })

    const currentSyncGeneration = (
      current: LocalModelAcquisitionCoordination,
      modelId: ProviderModelId,
      generation: number,
    ) => current.syncs.get(modelId)?.generation === generation

    return LocalModelAcquisitionCoordinator.of({
      state: SubscriptionRef.get(state),
      changes: state.changes.pipe(Stream.drop(1), Stream.map(() => undefined)),
      admitSync: (modelId, replaceGeneration) => SubscriptionRef.modify(state, (current) => {
        if (current.removals.get(modelId)?._tag === "Removing") {
          return [Option.none(), current]
        }
        const existing = current.syncs.get(modelId)
        if (existing !== undefined
          && !Option.contains(replaceGeneration, existing.generation)) {
          return [Option.none(), current]
        }
        const generation = current.nextGeneration
        return [Option.some(generation), {
          ...current,
          nextGeneration: generation + 1,
          syncs: updateMap(current.syncs, (next) => next.set(modelId, {
            _tag: "Admitting",
            generation,
            cancelRequested: false,
          })),
          removals: updateMap(current.removals, (next) => next.delete(modelId)),
        }]
      }),
      correlateSync: (modelId, generation, downloadId) => SubscriptionRef.modify(
        state,
        (current) => {
          const existing = current.syncs.get(modelId)
          if (existing?._tag !== "Admitting" || existing.generation !== generation) {
            return [Option.none(), current]
          }
          return [Option.some(existing.cancelRequested), {
            ...current,
            syncs: updateMap(current.syncs, (next) => next.set(modelId, {
              _tag: "Correlated",
              generation,
              downloadId,
            })),
          }]
        },
      ),
      failSyncAdmission: (modelId, generation, failure) => SubscriptionRef.update(
        state,
        (current) => current.syncs.get(modelId)?._tag === "Admitting"
            && currentSyncGeneration(current, modelId, generation)
          ? {
              ...current,
              syncs: updateMap(current.syncs, (next) => next.set(modelId, {
                _tag: "AdmissionFailed",
                generation,
                failure,
              })),
            }
          : current,
      ),
      finishSync: (modelId, generation) => SubscriptionRef.update(state, (current) => {
        const existing = current.syncs.get(modelId)
        if (existing?.generation !== generation) {
          return current
        }
        return {
          ...current,
          syncs: updateMap(current.syncs, (next) => next.delete(modelId)),
        }
      }),
      requestSyncCancellation: (modelId) => SubscriptionRef.modify(state, (current) => {
        const existing = current.syncs.get(modelId)
        if (existing?._tag === "Correlated") {
          return [Option.some({
            generation: existing.generation,
            downloadId: existing.downloadId,
          }), current]
        }
        if (existing?._tag !== "Admitting" || existing.cancelRequested) {
          return [Option.none(), current]
        }
        return [Option.none(), {
          ...current,
          syncs: updateMap(current.syncs, (next) => next.set(modelId, {
            ...existing,
            cancelRequested: true,
          })),
        }]
      }),
      admitRemoval: (modelId) => SubscriptionRef.modify(state, (current) => {
        const synchronization = current.syncs.get(modelId)
        if ((synchronization !== undefined && synchronization._tag !== "AdmissionFailed")
          || current.removals.get(modelId)?._tag === "Removing") {
          return [Option.none(), current]
        }
        const generation = current.nextGeneration
        return [Option.some(generation), {
          ...current,
          nextGeneration: generation + 1,
          removals: updateMap(current.removals, (next) => next.set(modelId, {
            _tag: "Removing",
            generation,
          })),
          syncs: updateMap(current.syncs, (next) => next.delete(modelId)),
        }]
      }),
      failRemoval: (modelId, generation, failure) => SubscriptionRef.update(state, (current) => {
        const existing = current.removals.get(modelId)
        if (existing?.generation !== generation) return current
        return {
          ...current,
          removals: updateMap(current.removals, (next) => next.set(modelId, {
            _tag: "RemoveFailed",
            generation,
            failure,
          })),
        }
      }),
      finishRemoval: (modelId, generation) => SubscriptionRef.update(state, (current) => {
        if (current.removals.get(modelId)?.generation !== generation) return current
        return {
          ...current,
          removals: updateMap(current.removals, (next) => next.delete(modelId)),
        }
      }),
    })
  }))
