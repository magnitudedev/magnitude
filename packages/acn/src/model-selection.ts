import { Context, Effect, Layer, Option, Stream } from 'effect'
import type {
  ProviderModelIdentity,
  SlotId,
  SlotSelection,
} from '@magnitudedev/acn-protocol'
import {
  MagnitudeStorage,
  type ModelState,
  type StateDocumentError,
  type StateHandle,
} from '@magnitudedev/storage'

export interface ModelSelectionState {
  readonly slots: ModelState['slots']
  readonly recentModels: ModelState['recentModels']
  readonly favorites: ModelState['favorites']
}

export interface ModelSelectionApi {
  readonly get: Effect.Effect<ModelSelectionState>
  readonly changes: Stream.Stream<ModelSelectionState>
  readonly updateSlot: (
    slotId: SlotId,
    selection: Option.Option<SlotSelection>,
  ) => Effect.Effect<void, StateDocumentError>
  readonly recordUse: (
    slotId: SlotId,
    model: ProviderModelIdentity,
  ) => Effect.Effect<void, StateDocumentError>
  readonly setFavorite: (
    model: ProviderModelIdentity,
    favorite: boolean,
  ) => Effect.Effect<void, StateDocumentError>
}

export class ModelSelection extends Context.Tag('ModelSelection')<
  ModelSelection,
  ModelSelectionApi
>() {}

const RECENCY_LIMIT = 32
const slotKey = (slotId: SlotId): 'primary' | 'secondary' =>
  slotId === 'primary' ? 'primary' : 'secondary'
const matches = (left: ProviderModelIdentity, right: ProviderModelIdentity): boolean =>
  left.providerId === right.providerId && left.providerModelId === right.providerModelId

const moveToFront = (
  values: readonly ProviderModelIdentity[],
  model: ProviderModelIdentity,
): readonly ProviderModelIdentity[] => [
  model,
  ...values.filter((candidate) => !matches(candidate, model)),
].slice(0, RECENCY_LIMIT)

const select = (state: ModelState): ModelSelectionState => ({
  slots: state.slots,
  recentModels: state.recentModels,
  favorites: state.favorites,
})

export const makeModelSelection = (
  state: StateHandle<ModelState, StateDocumentError>,
): ModelSelectionApi => ({
  get: state.get.pipe(Effect.map(select)),
  changes: state.changes.pipe(Stream.map(select)),
  updateSlot: (slotId, selection) => state.update((current) => ({
    ...current,
    slots: { ...current.slots, [slotKey(slotId)]: selection },
    recentModels: Option.match(selection, {
      onNone: () => current.recentModels,
      onSome: (selected) => ({
        ...current.recentModels,
        [slotKey(slotId)]: moveToFront(current.recentModels[slotKey(slotId)], {
          providerId: selected.providerId,
          providerModelId: selected.providerModelId,
        }),
      }),
    }),
  })).pipe(Effect.asVoid),
  recordUse: (slotId, model) => state.update((current) => ({
    ...current,
    recentModels: {
      ...current.recentModels,
      [slotKey(slotId)]: moveToFront(current.recentModels[slotKey(slotId)], model),
    },
  })).pipe(Effect.asVoid),
  setFavorite: (model, favorite) => state.update((current) => ({
    ...current,
    favorites: favorite
      ? moveToFront(current.favorites, model)
      : current.favorites.filter((candidate) => !matches(candidate, model)),
  })).pipe(Effect.asVoid),
})

export const ModelSelectionLive: Layer.Layer<ModelSelection, never, MagnitudeStorage> =
  Layer.effect(ModelSelection, Effect.gen(function* () {
    const storage = yield* MagnitudeStorage
    return ModelSelection.of(makeModelSelection(storage.models))
  }))
