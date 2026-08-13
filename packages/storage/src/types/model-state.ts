import {
  ProviderModelIdentitySchema,
  SlotSelectionSchema,
} from '@magnitudedev/acn-protocol'
import { Option, Schema } from 'effect'

const SerializableOptional = <A, I>(schema: Schema.Schema<A, I>) =>
  Schema.optionalWith(schema, { as: 'Option', exact: true } as const)

const PersistedSlotsSchema = Schema.Struct({
  primary: SerializableOptional(SlotSelectionSchema),
  secondary: SerializableOptional(SlotSelectionSchema),
})

const RuntimeSlotsSchema = Schema.Struct({
  primary: Schema.OptionFromSelf(SlotSelectionSchema),
  secondary: Schema.OptionFromSelf(SlotSelectionSchema),
})

const RecentModelsSchema = Schema.Struct({
  primary: Schema.Array(ProviderModelIdentitySchema),
  secondary: Schema.Array(ProviderModelIdentitySchema),
})

const PersistedModelStateSchema = Schema.Struct({
  slots: SerializableOptional(PersistedSlotsSchema),
  recentModels: SerializableOptional(RecentModelsSchema),
  favorites: SerializableOptional(Schema.Array(ProviderModelIdentitySchema)),
})

const ModelStateRuntimeSchema = Schema.Struct({
  slots: RuntimeSlotsSchema,
  recentModels: RecentModelsSchema,
  favorites: Schema.Array(ProviderModelIdentitySchema),
})

export type ModelState = typeof ModelStateRuntimeSchema.Type

const identityKey = (identity: {
  readonly providerId: string
  readonly providerModelId: string
}): string => `${identity.providerId}\0${identity.providerModelId}`

const uniqueIdentities = <A extends {
  readonly providerId: string
  readonly providerModelId: string
}>(values: readonly A[], limit?: number): readonly A[] => {
  const seen = new Set<string>()
  const result: A[] = []
  for (const value of values) {
    const key = identityKey(value)
    if (seen.has(key)) continue
    seen.add(key)
    result.push(value)
    if (limit !== undefined && result.length === limit) break
  }
  return result
}

export const EMPTY_MODEL_STATE: ModelState = {
  slots: { primary: Option.none(), secondary: Option.none() },
  recentModels: { primary: [], secondary: [] },
  favorites: [],
}

export const ModelStateSchema = Schema.transform(
  PersistedModelStateSchema,
  Schema.typeSchema(ModelStateRuntimeSchema),
  {
    strict: true,
    decode: (persisted): ModelState => {
      const recentModels = Option.getOrElse(persisted.recentModels, () => ({
        primary: [],
        secondary: [],
      }))
      return {
        slots: Option.getOrElse(persisted.slots, () => ({
          primary: Option.none(),
          secondary: Option.none(),
        })),
        recentModels: {
          primary: uniqueIdentities(recentModels.primary, 32),
          secondary: uniqueIdentities(recentModels.secondary, 32),
        },
        favorites: uniqueIdentities(Option.getOrElse(persisted.favorites, () => [])),
      }
    },
    encode: (state) => ({
      slots: Option.some(state.slots),
      recentModels: Option.some(state.recentModels),
      favorites: Option.some(state.favorites),
    }),
  },
)
