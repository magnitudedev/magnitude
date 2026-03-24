import { Context, type Effect } from 'effect'
import type { BoundModel } from '../model/bound-model'
import type { ModelError } from '../errors/model-error'

export interface ModelResolverShape<TSlot extends string> {
  readonly resolve: (slot: TSlot) => Effect.Effect<BoundModel, ModelError>
}

export const ModelResolver = Context.GenericTag<ModelResolverShape<string>>('ModelResolver')
export type ModelResolver = Context.Tag.Identifier<typeof ModelResolver>