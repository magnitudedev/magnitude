import { Context, type Effect } from 'effect'
import type { Model } from '../model/model'
import type { ModelSlot } from '../state/provider-state'
import type { BoundModel } from '../model/bound-model'
import type { ModelError } from '../errors/model-error'

export interface ContextLimits {
  readonly hardCap: number
  readonly softCap: number
}

export class ModelResolver extends Context.Tag('ModelResolver')<
  ModelResolver,
  {
    readonly resolve: (slot: ModelSlot) => Effect.Effect<BoundModel, ModelError>
    readonly peek: (slot?: ModelSlot) => Model | null
    readonly contextLimits: (slot?: ModelSlot) => ContextLimits
    readonly contextWindow: (slot?: ModelSlot) => number
  }
>() {}