import { Context, type Effect } from 'effect'
import type { ModelSlot } from '../state/provider-state'
import type { BoundModel } from '../model/bound-model'
import type { ModelError } from '../errors/model-error'

export class ModelResolver extends Context.Tag('ModelResolver')<
  ModelResolver,
  {
    readonly resolve: (slot: ModelSlot) => Effect.Effect<BoundModel, ModelError>
  }
>() {}