import { Context, Effect } from 'effect'
import type { ModelDefinition } from '../types'

export class ModelCatalog extends Context.Tag('ModelCatalog')<
  ModelCatalog,
  {
    refresh: () => Effect.Effect<void>
    getModels: (providerId: string) => Effect.Effect<readonly ModelDefinition[]>
  }
>() {}