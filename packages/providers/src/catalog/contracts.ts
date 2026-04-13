import { AppConfig, CatalogCache } from '@magnitudedev/storage'
import { Context, Data, Effect } from 'effect'
import type { ProviderAuth } from '../runtime/contracts'
import type { ModelDefinition, ProviderDefinition } from '../types'

export type CatalogSourceEnv = CatalogCache | AppConfig | ProviderAuth

export interface CatalogSource {
  readonly id: string
  readonly priority: number
  readonly supports: (provider: ProviderDefinition) => boolean
  readonly refresh: (provider: ProviderDefinition) => Effect.Effect<
    readonly ModelDefinition[],
    CatalogError,
    CatalogSourceEnv
  >
}

export class CatalogTransportError extends Data.TaggedError('CatalogTransportError')<{
  sourceId: string
  providerId: string
  message: string
  cause?: unknown
}> {}

export class CatalogAuthError extends Data.TaggedError('CatalogAuthError')<{
  sourceId: string
  providerId: string
  message: string
}> {}

export class CatalogSchemaError extends Data.TaggedError('CatalogSchemaError')<{
  sourceId: string
  providerId: string
  message: string
  cause?: unknown
}> {}

export type CatalogError =
  | CatalogTransportError
  | CatalogAuthError
  | CatalogSchemaError

export class CatalogSourceRegistry extends Context.Tag('CatalogSourceRegistry')<
  CatalogSourceRegistry,
  { readonly list: () => readonly CatalogSource[] }
>() {}

export class ModelCatalog extends Context.Tag('ModelCatalog')<
  ModelCatalog,
  {
    refresh: () => Effect.Effect<void>
    getModels: (providerId: string) => Effect.Effect<readonly ModelDefinition[]>
  }
>() {}