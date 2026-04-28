import { Data, Effect } from "effect"
import type { ProviderModel } from "../lib/model/provider-model"

export interface CatalogueSource {
  readonly id: string
  readonly fetch: () => Effect.Effect<
    ReadonlyMap<string, readonly ProviderModel[]>,
    CatalogueError
  >
}

export class CatalogueTransportError extends Data.TaggedError("CatalogueTransportError")<{
  readonly sourceId: string
  readonly providerId: string
  readonly message: string
  readonly cause?: unknown
}> {}

export class CatalogueAuthError extends Data.TaggedError("CatalogueAuthError")<{
  readonly sourceId: string
  readonly providerId: string
  readonly message: string
}> {}

export class CatalogueSchemaError extends Data.TaggedError("CatalogueSchemaError")<{
  readonly sourceId: string
  readonly providerId: string
  readonly message: string
  readonly cause?: unknown
}> {}

export type CatalogueError =
  | CatalogueTransportError
  | CatalogueAuthError
  | CatalogueSchemaError
