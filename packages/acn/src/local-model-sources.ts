import { Context, Effect, Layer, ParseResult, Schema, Stream } from "effect"
import { IcnCatalog, IcnDiscovery, type IcnDiscoveryService } from "@magnitudedev/icn"
import {
  CatalogFormModelIdSchema,
  HuggingFaceFormModelIdSchema,
} from "@magnitudedev/acn-protocol"
import type { CatalogModel, DiscoveredModel } from "@magnitudedev/icn-protocol/schemas"
import { materializeProjection } from "./materialized-projection"

export interface CatalogModelSource {
  readonly id: typeof CatalogFormModelIdSchema.Type
  readonly source: Omit<CatalogModel, "id">
}

export interface DiscoveredModelSource {
  readonly id: typeof HuggingFaceFormModelIdSchema.Type
  readonly source: Omit<DiscoveredModel, "id">
}

export interface LocalModelSourcesState {
  readonly catalogRevision: number
  readonly discoveryRevision: number
  readonly reconciliationComplete: boolean
  readonly catalogModels: readonly CatalogModelSource[]
  readonly discoveredModels: readonly DiscoveredModelSource[]
}

export interface LocalModelSourcesApi {
  readonly state: Effect.Effect<LocalModelSourcesState>
  readonly changes: Stream.Stream<LocalModelSourcesState>
  readonly refreshDiscovery: IcnDiscoveryService["reconcile"]
}

export class LocalModelSources extends Context.Tag("LocalModelSources")<LocalModelSources, LocalModelSourcesApi>() {}

const catalogModelId = (value: string): Effect.Effect<typeof CatalogFormModelIdSchema.Type, ParseResult.ParseError> =>
  Schema.decodeUnknown(CatalogFormModelIdSchema)(value)

const discoveredModelId = (value: string): Effect.Effect<typeof HuggingFaceFormModelIdSchema.Type, ParseResult.ParseError> =>
  Schema.decodeUnknown(HuggingFaceFormModelIdSchema)(value)

export const LocalModelSourcesLive: Layer.Layer<LocalModelSources, never, IcnCatalog | IcnDiscovery> =
  Layer.scoped(LocalModelSources, Effect.gen(function* () {
    const catalog = yield* IcnCatalog
    const discovery = yield* IcnDiscovery
    const project = Effect.all({ catalog: catalog.get, discovery: discovery.get }).pipe(
      Effect.flatMap(({ catalog, discovery }) => Effect.all({
        catalogModels: Effect.forEach(catalog.state.models, ({ id, ...source }) => catalogModelId(id).pipe(
          Effect.map((id): CatalogModelSource => ({ id, source })),
        )),
        discoveredModels: Effect.forEach(discovery.state.models, ({ id, ...source }) => discoveredModelId(id).pipe(
          Effect.map((id): DiscoveredModelSource => ({ id, source })),
        )),
      }).pipe(Effect.map(({ catalogModels, discoveredModels }): LocalModelSourcesState => ({
        catalogRevision: catalog.state.revision,
        discoveryRevision: discovery.state.revision,
        reconciliationComplete: catalog.state.reconciliationComplete && discovery.state.reconciliationComplete,
        catalogModels,
        discoveredModels,
      })))),
      Effect.orDie,
    )
    const projection = yield* materializeProjection({
      project,
      invalidations: Stream.merge(catalog.changes, discovery.changes),
      equivalent: (left, right) => left.catalogRevision === right.catalogRevision
        && left.discoveryRevision === right.discoveryRevision,
    })
    return LocalModelSources.of({
      state: projection.get,
      changes: projection.changes,
      refreshDiscovery: discovery.reconcile,
    })
  }))
