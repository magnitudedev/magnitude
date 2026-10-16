import { Context, Effect, Layer, ParseResult, Schema, Stream } from "effect"
import { IcnCatalog, IcnLocalModels, type IcnLocalModelsService } from "@magnitudedev/icn"
import {
  CatalogFormModelIdSchema,
  CustomFormModelIdSchema,
} from "@magnitudedev/acn-protocol"
import type { CatalogModel, CustomModel, UnsupportedLocalModel } from "@magnitudedev/icn-protocol/schemas"
import { materializeProjection } from "./materialized-projection"

export interface CatalogModelSource {
  readonly id: typeof CatalogFormModelIdSchema.Type
  readonly source: Omit<CatalogModel, "id">
}

export interface CustomModelSource {
  readonly id: typeof CustomFormModelIdSchema.Type
  readonly source: Omit<CustomModel, "modelId">
}

export interface LocalModelSourcesState {
  readonly catalogRevision: number
  readonly localModelsRevision: number
  readonly reconciliationComplete: boolean
  readonly catalogModels: readonly CatalogModelSource[]
  readonly customModels: readonly CustomModelSource[]
  readonly hiddenModels: readonly CustomModelSource[]
  readonly unsupported: readonly UnsupportedLocalModel[]
}

export interface LocalModelSourcesApi {
  readonly state: Effect.Effect<LocalModelSourcesState>
  readonly changes: Stream.Stream<LocalModelSourcesState>
  readonly refreshLocalModels: IcnLocalModelsService["reconcile"]
}

export class LocalModelSources extends Context.Tag("LocalModelSources")<LocalModelSources, LocalModelSourcesApi>() {}

const catalogModelId = (value: string): Effect.Effect<typeof CatalogFormModelIdSchema.Type, ParseResult.ParseError> =>
  Schema.decodeUnknown(CatalogFormModelIdSchema)(value)

/** A row whose identity the protocol cannot carry is a native contract defect; it is logged, not fatal. */
const customModelSources = (models: readonly CustomModel[]): Effect.Effect<readonly CustomModelSource[]> =>
  Effect.forEach(models, ({ modelId, ...source }) =>
    Schema.decodeUnknown(CustomFormModelIdSchema)(modelId).pipe(
      Effect.map((id): readonly CustomModelSource[] => [{ id, source }]),
      Effect.catchAll(() => Effect.logWarning("ICN published a custom model with a non-canonical ID").pipe(
        Effect.annotateLogs({ modelId }),
        Effect.as([]),
      )),
    )).pipe(Effect.map((rows) => rows.flat()))

export const LocalModelSourcesLive: Layer.Layer<LocalModelSources, never, IcnCatalog | IcnLocalModels> =
  Layer.scoped(LocalModelSources, Effect.gen(function* () {
    const catalog = yield* IcnCatalog
    const localModels = yield* IcnLocalModels
    const project = Effect.all({ catalog: catalog.get, local: localModels.get }).pipe(
      Effect.flatMap(({ catalog, local }) => Effect.all({
        catalogModels: Effect.forEach(catalog.state.models, ({ id, ...source }) => catalogModelId(id).pipe(
          Effect.map((id): CatalogModelSource => ({ id, source })),
        )),
        customModels: customModelSources(local.state.models),
        hiddenModels: customModelSources(local.state.hidden),
      }).pipe(Effect.map(({ catalogModels, customModels, hiddenModels }): LocalModelSourcesState => ({
        catalogRevision: catalog.state.revision,
        localModelsRevision: local.state.revision,
        reconciliationComplete: catalog.state.reconciliationComplete && local.state.reconciliationComplete,
        catalogModels,
        customModels,
        hiddenModels,
        unsupported: local.state.unsupported,
      })))),
      Effect.orDie,
    )
    const projection = yield* materializeProjection({
      project,
      invalidations: Stream.merge(catalog.changes, localModels.changes),
      equivalent: (left, right) => left.catalogRevision === right.catalogRevision
        && left.localModelsRevision === right.localModelsRevision,
    })
    return LocalModelSources.of({
      state: projection.get,
      changes: projection.changes,
      refreshLocalModels: localModels.reconcile,
    })
  }))
