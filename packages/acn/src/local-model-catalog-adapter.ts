import { Context, Effect, Layer, Option, Stream } from "effect"
import type {
  CatalogIdentity,
  ModelServingConfiguration,
  RecommendableModel,
} from "@magnitudedev/acn-protocol"
import { IcnModels, type IcnModelsService } from "@magnitudedev/icn"
import type { InferenceModel } from "@magnitudedev/icn-protocol/schemas"
import {
  catalogIdentityFromIcn,
  catalogModelDefinitionFromIcn,
  catalogModelEffectiveConfigurationFromIcn,
} from "./local-model-icn-adapter"
import { materializeProjection } from "./materialized-projection"

export interface AdaptedLocalModelCatalogEntry {
  readonly source: InferenceModel
  readonly identity: CatalogIdentity
  readonly model: RecommendableModel
  readonly effectiveConfiguration: Option.Option<ModelServingConfiguration>
}

export interface AdaptedLocalModelCatalog {
  readonly sourceRevision: number
  readonly reconciliationComplete: boolean
  readonly entries: readonly AdaptedLocalModelCatalogEntry[]
}

export interface LocalModelCatalogAdapterApi {
  readonly state: Effect.Effect<AdaptedLocalModelCatalog>
  readonly changes: Stream.Stream<AdaptedLocalModelCatalog>
  readonly refresh: IcnModelsService["refresh"]
}

export class LocalModelCatalogAdapter extends Context.Tag("LocalModelCatalogAdapter")<
  LocalModelCatalogAdapter,
  LocalModelCatalogAdapterApi
>() {}

export const LocalModelCatalogAdapterLive: Layer.Layer<
  LocalModelCatalogAdapter,
  never,
  IcnModels
> = Layer.scoped(LocalModelCatalogAdapter, Effect.gen(function* () {
  const models = yield* IcnModels
  const project = models.get.pipe(
    Effect.flatMap((snapshot) => Effect.forEach(
      snapshot.state.models,
      (source) => Effect.all({
        source: Effect.succeed(source),
        identity: catalogIdentityFromIcn(source),
        model: catalogModelDefinitionFromIcn(source),
        effectiveConfiguration: catalogModelEffectiveConfigurationFromIcn(source),
      }),
    ).pipe(Effect.map((entries): AdaptedLocalModelCatalog => ({
      sourceRevision: snapshot.revision,
      reconciliationComplete: snapshot.state.reconciliationComplete,
      entries,
    })))),
    Effect.orDie,
  )
  const projection = yield* materializeProjection({
    project,
    invalidations: models.changes,
    equivalent: (left, right) => left.sourceRevision === right.sourceRevision,
  })

  return LocalModelCatalogAdapter.of({
    state: projection.get,
    changes: projection.changes,
    refresh: models.refresh,
  })
}))
