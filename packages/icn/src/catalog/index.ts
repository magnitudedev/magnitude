import { Context, Duration, Effect, Layer, Schema, Stream } from "effect"
import { RecommendableModelCatalog } from "@magnitudedev/icn-protocol/schemas"
import { IcnModels, type IcnModelsService } from "../models/index.js"
import type { IcnObservedSnapshot } from "../observed-state.js"

export interface IcnCatalogService {
  readonly get: Effect.Effect<IcnObservedSnapshot<RecommendableModelCatalog>>
  readonly changes: Stream.Stream<IcnObservedSnapshot<RecommendableModelCatalog>>
  readonly ready: Effect.Effect<boolean>
  readonly refresh: Effect.Effect<void, unknown>
}

export class IcnCatalog extends Context.Tag("@magnitudedev/icn/IcnCatalog")<
  IcnCatalog,
  IcnCatalogService
>() {}

export interface IcnCatalogOptions {
  readonly refreshInterval?: Duration.DurationInput
}

const catalogSnapshot = (
  snapshot: Effect.Effect.Success<IcnModelsService["get"]>,
) =>
  Schema.validate(RecommendableModelCatalog)({
    models: snapshot.state.catalogModels.map((model) => ({
      ...model,
      configuration: model.desiredConfiguration,
    })),
    diagnostics: snapshot.state.diagnostics,
  }).pipe(Effect.map((state) => ({ revision: snapshot.revision, state })))

export const makeIcnCatalog = (
  _options: IcnCatalogOptions = {},
): Layer.Layer<IcnCatalog, never, IcnModels> =>
  Layer.effect(
    IcnCatalog,
    Effect.gen(function* () {
      const models = yield* IcnModels
      return IcnCatalog.of({
        get: models.get.pipe(Effect.flatMap(catalogSnapshot), Effect.orDie),
        changes: models.changes.pipe(Stream.mapEffect((snapshot) =>
          catalogSnapshot(snapshot).pipe(Effect.orDie))),
        ready: models.initialized,
        refresh: models.refresh,
      })
    }),
  )
