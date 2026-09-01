import { Context, Effect, Layer, Option, Stream } from "effect"
import {
  Models,
  type LocalModelsState,
  type ModelCatalogEntry,
  type ModelCatalogState,
  type ProviderCatalogEntry,
  type ProviderCatalogFailure,
  type ProviderModelCatalogEntry,
  type ProviderModelCatalogState,
} from "@magnitudedev/acn-protocol"
import type { ProviderId } from "@magnitudedev/sdk"
import { PROVIDER_ID as LOCAL_PROVIDER_ID } from "@magnitudedev/icn/provider"
import { AcnChanges } from "./changes"
import { LocalModels } from "./local-models"
import { ProviderModelCatalog } from "./provider-model-catalog"

export interface ModelCatalogApi {
  readonly state: Effect.Effect<ModelCatalogState>
  readonly changes: Stream.Stream<void>
  readonly refresh: (providerId: Option.Option<ProviderId>) => Effect.Effect<void>
}

export class ModelCatalog extends Context.Tag("ModelCatalog")<ModelCatalog, ModelCatalogApi>() {}

const providerContents = (state: ProviderModelCatalogState): {
  readonly providers: readonly ProviderCatalogEntry[]
  readonly models: readonly ProviderModelCatalogEntry[]
  readonly failures: readonly ProviderCatalogFailure[]
} => {
  switch (state._tag) {
    case "Loading": return { providers: [], models: [], failures: [] }
    case "Ready": return { providers: state.providers, models: state.models, failures: [] }
    case "Refreshing":
    case "Degraded": return { providers: state.providers, models: state.models, failures: state.failures }
    case "Unavailable": return { providers: state.providers, models: [], failures: state.failures }
  }
}

export const projectModelCatalog = (
  providers: ProviderModelCatalogState,
  local: LocalModelsState,
): ModelCatalogState => {
  const contents = providerContents(providers)
  const localOfferings = new Map(contents.models
    .filter((model) => model.providerId === LOCAL_PROVIDER_ID)
    .map((model) => [model.providerModelId, model] as const))
  const models: ModelCatalogEntry[] = [
    ...contents.models
      .filter((model) => model.providerId !== LOCAL_PROVIDER_ID)
      .map((offering): ModelCatalogEntry => ({ _tag: "Remote", offering })),
    ...local.models.map((product): ModelCatalogEntry => ({
      _tag: "Local",
      product,
      offering: Option.fromNullable(localOfferings.get(product.modelId)),
    })),
  ]
  const failures = [...contents.failures]
  const fields = {
    providers: contents.providers,
    models,
    failures,
    localModelPreparation: local.preparation,
  }
  if (failures.length > 0) return { _tag: "Degraded", ...fields }
  if (providers._tag === "Loading" || providers._tag === "Refreshing" || !local.preparation.discovery.complete) {
    return { _tag: "Refreshing", ...fields }
  }
  return { _tag: "Ready", ...fields }
}

export const ModelCatalogLive: Layer.Layer<
  ModelCatalog,
  never,
  ProviderModelCatalog | LocalModels | AcnChanges
> = Layer.scoped(ModelCatalog, Effect.gen(function* () {
  const providers = yield* ProviderModelCatalog
  const local = yield* LocalModels
  const changes = yield* AcnChanges
  const sourceChanges = Stream.merge(
    providers.changes.pipe(Stream.map(() => undefined)),
    local.changes.pipe(Stream.map(() => undefined)),
  )
  yield* sourceChanges.pipe(
    Stream.runForEach(() => changes.publish({ query: Models.GetCatalog.name })),
    Effect.forkScoped,
  )
  return ModelCatalog.of({
    state: Effect.zipWith(providers.state, local.state, projectModelCatalog),
    changes: sourceChanges,
    refresh: (providerId) => Option.match(providerId, {
      onNone: () => Effect.all([providers.refresh(Option.none()), local.refresh], {
        concurrency: "unbounded",
        discard: true,
      }).pipe(Effect.catchAllCause((cause) => Effect.logWarning("Unable to refresh model catalog").pipe(
        Effect.annotateLogs({ cause: String(cause) }),
      ))),
      onSome: (id) => providers.refresh(Option.some(id)),
    }),
  })
}))
