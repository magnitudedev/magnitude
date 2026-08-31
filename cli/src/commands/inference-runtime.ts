import { Atom, Registry } from "@effect-atom/atom"
import { FetchHttpClient } from "@effect/platform"
import { Client, Mutation } from "@magnitudedev/effect-query"
import {
  LocalModelSchema,
  CatalogFormModelIdSchema,
  localModelIsInstalled,
  localModelStorageBytes,
  MagnitudeBoundary,
  ModelIdSchema,
  ProviderCatalogFailureSchema,
  type CatalogFormModelId,
  type MagnitudeImplementationError,
  magnitudeImplementationsLayer,
} from "@magnitudedev/sdk"
import { Effect, Layer, Option, Schema } from "effect"
import { existingAcnConnection } from "../server/acn-connection"
import { ensureTrailingNewline, runCommand } from "./output"

const decodeModelId = Schema.decodeUnknown(ModelIdSchema)
const decodeCatalogFormModelId = Schema.decodeUnknown(CatalogFormModelIdSchema)
type ModelId = typeof ModelIdSchema.Type
type CliModelsClient = Pick<
  Client.Materialized<typeof MagnitudeBoundary, unknown, MagnitudeImplementationError>,
  "Models"
>

const PullResultSchema = Schema.Struct({
  operation: Schema.Literal("pull"),
  modelId: ModelIdSchema,
  outcome: Schema.Literal("Started", "AlreadyCurrent"),
})
const AddressedMutationResultSchema = Schema.Struct({
  operation: Schema.Literal("remove", "cancel", "load"),
  modelId: ModelIdSchema,
})
const StopResultSchema = Schema.Struct({ operation: Schema.Literal("stop") })
const ModelsStatusSchema = Schema.Struct({ models: Schema.Array(LocalModelSchema) })
const CatalogResultSchema = Schema.Union(
  Schema.TaggedStruct("Initializing", {}),
  Schema.Struct({
    _tag: Schema.Literal("Ready", "Refreshing", "Degraded"),
    models: Schema.Array(LocalModelSchema),
    failures: Schema.Array(ProviderCatalogFailureSchema),
    localModelsReconciliationComplete: Schema.Boolean,
  }),
)

const formatBytes = (bytes: number): string => {
  const units = ["B", "KB", "MB", "GB", "TB"] as const
  let value = bytes
  let unit = 0
  while (value >= 1_000 && unit < units.length - 1) {
    value /= 1_000
    unit += 1
  }
  return `${value >= 10 || unit === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`
}

const percent = (completed: number, total: number): string =>
  `${total === 0 ? 0 : Math.round((completed / total) * 100)}%`

const modelResidency = (model: typeof LocalModelSchema.Type) => {
  if (model._tag === "Discovered") {
    return model.state._tag === "Ready" ? model.state.residencyState : undefined
  }
  return "residencyState" in model.acquisitionState
    ? model.acquisitionState.residencyState
    : undefined
}

const catalogPresentation = (state: typeof CatalogResultSchema.Type): string => {
  if (state._tag === "Initializing") return "Model catalog is initializing.\n"
  const rows = state.models.filter((model) => model._tag === "Catalog").map((model) => {
    const catalog = model.catalogData
    const assessment = model.servingState._tag === "Assessed"
      ? model.servingState.assessment : undefined
    const speculativeMethod = model.servingState._tag === "Assessed"
      ? Option.getOrUndefined(model.servingState.speculativeMethod)
      : undefined
    const requiredBytes = assessment?._tag === "Fits"
      ? assessment.memory.totalRequiredBytes
      : assessment?._tag === "DoesNotFit" ? assessment.totalRequiredBytes : undefined
    return [
      model.modelId,
      `${model.presentation.displayName} (${model.presentation.variantLabel})`,
      model.servingState._tag === "Assessed"
        ? model.servingState.metadata.quantizationName
        : model.presentation.variantLabel,
      `QAT: ${catalog.quantizationAware ? "yes" : "no"}`,
      requiredBytes === undefined
        ? `storage: ${Option.match(localModelStorageBytes(model), { onNone: () => "unknown", onSome: formatBytes })}`
        : `memory: ${formatBytes(requiredBytes)}`,
      assessment?._tag ?? model.servingState._tag,
      `AA: ${catalog.intelligence.score}`,
      speculativeMethod?._tag ?? "standard",
      model.acquisitionState._tag,
      model.presentation.sourceUrls[0] ?? "catalog",
    ].join("  ")
  })
  return ensureTrailingNewline(rows.join("\n"))
}

const modelStatus = (model: typeof LocalModelSchema.Type): string => {
  if (model._tag === "Discovered") return modelResidency(model)?._tag ?? model.state._tag
  const acquisition = model.acquisitionState
  if (acquisition._tag === "Installing" || acquisition._tag === "Updating") {
    return `${acquisition._tag} ${percent(acquisition.progress.completedBytes, acquisition.progress.totalBytes)}`
  }
  if (!("residencyState" in acquisition)) return acquisition._tag
  const residency = acquisition.residencyState
  if (residency._tag === "Loading" && Option.isSome(residency.progress)) {
    return `Loading ${Math.round(residency.progress.value * 100)}%`
  }
  return residency._tag
}

const withClient = <A>(
  use: (client: CliModelsClient, registry: Registry.Registry) => Effect.Effect<A, unknown>,
) => Effect.scoped(Effect.gen(function* () {
  const connection = yield* existingAcnConnection
  const registry = Registry.make()
  yield* Effect.addFinalizer(() => Effect.sync(() => registry.dispose()))
  const client = Client.make(
    MagnitudeBoundary,
    magnitudeImplementationsLayer(connection.protocolLayer.pipe(
      Layer.provide(FetchHttpClient.layer),
    )),
  )
  return yield* use(client, registry)
}))

export const showModelCatalog = () => runCommand({
  effect: withClient((client, registry) => Registry.getResult(
    registry,
    Atom.make((get) => get(client.Models.GetCatalog({})).result),
  ).pipe(Effect.map((state) => state._tag === "Initializing"
    ? state
    : {
        ...state,
        models: state.models.flatMap((entry) => entry._tag === "Local"
          ? [entry.product]
          : []),
      }))),
  schema: CatalogResultSchema,
  render: catalogPresentation,
})

export const pullModel = (modelInput: string) => runCommand({
  effect: withClient((client, registry) => decodeCatalogFormModelId(modelInput).pipe(
    Effect.flatMap((modelId) => Mutation.execute(client.Models.SyncLocalModel, {
      modelId,
    }).pipe(Effect.map(({ outcome }) => ({
      operation: "pull" as const,
      modelId,
      outcome,
    })))),
    Effect.provideService(Registry.AtomRegistry, registry),
  )),
  schema: PullResultSchema,
  render: ({ modelId, outcome }) => outcome === "AlreadyCurrent"
    ? `${modelId} is already up to date.\n`
    : `Pulling ${modelId} in the background.\n`,
})

const modelMutation = <Id extends ModelId>(
  modelInput: string,
  operation: "remove" | "cancel" | "load",
  decode: (input: string) => Effect.Effect<Id, unknown>,
  execute: (client: CliModelsClient, registry: Registry.Registry, modelId: Id) => Effect.Effect<unknown, unknown>,
) => runCommand({
  effect: withClient((client, registry) => decode(modelInput).pipe(
    Effect.flatMap((modelId) => execute(client, registry, modelId).pipe(
      Effect.as({ operation, modelId }),
    )),
  )),
  schema: AddressedMutationResultSchema,
  render: ({ modelId }) => `${operation === "remove" ? "Removed" : operation === "cancel" ? "Cancelled work for" : "Loading"} ${modelId}.\n`,
})

export const removeModel = (modelId: string) => modelMutation<CatalogFormModelId>(modelId, "remove", decodeCatalogFormModelId,
  (client, registry, decoded) => Mutation.execute(client.Models.RemoveLocalModel, {
    modelId: decoded,
  }).pipe(Effect.provideService(Registry.AtomRegistry, registry)))

export const cancelDownload = (modelId: string) => modelMutation<CatalogFormModelId>(modelId, "cancel", decodeCatalogFormModelId,
  (client, registry, decoded) => Mutation.execute(client.Models.CancelLocalModelSync, {
    modelId: decoded,
  }).pipe(Effect.provideService(Registry.AtomRegistry, registry)))

export const listInstances = () => runCommand({
  effect: withClient((client, registry) => Registry.getResult(
    registry,
    Atom.make((get) => get(client.Models.GetCatalog({})).result),
  ).pipe(Effect.map((state) => ({
    models: state._tag === "Initializing" ? [] : state.models.flatMap((entry) =>
      entry._tag === "Local" && localModelIsInstalled(entry.product) ? [entry.product] : []),
  })))),
  schema: ModelsStatusSchema,
  render: ({ models }) => models.length === 0
    ? "No installed models.\n"
    : ensureTrailingNewline(models.map((model) =>
        `${model.modelId}  ${model.presentation.variantLabel}  ${Option.match(localModelStorageBytes(model), { onNone: () => "unknown", onSome: formatBytes })}  ${modelStatus(model)}`
      ).join("\n")),
})

export const loadInstance = (modelId: string) => modelMutation(modelId, "load", decodeModelId,
  (client, registry, decoded) => Mutation.execute(client.Models.LoadLocalModel, {
    modelId: decoded,
  }).pipe(Effect.provideService(Registry.AtomRegistry, registry)))

export const stopInstance = () => runCommand({
  effect: withClient((client, registry) => Mutation.execute(
    client.Models.StopActiveLocalModel,
    {},
  ).pipe(
    Effect.provideService(Registry.AtomRegistry, registry),
    Effect.as({ operation: "stop" as const }),
  )),
  schema: StopResultSchema,
  render: () => "Stopped the active local model.\n",
})
