import { Atom, Registry } from "@effect-atom/atom"
import { FetchHttpClient } from "@effect/platform"
import { Client, Mutation } from "@magnitudedev/effect-query"
import {
  LocalModelSchema,
  LocalModelDiscoveryStateSchema,
  LocalModelInventoryStateSchema,
  MagnitudeBoundary,
  ProviderModelIdSchema,
  ProviderCatalogFailureSchema,
  type MagnitudeImplementationError,
  magnitudeImplementationsLayer,
} from "@magnitudedev/sdk"
import { Effect, Layer, Option, Schema } from "effect"
import { existingAcnConnection } from "../server/acn-connection"
import { ensureTrailingNewline, runCommand } from "./output"

const decodeModelId = Schema.decodeUnknown(ProviderModelIdSchema)
type ModelId = typeof ProviderModelIdSchema.Type
type CliModelsClient = Pick<
  Client.Materialized<typeof MagnitudeBoundary, unknown, MagnitudeImplementationError>,
  "Models"
>

const PullResultSchema = Schema.Struct({
  operation: Schema.Literal("pull"),
  modelId: ProviderModelIdSchema,
  outcome: Schema.Literal("Started", "AlreadyCurrent"),
})
const AddressedMutationResultSchema = Schema.Struct({
  operation: Schema.Literal("remove", "cancel", "load"),
  modelId: ProviderModelIdSchema,
})
const StopResultSchema = Schema.Struct({ operation: Schema.Literal("stop") })
const ModelsStatusSchema = Schema.Struct({ models: Schema.Array(LocalModelSchema) })
const CatalogResultSchema = Schema.Union(
  Schema.TaggedStruct("Initializing", {}),
  Schema.Struct({
    _tag: Schema.Literal("Ready", "Refreshing", "Degraded"),
    models: Schema.Array(LocalModelSchema),
    failures: Schema.Array(ProviderCatalogFailureSchema),
    localInventoryState: LocalModelInventoryStateSchema,
    localDiscoveryState: LocalModelDiscoveryStateSchema,
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

const catalogPresentation = (state: typeof CatalogResultSchema.Type): string => {
  if (state._tag === "Initializing") return "Model catalog is initializing.\n"
  const rows = state.models.map((model) => {
    const target = model.bundle._tag === "Standalone" ? model.bundle.package : model.bundle.target
    const catalog = model.catalogMembershipState._tag === "InCatalog"
      ? model.catalogMembershipState.catalogData : undefined
    const assessment = model.servingState._tag === "Assessed"
      ? model.servingState.assessment : undefined
    const requiredBytes = assessment?._tag === "Fits"
      ? assessment.memory.totalRequiredBytes
      : assessment?._tag === "DoesNotFit" ? assessment.totalRequiredBytes : undefined
    const source = target.source._tag === "HuggingFace"
      ? `https://huggingface.co/${target.source.repository}` : target.source._tag
    return [
      model.modelId,
      `${model.presentation.displayName} (${model.presentation.variantLabel})`,
      target.properties.quantizationName,
      `QAT: ${catalog?.quantizationAware === true ? "yes" : "no"}`,
      requiredBytes === undefined ? `download: ${formatBytes(model.downloadBytes)}` : `memory: ${formatBytes(requiredBytes)}`,
      assessment?._tag ?? model.servingState._tag,
      catalog === undefined ? "AA: pending" : `AA: ${catalog.intelligence.score}`,
      model.bundle._tag === "SpeculativeDecoding" ? model.bundle.method._tag : "standard",
      model.acquisitionState._tag,
      source,
    ].join("  ")
  })
  const assessment = state.localDiscoveryState.progress.find(({ id }) => id === "assessment")
  const pending = assessment !== undefined
    && Option.isSome(assessment.totalItems)
    && Option.isSome(assessment.completedItems)
    ? assessment.totalItems.value - assessment.completedItems.value
    : 0
  return ensureTrailingNewline([
    ...rows,
    ...(pending > 0 ? [`${pending} model${pending === 1 ? " is" : "s are"} still being assessed.`] : []),
  ].join("\n"))
}

const modelStatus = (model: typeof LocalModelSchema.Type): string => {
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
  effect: withClient((client, registry) => decodeModelId(modelInput).pipe(
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

const modelMutation = (
  modelInput: string,
  operation: "remove" | "cancel" | "load",
  execute: (client: CliModelsClient, registry: Registry.Registry, modelId: ModelId) => Effect.Effect<unknown, unknown>,
) => runCommand({
  effect: withClient((client, registry) => decodeModelId(modelInput).pipe(
    Effect.flatMap((modelId) => execute(client, registry, modelId).pipe(
      Effect.as({ operation, modelId }),
    )),
  )),
  schema: AddressedMutationResultSchema,
  render: ({ modelId }) => `${operation === "remove" ? "Removed" : operation === "cancel" ? "Cancelled work for" : "Loading"} ${modelId}.\n`,
})

export const removeModel = (modelId: string) => modelMutation(modelId, "remove",
  (client, registry, decoded) => Mutation.execute(client.Models.RemoveLocalModel, {
    modelId: decoded,
  }).pipe(Effect.provideService(Registry.AtomRegistry, registry)))

export const cancelDownload = (modelId: string) => modelMutation(modelId, "cancel",
  (client, registry, decoded) => Mutation.execute(client.Models.CancelLocalModelSync, {
    modelId: decoded,
  }).pipe(Effect.provideService(Registry.AtomRegistry, registry)))

export const listInstances = () => runCommand({
  effect: withClient((client, registry) => Registry.getResult(
    registry,
    Atom.make((get) => get(client.Models.GetCatalog({})).result),
  ).pipe(Effect.map((state) => ({
    models: state._tag === "Initializing" ? [] : state.models.flatMap((entry) =>
      entry._tag === "Local" && entry.product.acquisitionState._tag !== "NotInstalled"
        && entry.product.acquisitionState._tag !== "InstallFailed" ? [entry.product] : []),
  })))),
  schema: ModelsStatusSchema,
  render: ({ models }) => models.length === 0
    ? "No installed models.\n"
    : ensureTrailingNewline(models.map((model) =>
        `${model.modelId}  ${model.presentation.variantLabel}  ${formatBytes(model.downloadBytes)}  ${modelStatus(model)}`
      ).join("\n")),
})

export const loadInstance = (modelId: string) => modelMutation(modelId, "load",
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
