import {
  deriveHardwareMemoryView,
  formatLocalModelDisplayName,
  formatMemorySize,
  formatSpeculativeMethod,
  formatStorageSize,
  LOCAL_MODEL_RANKING_SCALE_LABELS,
  LOCAL_MODEL_RANKING_SCALE_VALUES,
  localModelIsInstalled,
  localModelRadarAxes,
  localModelServingProfile,
  localModelServingState,
  modelDownloadFailureMessage,
  rankedLocalModelOptions,
  targetPhysicalMemoryBytes,
} from "@magnitudedev/client-common"
import {
  CatalogFormModelIdSchema,
  type MagnitudeClient,
  ModelIdSchema,
  type CatalogLocalModel,
  type LocalInferenceHardware,
  type LocalModel,
  type ModelCatalogState,
  type ModelId,
} from "@magnitudedev/sdk"
import { Data, Effect, Option, Schema } from "effect"
import { existingAcnConnection } from "../server/acn-connection"
import {
  describeLocalHardware,
  formatContext,
} from "../features/local-inference/view-model"
import {
  ensureTrailingNewline,
  renderFields,
  renderTable,
  runCommand,
} from "./output"

type CliModelsClient = Pick<MagnitudeClient, "models">

class ModelCommandError extends Data.TaggedError("ModelCommandError")<{
  readonly message: string
}> {}

const withClient = <A>(use: (client: CliModelsClient) => Effect.Effect<A, unknown>) =>
  Effect.scoped(Effect.gen(function* () {
    const connection = yield* existingAcnConnection
    return yield* use(connection.client)
  }))

const readCatalog = (client: CliModelsClient) => client.models.getCatalog({})
const readHardware = (client: CliModelsClient) => client.models.getLocalEnvironment({})

const modelsForStatus = (models: readonly LocalModel[]): LocalModel[] => models
  .filter(model => model._tag === "Discovered" || model.acquisitionState._tag !== "NotInstalled")
  .sort((a,b) => formatLocalModelDisplayName(a).localeCompare(formatLocalModelDisplayName(b)) || a.modelId.localeCompare(b.modelId))

const localModels = (catalog: ModelCatalogState): readonly LocalModel[] =>
  catalog._tag === "Initializing" ? [] : catalog.models.flatMap((entry) =>
    entry._tag === "Local" ? [entry.product] : [])

export const renderCatalogStatus = (catalog: ModelCatalogState): string => {
  if (catalog._tag === "Initializing") return ensureTrailingNewline([
    "Model catalog preparation",
    "Discovery: In progress",
    "Assessment: In progress",
  ].join("\n"))

  const { discovery, assessment } = catalog.localModelPreparation
  return ensureTrailingNewline([
    "Model catalog preparation",
    `Discovery: ${discovery.complete ? "Complete" : "In progress"} - ${discovery.modelsFound} model${discovery.modelsFound === 1 ? "" : "s"} found`,
    `Assessment: ${assessment.complete ? "Complete" : "In progress"} - ${assessment.settledModels} of ${assessment.totalModels} model${assessment.totalModels === 1 ? "" : "s"} assessed`,
  ].join("\n"))
}

export const showCatalogStatus = () => runCommand({
  effect: withClient(readCatalog),
  render: renderCatalogStatus,
})

const requireLocalModels = (catalog: ModelCatalogState): Effect.Effect<readonly LocalModel[], ModelCommandError> =>
  catalog._tag === "Initializing"
    ? Effect.fail(new ModelCommandError({ message: "Local models are initializing. Try again shortly." }))
    : Effect.succeed(localModels(catalog))

const decodeCatalogId = (input: string) => Schema.decodeUnknown(CatalogFormModelIdSchema)(input).pipe(
  Effect.mapError(() => new ModelCommandError({ message: `Invalid catalog model ID: ${input}` })),
)

const decodeModelId = (input: string) => Schema.decodeUnknown(ModelIdSchema)(input).pipe(
  Effect.mapError(() => new ModelCommandError({ message: `Invalid model ID: ${input}` })),
)

const findModel = <M extends LocalModel>(
  models: readonly M[],
  modelId: string,
  kind = "model",
): Effect.Effect<M, ModelCommandError> => {
  const model = models.find((candidate) => candidate.modelId === modelId)
  return model === undefined
    ? Effect.fail(new ModelCommandError({ message: `Unknown ${kind}: ${modelId}` }))
    : Effect.succeed(model)
}

const servingState = (model: LocalModel) => Option.getOrUndefined(localModelServingState(model))

const residencyState = (model: LocalModel) => model._tag === "Discovered"
  ? model.state._tag === "Ready" ? model.state.residencyState : undefined
  : "residencyState" in model.acquisitionState ? model.acquisitionState.residencyState : undefined

const modelContext = (model: LocalModel): string => Option.match(localModelServingProfile(model), {
  onNone: () => "—",
  onSome: ({ contextLength }) => formatContext(contextLength),
})

const modelMemoryBytes = (model: LocalModel): number | undefined => {
  const serving = servingState(model)
  if (serving?._tag !== "Assessed") return undefined
  return serving.assessment._tag === "Fits"
    ? serving.assessment.memory.totalRequiredBytes
    : serving.assessment._tag === "DoesNotFit" ? serving.assessment.totalRequiredBytes : undefined
}

const modelMemory = (model: LocalModel): string => {
  const bytes = modelMemoryBytes(model)
  return bytes === undefined ? "—" : formatMemorySize(bytes)
}

const assessmentSummary = (models: readonly LocalModel[]): { assessing: number; failed: number } => ({
  assessing: models.filter((model) => model._tag === "Catalog" && model.servingState._tag === "Assessing").length,
  failed: models.filter((model) => model._tag === "Catalog" && model.servingState._tag === "Failed").length,
})

const assessmentNotice = (
  models: readonly LocalModel[],
  subject: "list" | "recommendations",
): string[] => {
  const { assessing, failed } = assessmentSummary(models)
  return [
    ...(assessing > 0 ? [`Assessing ${assessing} additional catalog model${assessing === 1 ? "" : "s"}; ${
      subject === "list" ? "this list may grow" : "recommendations may change"
    }.`] : []),
    ...(failed > 0 ? [`Assessment failed for ${failed} catalog model${failed === 1 ? "" : "s"}.`] : []),
  ]
}

const catalogWarnings = (catalog: ModelCatalogState): string[] => catalog._tag === "Initializing"
  ? []
  : [...new Set(catalog.failures.map(({ message }) => message.trim()).filter(Boolean))]
    .map((message) => `Catalog warning: ${message}`)

const fittingCatalogModels = (models: readonly LocalModel[]): CatalogLocalModel[] => models
  .filter((model): model is CatalogLocalModel => model._tag === "Catalog"
    && model.servingState._tag === "Assessed"
    && model.servingState.assessment._tag === "Fits")
  .sort((left, right) => left.presentation.displayName.localeCompare(right.presentation.displayName)
    || String(left.presentation.variantLabel).localeCompare(String(right.presentation.variantLabel))
    || left.modelId.localeCompare(right.modelId))

const radarDetail = (model: LocalModel, label: string): string =>
  Option.getOrUndefined(localModelRadarAxes(model))?.find((axis) => axis.label === label)?.detail ?? "—"

const speedLabel = (model: CatalogLocalModel): string => radarDetail(model, "SPEED")

const accelerationLabel = (model: CatalogLocalModel): string =>
  model.servingState._tag === "Assessed"
    ? Option.match(model.servingState.speculativeMethod, {
        onNone: () => "None",
        onSome: formatSpeculativeMethod,
      })
    : "None"

export const renderCatalog = (catalog: ModelCatalogState): string => {
  if (catalog._tag === "Initializing") return "The local model catalog is initializing.\n"
  const models = localModels(catalog)
  const compatible = fittingCatalogModels(models)
  const notice = [...assessmentNotice(models, "list"), ...catalogWarnings(catalog)]
  if (compatible.length === 0) {
    const lead = notice.length > 0
      ? "No compatible catalog models are available yet."
      : "No catalog models are compatible with this computer."
    return ensureTrailingNewline([lead, ...notice].join("\n"))
  }
  const table = renderTable(compatible, [
    { heading: "MODEL", value: formatLocalModelDisplayName },
    { heading: "MEMORY", value: modelMemory },
    { heading: "SPEED", value: speedLabel },
    { heading: "CONTEXT", value: modelContext },
    { heading: "ACCELERATION", value: accelerationLabel },
    { heading: "MODEL ID", value: ({ modelId }) => modelId },
  ])
  return ensureTrailingNewline([
    `Local model catalog - ${compatible.length} compatible model${compatible.length === 1 ? "" : "s"}`,
    "",
    table.trimEnd(),
    ...(notice.length > 0 ? ["", ...notice] : []),
  ].join("\n"))
}

export const showModelCatalog = () => runCommand({
  effect: withClient(readCatalog),
  render: renderCatalog,
})

const preferenceNames = ["fastest", "faster", "balanced", "smarter", "smartest"] as const

const parsePreference = (input: string) => {
  const index = preferenceNames.indexOf(input.toLowerCase() as typeof preferenceNames[number])
  return index < 0
    ? Effect.fail(new ModelCommandError({
        message: "Preference must be one of: fastest, faster, balanced, smarter, smartest",
      }))
    : Effect.succeed({
        label: LOCAL_MODEL_RANKING_SCALE_LABELS[index]!,
        value: LOCAL_MODEL_RANKING_SCALE_VALUES[index]!,
      })
}

const parseLimit = (input: string) => {
  const limit = Number(input)
  return Number.isSafeInteger(limit) && limit > 0
    ? Effect.succeed(limit)
    : Effect.fail(new ModelCommandError({ message: "Limit must be a positive integer" }))
}

const capabilityLabels = (model: CatalogLocalModel): string[] => {
  if (model.servingState._tag !== "Assessed") return []
  const capabilities = model.servingState.capabilities
  return [
    ...(capabilities.vision ? ["Vision"] : []),
    ...(capabilities.tools ? ["tools"] : []),
    ...(capabilities.structuredOutput ? ["structured output"] : []),
    ...(capabilities.reasoning.supported ? ["reasoning"] : []),
  ]
}

const renderRecommendation = (model: CatalogLocalModel, index: number): string => {
  const capabilities = capabilityLabels(model)
  return [
    `${index + 1}. ${formatLocalModelDisplayName(model)}`,
    `   ID: ${model.modelId}`,
    `   ${speedLabel(model)} - ${modelMemory(model)} memory - ${modelContext(model)} context`,
    `   Intelligence ${Math.round(model.catalogData.intelligence.score)}% - Accuracy ${radarDetail(model, "ACCURACY")} - Acceleration ${accelerationLabel(model)}`,
    ...(capabilities.length > 0 ? [`   ${capabilities.join(", ")}`] : []),
  ].join("\n")
}

export const showRecommendations = (preferenceInput: string, limitInput: string) => runCommand({
  effect: withClient((client) => Effect.gen(function* () {
    const [catalog, hardware, preference, limit] = yield* Effect.all([
      readCatalog(client),
      readHardware(client),
      parsePreference(preferenceInput),
      parseLimit(limitInput),
    ])
    const models = localModels(catalog)
    const ranked = rankedLocalModelOptions(models.map((model) => ({
      id: String(model.modelId),
      kind: localModelIsInstalled(model) ? "stored" as const : "downloadable" as const,
      model,
    })), {
      fastToSmart: preference.value,
      memoryBudgetBytes: targetPhysicalMemoryBytes(hardware),
    }, limit).flatMap(({ model }) => model._tag === "Catalog" ? [model] : [])
    return { catalog, models, hardware, preference, ranked }
  })),
  render: renderRecommendations,
})

export function renderRecommendations({
  catalog,
  models,
  hardware,
  preference,
  ranked,
}: {
  readonly catalog: ModelCatalogState
  readonly models: readonly LocalModel[]
  readonly hardware: LocalInferenceHardware
  readonly preference: { readonly label: string }
  readonly ranked: readonly CatalogLocalModel[]
}): string {
  if (catalog._tag === "Initializing") return "The local model catalog is initializing.\n"
  const system = describeLocalHardware(hardware).system.name
  const notice = [...assessmentNotice(models, "recommendations"), ...catalogWarnings(catalog)]
  if (ranked.length === 0) return ensureTrailingNewline([
    `No compatible recommendations are available for ${preference.label}.`,
    ...notice,
  ].join("\n"))
  return ensureTrailingNewline([
    `Local model recommendations - ${preference.label}`,
    `${system} - ${formatMemorySize(hardware.totalSystemMemoryBytes)}`,
    "",
    ranked.map(renderRecommendation).join("\n\n"),
    ...(notice.length > 0 ? ["", ...notice] : []),
    "",
    "Learn more: magnitude docs recommendations",
  ].join("\n"))
}

const formatParameters = (model: CatalogLocalModel): string => {
  const format = (value: number) => value >= 1_000_000_000
    ? `${Number((value / 1_000_000_000).toPrecision(3))}B`
    : `${Number((value / 1_000_000).toPrecision(3))}M`
  const parameters = model.catalogData.parameterization
  return parameters.architecture === "dense"
    ? `Dense - ${format(parameters.totalParameters)}`
    : `MoE - ${format(parameters.totalParameters)} total / ${format(parameters.activeParameters)} active`
}

const formatReleaseDate = (value: string): string => {
  const date = new Date(`${value.length === 7 ? `${value}-01` : value}T00:00:00Z`)
  return Number.isNaN(date.valueOf())
    ? value
    : new Intl.DateTimeFormat("en", { month: "long", year: "numeric", timeZone: "UTC" }).format(date)
}

const renderCatalogDetail = (model: CatalogLocalModel): string => {
  const serving = model.servingState
  const header = [
    formatLocalModelDisplayName(model),
    ...(model.presentation.description.trim().length > 0 ? [model.presentation.description.trim()] : []),
    renderFields([
      ["Model ID", model.modelId],
      ["Released", formatReleaseDate(model.catalogData.releaseDate)],
      ["Architecture", formatParameters(model)],
      ["Context", modelContext(model)],
      ...(serving._tag === "Assessed" ? [["Capabilities", capabilityLabels(model).join(", ") || "Text"]] as const : []),
    ]),
  ]
  const performance = (() => {
    if (serving._tag === "Assessing") return ["Assessment", renderFields([["Status", "Assessing"]])]
    if (serving._tag === "Failed") return ["Assessment", renderFields([["Status", "Failed"], ["Reason", serving.failure.message]])]
    if (serving.assessment._tag === "DoesNotFit") return ["Performance on this machine", renderFields([
      ["Status", "Does not fit"],
      ["Memory required", formatMemorySize(serving.assessment.totalRequiredBytes)],
      ["Additional memory", formatMemorySize(serving.assessment.deficitBytes)],
    ])]
    if (serving.assessment._tag === "Incompatible") return ["Performance on this machine", renderFields([
      ["Status", "Incompatible"],
      ["Reason", serving.assessment.failure.message],
    ])]
    return ["Performance on this machine", renderFields([
      ["Speed", speedLabel(model)],
      ["Memory", radarDetail(model, "MEMORY")],
      ["Intelligence", `${Math.round(model.catalogData.intelligence.score)}%`],
      ["Accuracy", radarDetail(model, "ACCURACY")],
      ["Acceleration", accelerationLabel(model)],
    ])]
  })()
  return ensureTrailingNewline([
    ...header,
    "",
    ...performance,
    "",
    "Distribution",
    renderFields([
      ["Download size", formatStorageSize(model.storageBytes)],
      ...(Option.isSome(model.presentation.license) ? [["License", model.presentation.license.value]] as const : []),
      ...(model.presentation.sourceUrls[0] === undefined ? [] : [["Source", model.presentation.sourceUrls[0]]] as const),
    ]),
  ].join("\n"))
}

export const showCatalogModel = (modelInput: string) => runCommand({
  effect: withClient((client) => Effect.gen(function* () {
    const modelId = yield* decodeCatalogId(modelInput)
    const models = (yield* requireLocalModels(yield* readCatalog(client)))
      .filter((model): model is CatalogLocalModel => model._tag === "Catalog")
    return yield* findModel(models, modelId, "catalog model")
  })),
  render: renderCatalogDetail,
})

export const pullModel = (modelInput: string) => runCommand({
  effect: withClient((client) => decodeCatalogId(modelInput).pipe(
    Effect.flatMap((modelId) => client.models.syncLocalModel({ modelId }).pipe(
      Effect.map(({ outcome }) => ({ modelId, outcome })),
    )),
  )),
  render: ({ modelId, outcome }) => outcome === "AlreadyCurrent"
    ? `${modelId} is already installed and up to date.\n`
    : `Downloading or updating ${modelId}.\nCheck progress: magnitude models status ${modelId}\n`,
})

const modelMutation = <Id extends ModelId>(
  modelInput: string,
  decode: (input: string) => Effect.Effect<Id, unknown>,
  execute: (client: CliModelsClient, modelId: Id) => Effect.Effect<unknown, unknown>,
  render: (modelId: Id) => string,
) => runCommand({
  effect: withClient((client) => decode(modelInput).pipe(
    Effect.flatMap((modelId) => execute(client, modelId).pipe(
      Effect.as(modelId),
    )),
  )),
  render,
})

export const cancelDownload = (modelInput: string) => modelMutation(
  modelInput,
  decodeCatalogId,
  (client, modelId) => client.models.cancelLocalModelSync({ modelId }),
  (modelId) => `Cancelled work for ${modelId}.\n`,
)

export const removeModel = (modelInput: string) => modelMutation(
  modelInput,
  decodeCatalogId,
  (client, modelId) => client.models.removeLocalModel({ modelId }),
  (modelId) => `Removed ${modelId} from this computer.\nThe model remains available in the catalog.\n`,
)

const residencyLabel = (model: LocalModel): string => {
  const residency = residencyState(model)
  if (residency === undefined) return "Unloaded"
  switch (residency._tag) {
    case "Requested": return "Loading"
    case "Loading": return Option.match(residency.progress, {
      onNone: () => "Loading",
      onSome: (progress) => `Loading ${Math.round(progress * 100)}%`,
    })
    case "Failed": return `Failed - ${residency.failure.message}`
    default: return residency._tag
  }
}

const percent = (completed: number, total: number): string =>
  `${total === 0 ? 0 : Math.round((completed / total) * 100)}%`

const modelStatus = (model: LocalModel): string => {
  if (model._tag === "Discovered") {
    return model.state._tag === "Unavailable" ? `Failed - ${model.state.failure.message}` : residencyLabel(model)
  }
  const acquisition = model.acquisitionState
  switch (acquisition._tag) {
    case "Removing": return "Removing"
    case "Installing": return `Downloading ${percent(acquisition.progress.completedBytes, acquisition.progress.totalBytes)}`
    case "Updating": return `Updating ${percent(acquisition.progress.completedBytes, acquisition.progress.totalBytes)}`
    case "InstallFailed": return `Failed - ${modelDownloadFailureMessage(acquisition.failure)}`
    case "UpdateFailed": return `Failed - ${modelDownloadFailureMessage(acquisition.failure)}`
    case "RemoveFailed": return `Failed - ${acquisition.failure.message}`
    case "NotInstalled": return "Not installed"
    case "UpdateAvailable": {
      const residency = residencyLabel(model)
      return residency === "Unloaded" ? "Update available" : `${residency} - Update available`
    }
    case "Installed": return residencyLabel(model)
  }
}

export const renderModelsStatus = (models: readonly LocalModel[]): string => {
  const visible = modelsForStatus(models)
  if (visible.length === 0) return "No local models are on this computer.\n"
  return ensureTrailingNewline([
    "Local models",
    "",
    renderTable(visible, [
      { heading: "MODEL", value: formatLocalModelDisplayName },
      { heading: "MEMORY", value: modelMemory },
      { heading: "CONTEXT", value: modelContext },
      { heading: "STATUS", value: modelStatus },
      { heading: "MODEL ID", value: ({ modelId }) => modelId },
    ]).trimEnd(),
  ].join("\n"))
}

const installationFields = (model: LocalModel): readonly (readonly [string, string])[] => {
  if (model._tag === "Discovered") return [["Installation", model.state._tag === "Ready" ? "Installed" : "Unavailable"]]
  const state = model.acquisitionState
  if (state._tag === "Installing" || state._tag === "Updating") return [
    ["Installation", state._tag === "Installing" ? "Downloading" : "Updating"],
    ["Progress", `${percent(state.progress.completedBytes, state.progress.totalBytes)} - ${formatStorageSize(state.progress.completedBytes)} / ${formatStorageSize(state.progress.totalBytes)}`],
  ]
  if (state._tag === "InstallFailed" || state._tag === "UpdateFailed") return [
    ["Installation", "Failed"],
    ["Reason", modelDownloadFailureMessage(state.failure)],
  ]
  if (state._tag === "RemoveFailed") return [["Installation", "Removal failed"], ["Reason", state.failure.message]]
  return [
    ["Installation", state._tag === "NotInstalled" ? "Not installed" : state._tag === "Removing" ? "Removing" : "Installed"],
    ...(state._tag === "UpdateAvailable" ? [["Update", "Available"]] as const : []),
  ]
}

const renderModelDetail = (model: LocalModel): string => ensureTrailingNewline([
  formatLocalModelDisplayName(model),
  renderFields([
    ["Model ID", model.modelId],
    ...installationFields(model),
    ...(localModelIsInstalled(model) ? [["Runtime", residencyLabel(model)]] as const : []),
    ...(modelMemoryBytes(model) === undefined ? [] : [["Memory", modelMemory(model)]] as const),
    ...(modelContext(model) === "—" ? [] : [["Context", modelContext(model)]] as const),
  ]),
].join("\n"))

export const showModelsStatus = (modelInput?: string) => runCommand({
  effect: withClient((client) => Effect.gen(function* () {
    const catalog = yield* readCatalog(client)
    if (catalog._tag === "Initializing") return { _tag: "Initializing" as const }
    const models = localModels(catalog)
    if (modelInput === undefined) return { _tag: "List" as const, models }
    const modelId = yield* decodeModelId(modelInput)
    return { _tag: "Detail" as const, model: yield* findModel(models, modelId) }
  })),
  render: (result) => result._tag === "Initializing"
    ? "Local models are initializing.\n"
    : result._tag === "List" ? renderModelsStatus(result.models) : renderModelDetail(result.model),
})

export const loadInstance = (modelInput: string) => modelMutation(
  modelInput,
  decodeModelId,
  (client, modelId) => client.models.load({ modelId }),
  (modelId) => `Loaded ${modelId}.\n`,
)

export const stopInstance = () => runCommand({
  effect: withClient((client) => client.models.stop({})),
  render: () => "Stopped the active local model.\n",
})

const residentAllocation = (model: LocalModel) => {
  const residency = residencyState(model)
  if (residency?._tag === "Ready") return Option.some(residency.allocation)
  if (residency?._tag === "Stopping" && residency.allocation._tag === "Resident") {
    return Option.some(residency.allocation.allocation)
  }
  return Option.none()
}

const activePlan = (model: LocalModel) => {
  const residency = residencyState(model)
  if (residency?._tag === "Loading") return residency.plannedAllocation
  if (residency?._tag === "Stopping" && residency.allocation._tag === "Planned") {
    return residency.allocation.allocation
  }
  return Option.none()
}

const renderHardware = (hardware: LocalInferenceHardware, models: readonly LocalModel[]): string => {
  const presentation = describeLocalHardware(hardware)
  const current = models.find((model) => {
    const state = residencyState(model)?._tag
    return state === "Requested" || state === "Loading" || state === "Ready" || state === "Stopping"
  })
  const allocation = current === undefined ? Option.none() : residentAllocation(current)
  const plan = current === undefined ? Option.none() : activePlan(current)
  const memory = deriveHardwareMemoryView(hardware, allocation)
  const sections = [
    presentation.system.name,
    `  ${presentation.system.details.join("\n  ")}`,
    ...presentation.accelerators.flatMap((accelerator) => ["", accelerator.name, `  ${accelerator.details}`]),
    "",
    "Memory",
    ...memory.domains.flatMap((domain) => [
      `  ${domain.label}`,
      ...(domain.usedBytes === null || domain.freeBytes === null ? [`  ${domain.notice ?? "Current memory usage is unavailable."}`] : [
        `  ${formatMemorySize(domain.usedBytes)} / ${formatMemorySize(domain.totalBytes)} used`,
        ...(Option.isSome(allocation) ? [
          `  Weights       ${formatMemorySize(domain.fixedBytes ?? 0)}`,
          `  KV cache      ${formatMemorySize(domain.kvCacheBytes ?? 0)}`,
          `  System & apps ${formatMemorySize(domain.systemAndAppsBytes ?? 0)}`,
        ] : []),
        `  Free          ${formatMemorySize(domain.freeBytes)}`,
      ]),
    ]),
    "",
    "Current model",
    ...(current === undefined ? ["  None"] : [
      `  ${formatLocalModelDisplayName(current)} - ${residencyLabel(current)}`,
      ...Option.match(allocation, {
        onSome: (value) => [`  Context ${formatContext(value.contextWindowTokens)} - Parallelism ${value.parallelSequences}`],
        onNone: () => Option.match(plan, {
          onNone: () => [],
          onSome: (value) => [`  Context ${formatContext(value.contextWindowTokens)} - Parallelism ${value.parallelSequences}`],
        }),
      }),
    ]),
  ]
  return ensureTrailingNewline(sections.join("\n"))
}

export const showHardware = () => runCommand({
  effect: withClient((client) => Effect.all({
    hardware: readHardware(client),
    catalog: readCatalog(client),
  })),
  render: ({ hardware, catalog }) => renderHardware(hardware, localModels(catalog)),
})
