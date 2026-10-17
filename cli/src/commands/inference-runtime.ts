import {
  catalogSupportLabel,
  deriveHardwareMemoryView,
  formatLocalInferenceBackend,
  formatLocalModelDisplayName,
  formatMemorySize,
  formatStorageSize,
  LOCAL_MODEL_RANKING_SCALE_LABELS,
  LOCAL_MODEL_RANKING_SCALE_VALUES,
  localModelIsInstalled,
  localModelRadarAxes,
  rankedLocalModelOptions,
  targetPhysicalMemoryBytes,
} from "@magnitudedev/client-common"
import {
  LocalModelPreparationSchema,
  localModelDeprecation,
  type CatalogLocalModel,
  type LocalInferenceHardware,
  type LocalModel,
  type ModelCatalogState,
} from "@magnitudedev/sdk"
import { Effect, Option, Schema } from "effect"
import {
  catalogProducts,
  readCatalog,
  readHardware,
  withClient,
} from "./inference-client"
import { describeLocalHardware, formatContext } from "./inference-format"
import { CatalogModelJsonSchema, LocalInferenceHardwareJsonSchema, LocalModelJsonSchema } from "./json-schemas"
import { decodeCatalogId, ModelCommandError } from "./model-references"
import {
  ensureTrailingNewline,
  jsonDocument,
  renderFields,
  renderTable,
  commandEffect,
  asJson,
  type JsonOption,
} from "./output"
import {
  optional,
  plural,
  findModel,
  isCatalogModel,
  residencyState,
  modelContext,
  modelMemory,
  supportFields,
  accelerationLabel,
  capabilityLabels,
  residencyLabel,
  requireLocalModels,
} from "./model-format"
export * from "./hugging-face-runtime"

/** Runs a command, whether it is an Effect or has already started. */
export const execute = (command: Effect.Effect<void> | Promise<void>): Promise<void> =>
  Effect.isEffect(command) ? Effect.runPromise(command) : command

export * from "./custom-models-runtime"
export * from "./models-status-runtime"

const catalogModels = (catalog: ModelCatalogState): readonly CatalogLocalModel[] =>
  catalogProducts(catalog).flatMap((model) => model._tag === "Catalog" ? [model] : [])

export const renderCatalogStatus = (catalog: ModelCatalogState): string => {
  if (catalog._tag === "Initializing") return ensureTrailingNewline([
    "Model catalog preparation",
    "Assessment: In progress",
  ].join("\n"))

  const { assessment } = catalog.localModelPreparation
  const progress = `${assessment.settledModels} of ${plural(assessment.totalModels, "model")} assessed`
  const status = assessment.complete
    ? `Complete - ${progress}`
    : assessment.totalModels === 0
      ? "Not complete - no assessment targets reported"
      : `In progress - ${progress}`
  return ensureTrailingNewline([
    "Model catalog preparation",
    `Assessment: ${status}`,
  ].join("\n"))
}

export const CatalogStatusJsonSchema = Schema.Struct({ preparation: optional(LocalModelPreparationSchema) })

export const catalogStatusJson = jsonDocument(CatalogStatusJsonSchema, (catalog: ModelCatalogState) => ({
  preparation: catalog._tag === "Initializing" ? Option.none() : Option.some(catalog.localModelPreparation),
}))

export const showCatalogStatus = (options: JsonOption = {}) => commandEffect({
  effect: withClient(readCatalog),
  render: renderCatalogStatus,
  json: asJson(options, catalogStatusJson),
})

const assessmentSummary = (models: readonly CatalogLocalModel[]): { assessing: number; failed: number } => ({
  assessing: models.filter((model) => model.servingState._tag === "Assessing").length,
  failed: models.filter((model) => model.servingState._tag === "Failed").length,
})

const assessmentNotice = (
  models: readonly CatalogLocalModel[],
  subject: "list" | "recommendations",
): string[] => {
  const { assessing, failed } = assessmentSummary(models)
  return [
    ...(assessing > 0 ? [`Assessing ${plural(assessing, "additional catalog model")}; ${
      subject === "list" ? "this list may grow" : "recommendations may change"
    }.`] : []),
    ...(failed > 0 ? [`Assessment failed for ${plural(failed, "catalog model")}.`] : []),
  ]
}

const catalogFailureMessages = (catalog: ModelCatalogState): string[] => catalog._tag === "Initializing"
  ? []
  : [...new Set(catalog.failures.map(({ message }) => message.trim()).filter(Boolean))]

const catalogWarnings = (catalog: ModelCatalogState): string[] =>
  catalogFailureMessages(catalog).map((message) => `Catalog warning: ${message}`)

const fittingCatalogModels = (models: readonly CatalogLocalModel[]): CatalogLocalModel[] => models
  .filter(model => Option.isNone(localModelDeprecation(model))
    && model.servingState._tag === "Assessed"
    && model.servingState.assessment._tag === "Fits")
  .sort((left, right) => left.presentation.displayName.localeCompare(right.presentation.displayName)
    || String(left.presentation.variantLabel).localeCompare(String(right.presentation.variantLabel))
    || left.modelId.localeCompare(right.modelId))

const radarDetail = (model: CatalogLocalModel, label: string): string =>
  Option.getOrUndefined(localModelRadarAxes(model))?.find((axis) => axis.label === label)?.detail ?? "—"

const speedLabel = (model: CatalogLocalModel): string => radarDetail(model, "SPEED")

const labeledModelName = (model: CatalogLocalModel): string => Option.match(
  catalogSupportLabel(model.catalogData.support), {
    onNone: () => formatLocalModelDisplayName(model),
    onSome: (label) => `${formatLocalModelDisplayName(model)} [${label}]`,
  })

export const renderCatalog = (catalog: ModelCatalogState): string => {
  if (catalog._tag === "Initializing") return "The local model catalog is initializing.\n"
  const models = catalogModels(catalog)
  const compatible = fittingCatalogModels(models)
  const notice = [...assessmentNotice(models, "list"), ...catalogWarnings(catalog)]
  if (compatible.length === 0) {
    const lead = notice.length > 0
      ? "No compatible catalog models are available yet."
      : "No catalog models are compatible with this computer."
    return ensureTrailingNewline([lead, ...notice].join("\n"))
  }
  const table = renderTable(compatible, [
    { heading: "MODEL", value: labeledModelName },
    { heading: "MEMORY", value: modelMemory },
    { heading: "SPEED", value: speedLabel },
    { heading: "CONTEXT", value: modelContext },
    { heading: "ACCELERATION", value: accelerationLabel },
    { heading: "MODEL ID", value: ({ modelId }) => modelId },
  ])
  return ensureTrailingNewline([
    `Local model catalog - ${plural(compatible.length, "compatible model")}`,
    "",
    table.trimEnd(),
    ...(notice.length > 0 ? ["", ...notice] : []),
  ].join("\n"))
}

export const CatalogListJsonSchema = Schema.Struct({
  initializing: Schema.Boolean,
  models: Schema.Array(CatalogModelJsonSchema),
  assessing: Schema.Int,
  failed: Schema.Int,
  warnings: Schema.Array(Schema.String),
})

export const catalogListJson = jsonDocument(CatalogListJsonSchema, (catalog: ModelCatalogState) => {
  const models = catalogModels(catalog)
  return {
    initializing: catalog._tag === "Initializing",
    models: fittingCatalogModels(models),
    ...assessmentSummary(models),
    warnings: catalogFailureMessages(catalog),
  }
})

export const showModelCatalog = (options: JsonOption = {}) => commandEffect({
  effect: withClient(readCatalog),
  render: renderCatalog,
  json: asJson(options, catalogListJson),
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

export const parseLimit = (input: string, maximum = Number.MAX_SAFE_INTEGER) => {
  const limit = Number(input)
  return Number.isSafeInteger(limit) && limit > 0 && limit <= maximum
    ? Effect.succeed(limit)
    : Effect.fail(new ModelCommandError({
        message: maximum === Number.MAX_SAFE_INTEGER
          ? "Limit must be a positive integer"
          : `Limit must be an integer from 1 to ${maximum}`,
      }))
}

const renderRecommendation = (model: CatalogLocalModel, index: number): string => {
  const capabilities = capabilityLabels(model)
  return [
    `${index + 1}. ${labeledModelName(model)}`,
    `   ID: ${model.modelId}`,
    `   ${speedLabel(model)} - ${modelMemory(model)} memory - ${modelContext(model)} context`,
    `   Intelligence ${model.catalogData.intelligence}% - Fidelity ${radarDetail(model, "FIDELITY")} - Acceleration ${accelerationLabel(model)}`,
    ...(capabilities.length > 0 ? [`   ${capabilities.join(", ")}`] : []),
  ].join("\n")
}

export const HUGGING_FACE_HINT =
  "Looking for a specific version, finetune, or uncensored model? magnitude catalog search \"<model name>\""

export interface RecommendationsView {
  readonly catalog: ModelCatalogState
  readonly models: readonly CatalogLocalModel[]
  readonly hardware: LocalInferenceHardware
  readonly preference: { readonly label: string }
  readonly ranked: readonly CatalogLocalModel[]
}

export const RecommendationsJsonSchema = Schema.Struct({
  initializing: Schema.Boolean,
  preference: Schema.String,
  models: Schema.Array(CatalogModelJsonSchema),
})

export const recommendationsJson = jsonDocument(RecommendationsJsonSchema, (view: RecommendationsView) => ({
  initializing: view.catalog._tag === "Initializing",
  preference: view.preference.label,
  models: view.ranked,
}))

export const showRecommendations = (preferenceInput: string, limitInput: string, options: JsonOption = {}) => commandEffect({
  effect: Effect.all([parsePreference(preferenceInput), parseLimit(limitInput)]).pipe(
    Effect.flatMap(([preference, limit]) => withClient((client) => Effect.gen(function* () {
      const [catalog, hardware] = yield* Effect.all([readCatalog(client), readHardware(client)])
      const models = catalogModels(catalog)
      const ranked = rankedLocalModelOptions(models.map((model) => ({
        id: String(model.modelId),
        kind: localModelIsInstalled(model) ? "stored" as const : "downloadable" as const,
        model,
      })), {
        fastToSmart: preference.value,
        memoryBudgetBytes: targetPhysicalMemoryBytes(hardware),
      }, limit).flatMap(({ model }) => model._tag === "Catalog" ? [model] : [])
      return { catalog, models, hardware, preference, ranked }
    }))),
  ),
  render: renderRecommendations,
  json: asJson(options, recommendationsJson),
})

export function renderRecommendations({
  catalog,
  models,
  hardware,
  preference,
  ranked,
}: RecommendationsView): string {
  if (catalog._tag === "Initializing") return "The local model catalog is initializing.\n"
  const system = describeLocalHardware(hardware).system.name
  const notice = [...assessmentNotice(models, "recommendations"), ...catalogWarnings(catalog)]
  if (ranked.length === 0) return ensureTrailingNewline([
    `No compatible recommendations are available for ${preference.label}.`,
    ...notice,
    "",
    HUGGING_FACE_HINT,
  ].join("\n"))
  return ensureTrailingNewline([
    `Local model recommendations - ${preference.label}`,
    `${system} - ${formatMemorySize(hardware.totalSystemMemoryBytes)}`,
    "",
    ranked.map(renderRecommendation).join("\n\n"),
    ...(notice.length > 0 ? ["", ...notice] : []),
    "",
    "Learn more: magnitude docs recommendations",
    HUGGING_FACE_HINT,
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

export interface CatalogDetailView {
  readonly model: CatalogLocalModel
  readonly models: readonly LocalModel[]
}

export const otherVersionsCommand = (model: CatalogLocalModel): string =>
  `magnitude catalog search ${JSON.stringify(model.presentation.displayName)}`

export const renderCatalogDetail = ({ model, models }: CatalogDetailView): string => {
  const serving = model.servingState
  const header = [
    formatLocalModelDisplayName(model),
    ...(model.presentation.description.trim().length > 0 ? [model.presentation.description.trim()] : []),
    renderFields([
      ["Model ID", model.modelId],
      ...supportFields(model, models),
      ["Released", formatReleaseDate(model.catalogData.releaseDate)],
      ["Architecture", formatParameters(model)],
      ["Context", modelContext(model)],
      ...(serving._tag === "Assessed" ? [["Capabilities", capabilityLabels(model).join(", ") || "Text"]] as const : []),
    ]),
  ]
  const performance = (() => {
    if (serving._tag === "Assessing") return ["Assessment", renderFields([["Status", "Assessing memory and speed…"]])]
    if (serving._tag === "Failed") return ["Assessment", renderFields([["Status", "Assessment failed."], ["Reason", serving.failure.message]])]
    if (serving.assessment._tag === "DoesNotFit") return ["Performance on this machine", renderFields([
      ["Status", "Does not fit"],
      ["Memory required", formatMemorySize(serving.assessment.totalRequiredBytes)],
      ["Additional memory", formatMemorySize(serving.assessment.deficitBytes)],
    ])]
    return ["Performance on this machine", renderFields([
      ["Speed", speedLabel(model)],
      ["Fit", radarDetail(model, "FIT")],
      ["Intelligence", `${model.catalogData.intelligence}%`],
      ["Fidelity", radarDetail(model, "FIDELITY")],
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
    "",
    `Other versions: ${otherVersionsCommand(model)}`,
  ].join("\n"))
}

export const CatalogDetailJsonSchema = Schema.Struct({ model: CatalogModelJsonSchema })

export const catalogDetailJson = jsonDocument(CatalogDetailJsonSchema, ({ model }: CatalogDetailView) => ({ model }))

export const showCatalogModel = (modelInput: string, options: JsonOption = {}) => commandEffect({
  effect: decodeCatalogId(modelInput).pipe(Effect.flatMap(modelId => withClient((client) => Effect.gen(function* () {
    const models = yield* requireLocalModels(yield* readCatalog(client))
    return { model: yield* findModel(models, modelId, "catalog model", isCatalogModel), models }
  })))),
  render: renderCatalogDetail,
  json: asJson(options, catalogDetailJson),
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

export interface HardwareView {
  readonly hardware: LocalInferenceHardware
  readonly models: readonly LocalModel[]
}

const currentModel = (models: readonly LocalModel[]) => models.find((model) => {
  const state = residencyState(model)?._tag
  return state === "Requested" || state === "Loading" || state === "Ready" || state === "Stopping"
})

export const renderHardware = ({ hardware, models }: HardwareView): string => {
  const presentation = describeLocalHardware(hardware)
  const current = currentModel(models)
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
        onSome: (value) => [`  Context ${formatContext(value.contextWindowTokens)}`],
        onNone: () => Option.match(plan, {
          onNone: () => [],
          onSome: (value) => [`  Context ${formatContext(value.contextWindowTokens)} - Device ${formatLocalInferenceBackend(value.device.backend)}`],
        }),
      }),
    ]),
  ]
  return ensureTrailingNewline(sections.join("\n"))
}

export const HardwareJsonSchema = Schema.Struct({
  hardware: LocalInferenceHardwareJsonSchema,
  currentModel: optional(LocalModelJsonSchema),
})

export const hardwareJson = jsonDocument(HardwareJsonSchema, ({ hardware, models }: HardwareView) => ({
  hardware,
  currentModel: Option.fromNullable(currentModel(models)),
}))

export const showHardware = (options: JsonOption = {}) => commandEffect({
  effect: withClient((client) => Effect.all({
    hardware: readHardware(client),
    catalog: readCatalog(client),
  })).pipe(Effect.map(({ hardware, catalog }) => ({ hardware, models: catalogProducts(catalog) }))),
  render: renderHardware,
  json: asJson(options, hardwareJson),
})
