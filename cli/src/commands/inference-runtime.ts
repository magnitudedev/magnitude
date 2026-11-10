import {
  deriveHardwareMemoryView,
  deriveHardwareRoles,
  formatLocalInferenceBackend,
  formatLocalModelDisplayName,
  formatMemoryCapacity,
  formatMemorySize,
  formatStorageSize,
  localModelAllocation,
  localModelExpertFacts,
  modelServingFacts,
} from "@magnitudedev/client-common"
import {
  CountryCodeSchema,
  LocalModelFamilySchema,
  ModelIdSchema,
  localModelIntelligence,
  type CatalogLocalModel,
  type CountryCode,
  type LocalModelFamily,
  type RecommendedModel,
  type Recommendations,
  type LocalInferenceHardware,
  type LocalModel,
  type ModelCatalogState,
} from "@magnitudedev/sdk"
import { Effect, Option, Schema } from "effect"
import {
  catalogProducts,
  knowsModel,
  readCatalog,
  readHardware,
  readRecommendations,
  withClient,
} from "./inference-client"
import { describeLocalHardware, formatContext } from "./inference-format"
import { inspectionJson, readInspection, renderInspection, type InspectionView } from "./hugging-face-runtime"
import { CatalogModelJsonSchema, LocalInferenceHardwareJsonSchema, LocalModelJsonSchema } from "./json-schemas"
import { huggingFaceFileOfModelId, isLocalPathReference, ModelCommandError, parseHuggingFaceReference } from "./model-references"
import {
  modelDetailResultJson,
  readModelDetail,
  renderModelDetailResult,
  type ModelDetailResult,
} from "./models-status-runtime"
import {
  ensureTrailingNewline,
  jsonDocument,
  renderTable,
  commandEffect,
  asJson,
  type JsonDocument,
  type JsonOption,
} from "./output"
import {
  optional,
  plural,
  residencyState,
  accelerationLabel,
  assessmentContext,
  assessmentPlacement,
  assessmentSpeed,
  assessmentSpeedKnown,
  capabilityLabels,
  modelAssessment,
  modelQuant,
  renderQuantLadder,
  residencyLabel,
} from "./model-format"
export * from "./hugging-face-runtime"

/** Runs a command, whether it is an Effect or has already started. */
export const execute = (command: Effect.Effect<void> | Promise<void>): Promise<void> =>
  Effect.isEffect(command) ? Effect.runPromise(command) : command

export * from "./custom-models-runtime"
export * from "./models-status-runtime"

const catalogModels = (catalog: ModelCatalogState): readonly CatalogLocalModel[] =>
  catalogProducts(catalog).flatMap((model) => model._tag === "Catalog" ? [model] : [])

const assessingCount = (models: readonly CatalogLocalModel[]): { assessing: number; failed: number } => {
  const offered = models.filter((model) => model.catalogData.support._tag === "Supported")
  return {
    assessing: offered.filter((model) => model.servingState._tag === "Assessing").length,
    failed: offered.filter((model) => model.servingState._tag === "Failed").length,
  }
}

const assessmentNotice = (models: readonly CatalogLocalModel[]): string[] => {
  const { assessing, failed } = assessingCount(models)
  return [
    ...(assessing > 0 ? [`Assessing ${plural(assessing, "additional catalog model")}; recommendations may change.`] : []),
    ...(failed > 0 ? [`Assessment failed for ${plural(failed, "catalog model")}.`] : []),
  ]
}

const parseCountry = (input: string | undefined) => input === undefined
  ? Effect.succeed(Option.none<CountryCode>())
  : Schema.decodeUnknown(CountryCodeSchema)(input.toUpperCase()).pipe(
      Effect.map(Option.some),
      Effect.mapError(() => new ModelCommandError({
        message: "Country must be a two-letter ISO 3166-1 code, such as US",
      })),
    )

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

/** A recommended model with the catalog model of its chosen rung and of every rung of its ladder. */
export interface RecommendedCatalogModel {
  readonly recommendation: RecommendedModel
  readonly chosen: CatalogLocalModel
  readonly rungs: readonly CatalogLocalModel[]
}

const intelligenceLabel = (model: CatalogLocalModel): string =>
  Option.match(localModelIntelligence(model), { onNone: () => "not rated", onSome: (score) => `${score}%` })

const indent = (text: string, prefix: string): string =>
  text.trimEnd().split("\n").map((line) => `${prefix}${line}`).join("\n")

const renderRecommendation = (hardware: LocalInferenceHardware) => (
  { recommendation, chosen: model, rungs }: RecommendedCatalogModel,
  index: number,
): string => {
  const unifiedMemory = deriveHardwareRoles(hardware).unifiedMemory
  const assessment = modelAssessment(model)
  const capabilities = capabilityLabels(model)
  const expertFields = localModelExpertFacts(model, Option.some(hardware))
  const { developer } = recommendation
  return [
    `${index + 1}. ${formatLocalModelDisplayName(model)} - ${developer.name} (${developer.country}) - Intelligence ${intelligenceLabel(model)}`,
    `   ${modelQuant(model)} - ${formatStorageSize(model.storageBytes)} - ${assessmentSpeed(assessment)} - ${assessmentContext(assessment)} context - ${assessmentPlacement(assessment, unifiedMemory)}`,
    `   Model ID: ${model.modelId}${recommendation.meetsTarget || !assessmentSpeedKnown(assessment) ? "" : " - below 20 tok/s on this computer"}`,
    ...(expertFields.length > 0 ? [`   ${expertFields.map(([label, value]) => `${label} ${value}`).join(" - ")}`] : []),
    `   Acceleration ${accelerationLabel(model)}${capabilities.length > 0 ? ` - ${capabilities.join(", ")}` : ""}`,
    indent(renderQuantLadder(rungs, Option.some(model.modelId), unifiedMemory), "   "),
  ].join("\n")
}

export const HUGGING_FACE_HINT =
  "Looking for a specific version, finetune, or uncensored model? magnitude models search \"<model name>\""

export interface RecommendationsView {
  readonly catalog: ModelCatalogState
  readonly models: readonly CatalogLocalModel[]
  readonly hardware: LocalInferenceHardware
  readonly developerCountry: Option.Option<CountryCode>
  readonly recommended: readonly RecommendedCatalogModel[]
}

export const RecommendationsJsonSchema = Schema.Struct({
  initializing: Schema.Boolean,
  developerCountry: optional(Schema.String),
  models: Schema.Array(Schema.Struct({
    base: Schema.String,
    rungs: Schema.Array(Schema.String),
    meetsTarget: Schema.Boolean,
    meetsOnboardingFloor: Schema.Boolean,
    model: CatalogModelJsonSchema,
  })),
})

export const recommendationsJson = jsonDocument(RecommendationsJsonSchema, (view: RecommendationsView) => ({
  initializing: view.catalog._tag === "Initializing",
  developerCountry: view.developerCountry,
  models: view.recommended.map(({ recommendation, chosen }) => ({
    base: recommendation.base,
    rungs: recommendation.rungs,
    meetsTarget: recommendation.meetsTarget,
    meetsOnboardingFloor: recommendation.meetsOnboardingFloor,
    model: chosen,
  })),
}))

/** The recommendations with a chosen rung, in order, each with its chosen catalog model and its ladder's. */
export const recommendedCatalogModels = (
  recommendations: Recommendations,
  models: readonly CatalogLocalModel[],
  limit: number,
): readonly RecommendedCatalogModel[] => {
  const byId = new Map(models.map((model) => [model.modelId as string, model] as const))
  return recommendations.models.flatMap((recommendation) => Option.match(
    Option.flatMap(recommendation.chosen, (id) => Option.fromNullable(byId.get(id))), {
      onNone: () => [],
      onSome: (chosen) => [{
        recommendation,
        chosen,
        rungs: recommendation.rungs.flatMap((id) => Option.toArray(Option.fromNullable(byId.get(id)))),
      }],
    })).slice(0, limit)
}

export const showRecommendations = (
  countryInput: string | undefined,
  limitInput: string,
  options: JsonOption = {},
) => commandEffect({
  effect: Effect.all([parseCountry(countryInput), parseLimit(limitInput)]).pipe(
    Effect.flatMap(([developerCountry, limit]) => withClient((client) => Effect.gen(function* () {
      const [catalog, hardware, recommendations] = yield* Effect.all([
        readCatalog(client),
        readHardware(client),
        readRecommendations(client, developerCountry),
      ])
      const models = catalogModels(catalog)
      return {
        catalog,
        models,
        hardware,
        developerCountry,
        recommended: recommendedCatalogModels(recommendations, models, limit),
      }
    }))),
  ),
  render: renderRecommendations,
  json: asJson(options, recommendationsJson),
})

const catalogWarnings = (catalog: ModelCatalogState): string[] => catalog._tag === "Initializing"
  ? []
  : [...new Set(catalog.failures.map(({ message }) => message.trim()).filter(Boolean))].map((message) => `Catalog warning: ${message}`)

export function renderRecommendations({
  catalog,
  models,
  hardware,
  developerCountry,
  recommended,
}: RecommendationsView): string {
  if (catalog._tag === "Initializing") return "The local model catalog is initializing.\n"
  const system = describeLocalHardware(hardware).system.name
  const notice = [...assessmentNotice(models), ...catalogWarnings(catalog)]
  const scope = Option.match(developerCountry, { onNone: () => "", onSome: (country) => ` from ${country} developers` })
  if (recommended.length === 0) return ensureTrailingNewline([
    `No models${scope} run on this computer.`,
    ...notice,
    "",
    HUGGING_FACE_HINT,
  ].join("\n"))
  return ensureTrailingNewline([
    `Recommended for this computer${scope}`,
    `${system} - ${formatMemoryCapacity(hardware.totalSystemMemoryBytes)}`,
    "",
    recommended.map(renderRecommendation(hardware)).join("\n\n"),
    ...(notice.length > 0 ? ["", ...notice] : []),
    "",
    "* the quant chosen for this computer. Speeds are estimates in a long agent conversation.",
    "Learn more: magnitude docs recommendations",
    HUGGING_FACE_HINT,
  ].join("\n"))
}

const familyStatus = (status: LocalModelFamily["status"]): string => status === "ComingSoon" ? "Coming soon" : "Supported"

export const renderFamilies = (families: readonly LocalModelFamily[]): string => families.length === 0
  ? "Magnitude lists no model families.\n"
  : ensureTrailingNewline([
      "Model families Magnitude runs; their fine-tunes run too",
      "",
      renderTable(families, [
        { heading: "FAMILY", value: ({ name }) => name },
        { heading: "DEVELOPER", value: ({ developer }) => developer.name },
        { heading: "COUNTRY", value: ({ developer }) => developer.country },
        { heading: "ARCHITECTURES", value: ({ architectures }) => architectures.join(", ") },
        { heading: "STATUS", value: ({ status }) => familyStatus(status) },
      ]).trimEnd(),
    ].join("\n"))

export const FamiliesJsonSchema = Schema.Struct({ families: Schema.Array(LocalModelFamilySchema) })

export const familiesJson = jsonDocument(FamiliesJsonSchema, (families: readonly LocalModelFamily[]) => ({ families }))

export const showFamilies = (options: JsonOption = {}) => commandEffect({
  effect: withClient((client) => readRecommendations(client, Option.none())).pipe(Effect.map(({ families }) => families)),
  render: renderFamilies,
  json: asJson(options, familiesJson),
})

/** What `models show` names: a model by ID, or a Hugging Face repository to inspect. */
export type ShowResult =
  | { readonly _tag: "Model"; readonly result: ModelDetailResult }
  | { readonly _tag: "Repository"; readonly view: InspectionView }

export const showResultJson: JsonDocument<ShowResult> = {
  encode: (result) => result._tag === "Model"
    ? modelDetailResultJson.encode(result.result)
    : inspectionJson.encode(result.view),
}

export const renderShowResult = (result: ShowResult): string => result._tag === "Model"
  ? renderModelDetailResult(result.result)
  : renderInspection(result.view)

const SHOW_REFERENCE = "Expected a model ID, a Hugging Face link, or owner/repo[:QUANT]"

export const showModel = (input: string, options: JsonOption = {}) => commandEffect({
  effect: Effect.gen(function* () {
    const reference = input.trim()
    if (reference.length === 0 || isLocalPathReference(reference)) {
      return yield* new ModelCommandError({ message: `${SHOW_REFERENCE}: ${reference}` })
    }
    if (Schema.is(ModelIdSchema)(reference)) {
      const modelId = reference
      return yield* withClient((client) => Effect.gen(function* () {
        // A Hugging Face model's ID names its file, so one not on this computer is shown as that file in its repository.
        const hubFile = huggingFaceFileOfModelId(modelId)
        if (Option.isSome(hubFile) && !(yield* knowsModel(client, modelId))) {
          return { _tag: "Repository" as const, view: yield* readInspection(client, hubFile.value) }
        }
        return { _tag: "Model" as const, result: yield* readModelDetail(client, modelId) }
      }))
    }
    if (Option.isNone(parseHuggingFaceReference(reference))) {
      return yield* new ModelCommandError({ message: `${SHOW_REFERENCE}: ${reference}` })
    }
    return { _tag: "Repository" as const, view: yield* withClient((client) => readInspection(client, reference)) }
  }),
  render: renderShowResult,
  json: asJson(options, showResultJson),
})

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
  const allocation = current === undefined ? Option.none() : localModelAllocation(current)
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
        `  ${formatMemoryCapacity(domain.usedBytes)} / ${formatMemoryCapacity(domain.totalBytes)} used`,
        ...(Option.isSome(allocation) ? [
          `  Weights       ${formatMemoryCapacity(domain.weightsBytes ?? 0)}`,
          `  Context       ${formatMemoryCapacity(domain.kvCacheBytes ?? 0)}`,
          `  Overhead      ${formatMemoryCapacity(domain.overheadBytes ?? 0)}`,
          `  System & apps ${formatMemoryCapacity(domain.systemAndAppsBytes ?? 0)}`,
        ] : []),
        `  Free          ${formatMemoryCapacity(domain.freeBytes)}`,
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
      ...Option.match(allocation, { onNone: () => [], onSome: (value) => modelServingFacts(value, Option.some(hardware)) })
        .map(([label, value]) => `  ${label}: ${value}`),
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
