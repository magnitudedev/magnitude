import {
  catalogVariantOfName,
  customModelDrafterOffer,
  formatLocalModelDisplayName,
  formatModelLoadPercentage,
  formatModelOptimization,
  modelOptimizationFraction,
  formatStorageSize,
  localModelIsInstalled,
  localModelServingState,
  modelDownloadFailureMessage,
  unsupportedLocalModelSourceTag,
  unsupportedReasonMessages,
} from "@magnitudedev/client-common"
import {
  CatalogFormModelIdSchema,
  DrafterCandidateSchema,
  ModelIdSchema,
  UnsupportedLocalModelSchema,
  localModelDeprecation,
  type CustomLocalModel,
  type DrafterCandidate,
  type LocalModel,
  type ModelId,
  type UnsupportedLocalModel,
} from "@magnitudedev/sdk"
import { Effect, Option, Schema } from "effect"
import { renderDrafterChoices } from "./drafter-output"
import {
  catalogProducts,
  readCatalog,
  withClient,
  type CliModelsClient,
} from "./inference-client"
import { LocalModelJsonSchema } from "./json-schemas"
import { decodeModelId } from "./model-references"
import {
  ensureTrailingNewline,
  jsonDocument,
  renderFields,
  renderTable,
  runCommand,
  asJson,
  type JsonDocument,
  type JsonOption,
} from "./output"
import {
  optional,
  plural,
  findModel,
  modelContext,
  modelMemoryBytes,
  modelMemory,
  supportFields,
  accelerationLabel,
  UNLOADED,
  residencyLabel,
  percent,
} from "./model-format"

export const ModelIdJsonSchema = Schema.Struct({ modelId: ModelIdSchema })
const modelIdJson = jsonDocument(ModelIdJsonSchema, (modelId: ModelId) => ({ modelId }))

const modelMutation = (
  modelInput: string,
  execute: (client: CliModelsClient, modelId: ModelId) => Effect.Effect<unknown, unknown>,
  render: (modelId: ModelId) => string,
  options: JsonOption,
) => runCommand({
  effect: decodeModelId(modelInput).pipe(
    Effect.flatMap((modelId) => withClient(client => execute(client, modelId).pipe(Effect.as(modelId)))),
  ),
  render,
  json: asJson(options, modelIdJson),
})

export const cancelDownload = (modelInput: string, options: JsonOption = {}) => modelMutation(
  modelInput,
  (client, modelId) => client.models.cancelLocalModelSync({ modelId }),
  (modelId) => `Cancelled work for ${modelId}.\n`,
  options,
)

/** What removing a model did, which follows who owns its files. */
export const renderRemoval = (modelId: ModelId, origin: Option.Option<CustomLocalModel["origin"]>): string => {
  if (Schema.is(CatalogFormModelIdSchema)(modelId)) {
    return `Removed ${modelId} from this computer.\nThe model remains available in the catalog.\n`
  }
  switch (Option.getOrUndefined(origin)?._tag) {
    case "Imported": return `Forgot ${modelId}. Magnitude never deletes your file; it stays where it is.\n`
    case "Found": return `Hid ${modelId} from Magnitude. Its files are untouched.\nUnhide it: magnitude models unhide ${modelId}\n`
    default: return `Removed ${modelId}.\nMagnitude deleted only its own copy; files other apps manage are untouched.\n`
  }
}

export const removeModel = (modelInput: string, options: JsonOption = {}) => runCommand({
  effect: decodeModelId(modelInput).pipe(Effect.flatMap((modelId) => withClient((client) => Effect.gen(function* () {
    const origin = Option.flatMap(
      Option.fromNullable(catalogProducts(yield* readCatalog(client)).find((model) => model.modelId === modelId)),
      (model) => model._tag === "Custom" ? Option.some(model.origin) : Option.none(),
    )
    yield* client.models.removeLocalModel({ modelId })
    return { modelId, origin }
  })))),
  render: ({ modelId, origin }) => renderRemoval(modelId, origin),
  json: asJson(options, jsonDocument(ModelIdJsonSchema, ({ modelId }: { readonly modelId: ModelId }) => ({ modelId }))),
})

const unsupportedAssessmentReason = (model: LocalModel): Option.Option<string> => {
  const serving = Option.getOrUndefined(localModelServingState(model))
  return serving?._tag === "Assessed" && serving.assessment._tag === "Unsupported"
    ? Option.some(unsupportedReasonMessages(serving.assessment.reasons).join(". "))
    : Option.none()
}

const drafterTransferStatus = (model: LocalModel): Option.Option<string> => {
  if (model._tag !== "Custom") return Option.none()
  return Option.flatMap(model.drafter, ({ availability }) => {
    switch (availability._tag) {
      case "Present": return Option.none()
      case "Downloading": return Option.some(
        `Downloading drafter ${percent(availability.progress.completedBytes, availability.progress.totalBytes)}`)
      case "DownloadFailed": return Option.some(`Drafter download failed - ${modelDownloadFailureMessage(availability.failure)}`)
    }
  })
}

/** A newer revision of a custom model's attached catalog drafter exists and nothing is downloading it yet. */
export const drafterUpdateAvailable = (model: LocalModel): boolean =>
  model._tag === "Custom" && Option.exists(model.drafter, ({ updateAvailable, availability }) =>
    updateAvailable && availability._tag === "Present")

const readyStatus = (model: LocalModel): string => Option.getOrElse(
  Option.orElse(unsupportedAssessmentReason(model).pipe(Option.map((reason) => `Can't run - ${reason}`)),
    () => drafterTransferStatus(model)),
  () => residencyLabel(model))

const withUpdate = (model: LocalModel): string => {
  const status = readyStatus(model)
  return status === UNLOADED ? "Update available" : `${status} - Update available`
}

const modelStatus = (model: LocalModel): string => {
  const acquisition = model.acquisitionState
  const deprecation = Option.getOrUndefined(localModelDeprecation(model))
  if (deprecation !== undefined && acquisition._tag !== "Removing" && acquisition._tag !== "RemoveFailed") {
    return `Deprecated - switch to ${deprecation.replacement}`
  }
  switch (acquisition._tag) {
    case "Removing": return "Removing"
    case "Installing": return `Downloading ${percent(acquisition.progress.completedBytes, acquisition.progress.totalBytes)}`
    case "Updating": return `Updating ${percent(acquisition.progress.completedBytes, acquisition.progress.totalBytes)}`
    case "InstallFailed": return `Failed - ${modelDownloadFailureMessage(acquisition.failure)}`
    case "UpdateFailed": return `Failed - ${modelDownloadFailureMessage(acquisition.failure)}`
    case "Optimizing": return formatModelOptimization(acquisition.progress)
    case "RemoveFailed": return `Failed - ${acquisition.failure.message}`
    case "NotInstalled": return "Not installed"
    case "Unavailable": return `Unavailable - ${acquisition.failure.message}`
    case "UpdateAvailable": return withUpdate(model)
    case "Installed":
    case "External": return drafterUpdateAvailable(model) ? withUpdate(model) : readyStatus(model)
  }
}

const sourceLabel = (model: LocalModel): string => model._tag === "Catalog" ? "Catalog" : model.presentation.sourceLabel

/** Catalog models on this computer or with local work, and every custom model the service lists. */
const modelsForStatus = (models: readonly LocalModel[]): LocalModel[] => models
  .filter((model) => model._tag === "Custom" || model.acquisitionState._tag !== "NotInstalled")
  .sort((a, b) => formatLocalModelDisplayName(a).localeCompare(formatLocalModelDisplayName(b)) || a.modelId.localeCompare(b.modelId))

export interface ModelsStatusView {
  readonly models: readonly LocalModel[]
  readonly unsupported: readonly UnsupportedLocalModel[]
  readonly hidden: readonly LocalModel[]
  readonly all: boolean
}

const unsupportedReasonLine = (model: UnsupportedLocalModel): string =>
  unsupportedReasonMessages(model.reasons).join(". ")

export const renderModelsStatus = ({ models, unsupported, hidden, all }: ModelsStatusView): string => {
  const visible = modelsForStatus(models)
  const listed = all ? [...visible, ...hidden] : visible
  const isHidden = (model: LocalModel) => hidden.some(({ modelId }) => modelId === model.modelId)
  const footer = all ? [] : [
    ...(unsupported.length > 0
      ? [`${plural(unsupported.length, "other model")} on this computer can't run in Magnitude.`]
      : []),
    ...(hidden.length > 0 ? [`${plural(hidden.length, "hidden model")}.`] : []),
    ...(unsupported.length > 0 || hidden.length > 0 ? ["List them: magnitude models status --all"] : []),
  ]
  const unsupportedSection = all && unsupported.length > 0 ? [
    "",
    "Can't run in Magnitude",
    "",
    renderTable([...unsupported].sort((a, b) => a.displayName.localeCompare(b.displayName)), [
      { heading: "MODEL", value: ({ displayName }) => displayName },
      { heading: "SOURCE", value: unsupportedLocalModelSourceTag },
      { heading: "REASON", value: unsupportedReasonLine },
    ]).trimEnd(),
  ] : []
  if (listed.length === 0) return ensureTrailingNewline([
    "No local models are on this computer.",
    ...(footer.length > 0 ? ["", ...footer] : []),
    ...unsupportedSection,
  ].join("\n"))
  return ensureTrailingNewline([
    "Local models",
    "",
    renderTable(listed, [
      { heading: "MODEL", value: formatLocalModelDisplayName },
      { heading: "SOURCE", value: sourceLabel },
      { heading: "MEMORY", value: modelMemory },
      { heading: "CONTEXT", value: modelContext },
      { heading: "STATUS", value: (model) => isHidden(model) ? "Hidden" : modelStatus(model) },
      { heading: "MODEL ID", value: ({ modelId }) => modelId },
    ]).trimEnd(),
    ...(footer.length > 0 ? ["", ...footer] : []),
    ...unsupportedSection,
    ...(all && hidden.length > 0 ? ["", "Unhide a model: magnitude models unhide <model-id>"] : []),
  ].join("\n"))
}

export const ModelsStatusJsonSchema = Schema.Struct({
  initializing: Schema.Boolean,
  models: Schema.Array(LocalModelJsonSchema),
  unsupported: Schema.Array(UnsupportedLocalModelSchema),
  hidden: Schema.Array(LocalModelJsonSchema),
})

export const modelsStatusJson = jsonDocument(ModelsStatusJsonSchema, (view: ModelsStatusView) => ({
  initializing: false,
  models: modelsForStatus(view.models),
  unsupported: view.unsupported,
  hidden: view.hidden,
}))

const installationFields = (model: LocalModel): readonly (readonly [string, string])[] => {
  const state = model.acquisitionState
  if (state._tag === "Installing" || state._tag === "Updating") return [
    ["Installation", state._tag === "Installing" ? "Downloading" : "Updating"],
    ["Progress", `${percent(state.progress.completedBytes, state.progress.totalBytes)} - ${formatStorageSize(state.progress.completedBytes)} / ${formatStorageSize(state.progress.totalBytes)}`],
  ]
  if (state._tag === "Optimizing") return [
    ["Installation", "Optimizing"],
    ...Option.match(modelOptimizationFraction(state.progress), {
      onNone: () => [],
      onSome: (fraction) => [["Progress", formatModelLoadPercentage(fraction)]] as const,
    }),
  ]
  if (state._tag === "InstallFailed" || state._tag === "UpdateFailed") return [
    ["Installation", "Failed"],
    ["Reason", modelDownloadFailureMessage(state.failure)],
  ]
  if (state._tag === "RemoveFailed") return [["Installation", "Removal failed"], ["Reason", state.failure.message]]
  if (state._tag === "External") return [["Files", "On this computer, managed outside Magnitude"]]
  if (state._tag === "Unavailable") return [["Files", "Unavailable"], ["Reason", state.failure.message]]
  return [
    ["Installation", state._tag === "NotInstalled" ? "Not installed" : state._tag === "Removing" ? "Removing" : "Installed"],
    ...(state._tag === "UpdateAvailable" ? [["Update", "Available"]] as const : []),
  ]
}

const customSourceDetail = (model: CustomLocalModel): string => {
  switch (model.origin._tag) {
    case "Added": return `Hugging Face · ${model.origin.repository}`
    case "Imported": return `GGUF file · ${model.origin.path}`
    case "Found": return model.origin.source._tag === "Folder"
      ? `${model.presentation.sourceLabel} · ${model.origin.source.path}`
      : model.presentation.sourceLabel
  }
}

const attachedDrafterLabel = (model: CustomLocalModel): string => Option.match(model.drafter, {
  onNone: () => Option.match(customModelDrafterOffer(model), { onNone: () => "None", onSome: ({ label }) => `None · ${label}` }),
  onSome: (drafter) => [
    drafter.name,
    ...Option.toArray(drafterTransferStatus(model)),
  ].join(" · "),
})

export type DrafterChoices =
  | { readonly _tag: "NotApplicable" }
  | { readonly _tag: "Listed"; readonly candidates: readonly DrafterCandidate[] }
  | { readonly _tag: "Unavailable"; readonly message: string }

export interface ModelDetailView {
  readonly model: LocalModel
  readonly models: readonly LocalModel[]
  readonly drafters: DrafterChoices
}

const customFields = (model: CustomLocalModel, models: readonly LocalModel[]): readonly (readonly [string, string])[] => [
  ["Source", customSourceDetail(model)],
  ...Option.match(model.presentation.variantOf, {
    onNone: () => [],
    onSome: (variantOf) => [["Variant of", Option.match(catalogVariantOfName(models, model.presentation.variantOf), {
      onNone: () => variantOf,
      onSome: (name) => `${name} (${variantOf})`,
    })]] as const,
  }),
  ["Drafter", attachedDrafterLabel(model)],
  ...(drafterUpdateAvailable(model)
    ? [["Update", `A newer drafter is available - magnitude catalog pull ${model.modelId}`]] as const
    : []),
  ...Option.match(unsupportedAssessmentReason(model), { onNone: () => [], onSome: (reason) => [["Can't run", reason]] as const }),
]

const drafterSection = (view: ModelDetailView): string[] => {
  if (view.model._tag !== "Custom") return []
  const model = view.model
  const change = `Change: magnitude models drafter ${model.modelId} <auto | none | drafter>`
  switch (view.drafters._tag) {
    case "NotApplicable": return []
    case "Unavailable": return ["", `Drafter choices are unavailable: ${view.drafters.message}`, change]
    case "Listed": return [
      "",
      renderDrafterChoices(view.drafters.candidates, Option.map(model.drafter, ({ reference }) => reference)),
      "",
      change,
    ]
  }
}

export const renderModelDetail = (view: ModelDetailView): string => {
  const { model, models } = view
  return ensureTrailingNewline([
    formatLocalModelDisplayName(model),
    renderFields([
      ["Model ID", model.modelId],
      ...(model._tag === "Custom" ? customFields(model, models) : [["Source", "Catalog"] as const]),
      ...supportFields(model, models),
      ...installationFields(model),
      ...(localModelIsInstalled(model) ? [["Runtime", residencyLabel(model)]] as const : []),
      ...(modelMemoryBytes(model) === undefined ? [] : [["Memory", modelMemory(model)]] as const),
      ...(modelContext(model) === "—" ? [] : [["Context", modelContext(model)]] as const),
      ...(model._tag === "Catalog" ? [["Acceleration", accelerationLabel(model)]] as const : []),
    ]),
    ...drafterSection(view),
  ].join("\n"))
}

export const ModelDetailJsonSchema = Schema.Struct({
  model: LocalModelJsonSchema,
  drafterCandidates: optional(Schema.Array(DrafterCandidateSchema)),
  drafterCandidatesUnavailable: optional(Schema.String),
})

export const modelDetailJson = jsonDocument(ModelDetailJsonSchema, ({ model, drafters }: ModelDetailView) => ({
  model,
  drafterCandidates: drafters._tag === "Listed" ? Option.some(drafters.candidates) : Option.none(),
  drafterCandidatesUnavailable: drafters._tag === "Unavailable" ? Option.some(drafters.message) : Option.none(),
}))

const listModelDrafters = (client: CliModelsClient, model: LocalModel): Effect.Effect<DrafterChoices> =>
  model._tag !== "Custom"
    ? Effect.succeed({ _tag: "NotApplicable" })
    : client.models.listDrafterCandidates({ target: { _tag: "Model", modelId: model.modelId }, other: Option.none() }).pipe(
        Effect.map(({ candidates }): DrafterChoices => ({ _tag: "Listed", candidates })),
        Effect.catchAll((error) => Effect.succeed<DrafterChoices>({
          _tag: "Unavailable",
          message: typeof error === "object" && error !== null && "message" in error && typeof error.message === "string"
            ? error.message
            : String(error),
        })),
      )

export type ModelsStatusResult =
  | { readonly _tag: "Initializing" }
  | { readonly _tag: "List"; readonly view: ModelsStatusView }
  | { readonly _tag: "Detail"; readonly view: ModelDetailView }

export const modelsStatusResultJson: JsonDocument<ModelsStatusResult> = {
  encode: (result) => {
    switch (result._tag) {
      case "Initializing": return Schema.encode(ModelsStatusJsonSchema)({ initializing: true, models: [], unsupported: [], hidden: [] })
      case "List": return modelsStatusJson.encode(result.view)
      case "Detail": return modelDetailJson.encode(result.view)
    }
  },
}

export const readModelsStatus = (
  client: CliModelsClient,
  modelId: Option.Option<ModelId>,
  all: boolean,
): Effect.Effect<ModelsStatusResult, unknown> => Effect.gen(function* () {
  const catalog = yield* readCatalog(client)
  if (catalog._tag === "Initializing") return { _tag: "Initializing" as const }
  const models = catalogProducts(catalog)
  if (Option.isNone(modelId)) return {
    _tag: "List" as const,
    view: { models, unsupported: catalog.unsupportedLocalModels, hidden: catalog.hiddenLocalModels, all },
  }
  const model = yield* findModel(models, modelId.value)
  return { _tag: "Detail" as const, view: { model, models, drafters: yield* listModelDrafters(client, model) } }
})

export const renderModelsStatusResult = (result: ModelsStatusResult): string => result._tag === "Initializing"
  ? "Local models are initializing.\n"
  : result._tag === "List" ? renderModelsStatus(result.view) : renderModelDetail(result.view)

export const showModelsStatus = (modelInput?: string, options: JsonOption & { readonly all?: boolean } = {}) => runCommand({
  effect: (modelInput === undefined ? Effect.succeed(Option.none<ModelId>()) : decodeModelId(modelInput).pipe(Effect.map(Option.some))).pipe(
    Effect.flatMap((modelId) => withClient((client) => readModelsStatus(client, modelId, options.all === true))),
  ),
  render: renderModelsStatusResult,
  json: asJson(options, modelsStatusResultJson),
})

export const loadInstance = (modelInput: string, options: JsonOption = {}) => modelMutation(
  modelInput,
  (client, modelId) => client.models.load({ modelId }),
  (modelId) => `Load requested for ${modelId}.\nCheck progress: magnitude models status ${modelId}\n`,
  options,
)

export const stopInstance = (options: JsonOption = {}) => runCommand({
  effect: withClient((client) => client.models.stop({})),
  render: () => "Stopped the active local model.\n",
  json: asJson(options, jsonDocument(Schema.Struct({}), () => ({}))),
})
