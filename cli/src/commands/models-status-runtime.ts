import {
  catalogComponentName,
  catalogComponentOffers,
  catalogDrafterWithheldNote,
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
  preselectedDrafterCandidate,
  removalKind,
  unsupportedLocalModelSourceTag,
  unsupportedReasonMessages,
  type RemovalKind,
} from "@magnitudedev/client-common"
import {
  CatalogFormModelIdSchema,
  DrafterCandidateSchema,
  ModelIdSchema,
  UnsupportedLocalModelSchema,
  localModelDeprecation,
  installedAcquisition,
  type CatalogComponent,
  type CatalogMissingComponent,
  type CustomLocalModel,
  type DrafterCandidate,
  type DrafterReference,
  type LocalModel,
  type ModelId,
  type UnsupportedLocalModel,
} from "@magnitudedev/sdk"
import { Effect, Option, Schema } from "effect"
import { renderDrafterChoices } from "./drafter-output"
import {
  addressableModels,
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
  commandEffect,
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
) => commandEffect({
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

/** The parts Magnitude stores for a catalog model whose weights are found elsewhere, and where those weights are. */
export interface StoredCatalogParts {
  readonly components: readonly CatalogComponent[]
  readonly weightsPath: string
}

/** A catalog model's stored parts when its weights are only in another app's location: what Remove deletes. */
export const storedCatalogParts = (model: LocalModel): Option.Option<StoredCatalogParts> => {
  if (model._tag !== "Catalog" || model.storedComponents.length === 0) return Option.none()
  const installation = installedAcquisition(model.acquisitionState)?.installation
  return installation?._tag === "Resolved"
    ? Option.some({ components: model.storedComponents, weightsPath: installation.primaryPath })
    : Option.none()
}

/** What removing a model did, which follows who owns its files (`removalKind`): a found model, a catalog model's found copy included, is hidden. */
export const renderRemoval = (
  modelId: ModelId,
  kind: Option.Option<RemovalKind>,
  parts: Option.Option<StoredCatalogParts> = Option.none(),
): string => {
  if (Option.isSome(parts)) {
    const names = parts.value.components.map((component) => catalogComponentName(component).toLowerCase()).join(" and ")
    return `Removed the ${names} Magnitude downloaded for ${modelId}.\n`
      + `It still runs from ${parts.value.weightsPath}, which Magnitude never deletes.\n`
      + `To stop listing it: magnitude models hide ${modelId}\n`
  }
  switch (Option.getOrUndefined(kind)) {
    case "hide": return `Hid ${modelId} from Magnitude. Its files are untouched.\nUnhide it: magnitude models unhide ${modelId}\n`
    case "forget": return `Forgot ${modelId}. Magnitude never deletes your file; it stays where it is.\n`
    default: return Schema.is(CatalogFormModelIdSchema)(modelId)
      ? `Removed ${modelId} from this computer.\nThe model remains available in the catalog.\n`
      : `Removed ${modelId}.\nMagnitude deleted only its own copy; files other apps manage are untouched.\n`
  }
}

export const removeModel = (modelInput: string, options: JsonOption = {}) => commandEffect({
  effect: decodeModelId(modelInput).pipe(Effect.flatMap((modelId) => withClient((client) => Effect.gen(function* () {
    const model = Option.fromNullable(addressableModels(yield* readCatalog(client)).find((candidate) => candidate.modelId === modelId))
    const kind = Option.map(model, removalKind)
    const parts = Option.flatMap(model, storedCatalogParts)
    yield* client.models.removeLocalModel({ modelId })
    return { modelId, kind, parts }
  })))),
  render: ({ modelId, kind, parts }) => renderRemoval(modelId, kind, parts),
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

/** A newer revision of a custom model's attached catalog drafter exists. */
export const drafterUpdateAvailable = (model: LocalModel): boolean =>
  model._tag === "Custom" && Option.exists(model.drafter, ({ updateAvailable, availability }) => updateAvailable && availability._tag !== "Downloading")

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

const DRAFTER_TURNED_OFF = "Turned off"

/** The catalog model a custom model is a variant of, by name when the catalog lists it. */
const variantOfLabel = (model: LocalModel, models: readonly LocalModel[]): Option.Option<string> =>
  model._tag === "Catalog" ? Option.none() : Option.map(model.presentation.variantOf, (variantOf) =>
    Option.match(catalogVariantOfName(models, model.presentation.variantOf), {
      onNone: () => variantOf,
      onSome: (name) => `${name} (${variantOf})`,
    }))

/** The attached drafter, "Turned off", or the recommended drafter's offer when it isn't the one attached. */
const drafterSummary = (model: CustomLocalModel): Option.Option<string> => Option.match(model.drafter, {
  onSome: (drafter) => Option.some([
    drafter.name,
    ...Option.toArray(drafterTransferStatus(model)),
    ...Option.toArray(Option.map(customModelDrafterOffer(model), ({ label }) => label)),
  ].join(" · ")),
  onNone: () => model.drafterChoice._tag === "None"
    ? Option.some(DRAFTER_TURNED_OFF)
    : Option.map(customModelDrafterOffer(model), ({ label }) => label),
})

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
  const catalogModels = [...models, ...hidden]
  const visible = modelsForStatus(models)
  const listed = all ? [...visible, ...hidden] : visible
  const isHidden = (model: LocalModel) => hidden.some(({ modelId }) => modelId === model.modelId)
  const footer = all ? [] : [
    ...catalogComponentHints(visible),
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
      { heading: "VARIANT OF", value: (model) => Option.getOrElse(variantOfLabel(model, catalogModels), () => "—") },
      { heading: "DRAFTER", value: (model) => model._tag === "Catalog" ? "—" : Option.getOrElse(drafterSummary(model), () => "—") },
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

const componentCommand = (model: LocalModel, { component }: CatalogMissingComponent): string =>
  `magnitude catalog pull ${model.modelId} --only ${component.toLowerCase()}`

const componentSize = ({ downloadBytes }: CatalogMissingComponent): string =>
  downloadBytes > 0 ? ` (${formatStorageSize(downloadBytes)})` : ""

/**
 * One line per catalog component the model lacks, each with the command that downloads only it, and the
 * drafter it runs without because it fits only without it.
 */
const catalogComponentFields = (model: LocalModel): readonly (readonly [string, string])[] => [
  ...catalogComponentOffers(model).map((offer) => [
    catalogComponentName(offer.component),
    `Not downloaded${componentSize(offer)} - ${componentCommand(model, offer)}`,
  ] as const),
  ...Option.toArray(Option.map(catalogDrafterWithheldNote(model), (note) => ["Drafter", note] as const)),
]

/** The list's footer line for each catalog component a model lacks. */
const catalogComponentHints = (models: readonly LocalModel[]): string[] => models.flatMap((model) =>
  catalogComponentOffers(model).map((offer) =>
    `${catalogComponentName(offer.component)} for ${formatLocalModelDisplayName(model)} isn't downloaded${componentSize(offer)}: ${componentCommand(model, offer)}`))

const customSourceDetail = (model: CustomLocalModel): string => {
  switch (model.origin._tag) {
    case "Added": return `Hugging Face · ${model.origin.repository}`
    case "Imported": return `GGUF file · ${model.origin.path}`
    case "Found": return model.origin.source._tag === "Folder"
      ? `${model.presentation.sourceLabel} · ${model.origin.source.path}`
      : model.presentation.sourceLabel
  }
}

const attachedDrafterLabel = (model: CustomLocalModel): string => Option.isNone(model.drafter) && model.drafterChoice._tag === "Auto"
  ? Option.match(customModelDrafterOffer(model), { onNone: () => "None", onSome: ({ label }) => `None · ${label}` })
  : Option.getOrElse(drafterSummary(model), () => "None")

/** The drafter this model uses: none when turned off, the named one, or under auto the attached or preselected one. */
const currentDrafter = (model: CustomLocalModel, candidates: readonly DrafterCandidate[]): Option.Option<DrafterReference> => {
  switch (model.drafterChoice._tag) {
    case "None": return Option.none()
    case "Reference": return Option.some(model.drafterChoice.reference)
    case "Auto": return Option.orElse(
      Option.map(model.drafter, ({ reference }) => reference),
      () => Option.map(preselectedDrafterCandidate(candidates), ({ reference }) => reference),
    )
  }
}

export type DrafterChoices =
  | { readonly _tag: "NotApplicable" }
  | { readonly _tag: "Listed"; readonly candidates: readonly DrafterCandidate[] }
  | { readonly _tag: "Unavailable"; readonly message: string }

export interface ModelDetailView {
  readonly model: LocalModel
  readonly models: readonly LocalModel[]
  readonly drafters: DrafterChoices
  readonly hidden: boolean
}

const customFields = (model: CustomLocalModel, models: readonly LocalModel[]): readonly (readonly [string, string])[] => [
  ["Source", customSourceDetail(model)],
  ...Option.match(variantOfLabel(model, models), { onNone: () => [], onSome: (label) => [["Variant of", label]] as const }),
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
      renderDrafterChoices(view.drafters.candidates, currentDrafter(model, view.drafters.candidates)),
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
      ...(view.hidden ? [["Hidden", `Yes - unhide it: magnitude models unhide ${model.modelId}`]] as const : []),
      ...(model._tag === "Custom" ? customFields(model, models) : [["Source", "Catalog"] as const]),
      ...Option.match(model.presentation.license, { onNone: () => [], onSome: (license) => [["License", license] as const] }),
      ...supportFields(model, models),
      ...installationFields(model),
      ...catalogComponentFields(model),
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
  hidden: Schema.Boolean,
  drafterCandidates: optional(Schema.Array(DrafterCandidateSchema)),
  drafterCandidatesUnavailable: optional(Schema.String),
})

export const modelDetailJson = jsonDocument(ModelDetailJsonSchema, ({ model, drafters, hidden }: ModelDetailView) => ({
  model,
  hidden,
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
  if (Option.isNone(modelId)) return {
    _tag: "List" as const,
    view: { models: catalogProducts(catalog), unsupported: catalog.unsupportedLocalModels, hidden: catalog.hiddenLocalModels, all },
  }
  const models = addressableModels(catalog)
  const model = yield* findModel(models, modelId.value)
  return {
    _tag: "Detail" as const,
    view: {
      model,
      models,
      drafters: yield* listModelDrafters(client, model),
      hidden: catalog.hiddenLocalModels.some((hidden) => hidden.modelId === model.modelId),
    },
  }
})

export const renderModelsStatusResult = (result: ModelsStatusResult): string => result._tag === "Initializing"
  ? "Local models are initializing.\n"
  : result._tag === "List" ? renderModelsStatus(result.view) : renderModelDetail(result.view)

export const showModelsStatus = (modelInput?: string, options: JsonOption & { readonly all?: boolean } = {}) => commandEffect({
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

/** Stops the named model, by any ID the other commands accept, or else the active one. */
export const stopInstance = (modelInput?: string, options: JsonOption = {}) => modelInput === undefined
  ? commandEffect({
      effect: withClient((client) => client.models.stop({})),
      render: () => "Stopped the active local model.\n",
      json: asJson(options, jsonDocument(Schema.Struct({}), () => ({}))),
    })
  : modelMutation(
      modelInput.trim(),
      (client, modelId) => client.models.stopModel({ modelId }),
      (modelId) => `Stopped ${modelId}.\n`,
      options,
    )
