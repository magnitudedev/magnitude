import {
  catalogComponentName,
  catalogComponentOffers,
  catalogDrafterWithheldNote,
  catalogVariantOfName,
  customModelDrafterOffer,
  PENDING_DRAFTER_LABEL,
  formatLocalModelDisplayName,
  formatModelLoadPercentage,
  formatModelOptimization,
  MODEL_OPTIMIZATION_PENDING,
  MODEL_OPTIMIZATION_PENDING_DETAIL,
  modelOptimizationFraction,
  formatStorageSize,
  localModelAllocation,
  localModelExpertFacts,
  localModelIsInstalled,
  localModelSizeBytes,
  promptGroupPlacesFact,
  localModelServingState,
  preselectedDrafterCandidate,
  removalKind,
  acquiringCatalogParts,
  deriveHardwareRoles,
  formatModelParameters,
  unsupportedLocalModelSourceTag,
  unsupportedReasonMessages,
  type RemovalKind,
} from "@magnitudedev/client-common"
import {
  CatalogFormModelIdSchema,
  DrafterCandidateSchema,
  ModelIdSchema,
  UnsupportedLocalModelSchema,
  LocalModelPreparationSchema,
  localModelDeprecation,
  localModelIntelligence,
  installedAcquisition,
  type CatalogComponent,
  type CatalogLocalModel,
  type CatalogMissingComponent,
  type CustomLocalModel,
  type DrafterCandidate,
  type DrafterReference,
  type LocalInferenceHardware,
  type LocalModel,
  type LocalModelPreparation,
  type ModelCatalogState,
  type ModelId,
  type UnsupportedLocalModel,
} from "@magnitudedev/sdk"
import { Effect, Option, Schema } from "effect"
import { renderDrafterChoices } from "./drafter-output"
import { downloadFailureText, drafterRepository, modelRepository } from "./hugging-face-access"
import {
  addressableModels,
  catalogProducts,
  readCatalog,
  readHardware,
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
  doesNotFitLabel,
  memoryBreakdownFields,
  modelMemoryBytes,
  modelMemory,
  supportFields,
  accelerationLabel,
  UNLOADED,
  residencyLabel,
  percent,
  requireLocalModels,
  modelAssessment,
  modelQuant,
  modelSource,
  assessmentSpeed,
  fitDetailSections,
  renderQuantLadder,
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

/** Magnitude's own parts of a catalog model whose weights are found elsewhere: those it stores, whether it is still fetching more, and where the weights are. */
export interface StoredCatalogParts {
  readonly components: readonly CatalogComponent[]
  readonly fetching: boolean
  readonly weightsPath: string
}

/** A catalog model's own parts when its weights are only in another app's location: what Remove deletes or cancels. */
export const storedCatalogParts = (model: LocalModel): Option.Option<StoredCatalogParts> => {
  if (model._tag !== "Catalog" || removalKind(model) !== "parts") return Option.none()
  const installation = installedAcquisition(model.acquisitionState)?.installation
  return installation?._tag === "Resolved"
    ? Option.some({ components: model.storedComponents, fetching: acquiringCatalogParts(model), weightsPath: installation.primaryPath })
    : Option.none()
}

/** What removing a model did, which follows who owns its files (`removalKind`): a found model, a catalog model's found copy included, is hidden. */
export const renderRemoval = (
  modelId: ModelId,
  kind: Option.Option<RemovalKind>,
  parts: Option.Option<StoredCatalogParts> = Option.none(),
  kept: readonly CatalogComponent[] = [],
): string => {
  if (Option.isSome(parts)) {
    const names = (components: readonly CatalogComponent[]) =>
      components.map((component) => catalogComponentName(component).toLowerCase()).join(" and ")
    const removed = names(parts.value.components.filter((component) => !kept.includes(component)))
    return (parts.value.fetching ? `Cancelled the download Magnitude had in progress for ${modelId}.\n` : "")
      + (removed.length > 0 ? `Removed the ${removed} Magnitude downloaded for ${modelId}.\n` : "")
      + (kept.length > 0 ? `Kept the ${names(kept)} Magnitude downloaded for ${modelId}: another model on this computer uses ${kept.length > 1 ? "them" : "it"}.\n` : "")
      + `It still runs from ${parts.value.weightsPath}, which Magnitude never deletes.\n`
      + `To stop listing it: magnitude models hide ${modelId}\n`
  }
  switch (Option.getOrUndefined(kind)) {
    case "hide": return `Hid ${modelId} from Magnitude. Its files are untouched.\nUnhide it: magnitude models unhide ${modelId}\n`
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
    const { keptComponents } = yield* client.models.removeLocalModel({ modelId })
    return { modelId, kind, parts, keptComponents }
  })))),
  render: ({ modelId, kind, parts, keptComponents }) => renderRemoval(modelId, kind, parts, keptComponents),
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
  return Option.flatMap(model.drafter, ({ availability, reference }) => {
    switch (availability._tag) {
      case "Present": return Option.none()
      case "Downloading": return Option.some(
        `Downloading drafter ${percent(availability.progress.completedBytes, availability.progress.totalBytes)}`)
      case "DownloadFailed": return Option.some(`Drafter download failed - ${downloadFailureText(availability.failure, drafterRepository(reference))}`)
    }
  })
}

/** A newer revision of a custom model's attached catalog drafter exists. */
export const drafterUpdateAvailable = (model: LocalModel): boolean =>
  model._tag === "Custom" && Option.exists(model.drafter, ({ updateAvailable, availability }) => updateAvailable && availability._tag !== "Downloading")

const readyStatus = (model: LocalModel): string => Option.getOrElse(
  Option.orElse(unsupportedAssessmentReason(model).pipe(Option.map((reason) => `Can't run - ${reason}`)),
    () => Option.orElse(doesNotFitLabel(model), () => drafterTransferStatus(model))),
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
    case "InstallFailed": return `Failed - ${downloadFailureText(acquisition.failure, modelRepository(model.modelId))}`
    case "UpdateFailed": return `Failed - ${downloadFailureText(acquisition.failure, modelRepository(model.modelId))}`
    case "Optimizing": return formatModelOptimization(acquisition.progress)
    case "OptimizationPending": {
      const status = readyStatus(model)
      return status === UNLOADED ? MODEL_OPTIMIZATION_PENDING : `${status} - ${MODEL_OPTIMIZATION_PENDING}`
    }
    case "RemoveFailed": return `Failed - ${acquisition.failure.message}`
    case "NotInstalled": return "Not installed"
    case "Unavailable": return `Unavailable - ${acquisition.failure.message}`
    case "UpdateAvailable": return withUpdate(model)
    case "Installed":
    case "External": return drafterUpdateAvailable(model) ? withUpdate(model) : readyStatus(model)
  }
}

const DRAFTER_TURNED_OFF = "Turned off"

/** The catalog model a custom model is a variant of, by name when the catalog lists it. */
const variantOfLabel = (model: LocalModel, models: readonly LocalModel[]): Option.Option<string> =>
  model._tag === "Catalog" ? Option.none() : Option.map(model.presentation.variantOf, (variantOf) =>
    Option.match(catalogVariantOfName(models, model.presentation.variantOf), {
      onNone: () => variantOf,
      onSome: (name) => `${name} (${variantOf})`,
    }))

/**
 * The attached drafter, "Turned off", that the automatic one is still being found, or the recommended drafter's offer
 * when it isn't the one attached.
 */
const drafterSummary = (model: CustomLocalModel): Option.Option<string> => Option.match(model.drafter, {
  onSome: (drafter) => Option.some([
    drafter.name,
    ...Option.toArray(drafterTransferStatus(model)),
    ...Option.toArray(Option.map(customModelDrafterOffer(model), ({ label }) => label)),
  ].join(" · ")),
  onNone: () => model.drafterChoice._tag === "None"
    ? Option.some(DRAFTER_TURNED_OFF)
    : model.drafterPending
      ? Option.some(PENDING_DRAFTER_LABEL)
      : Option.map(customModelDrafterOffer(model), ({ label }) => label),
})

/** Catalog models on this computer or with local work, and every custom model the service lists. */
const modelsForStatus = (models: readonly LocalModel[]): LocalModel[] => models
  .filter((model) => model._tag === "Custom" || model.acquisitionState._tag !== "NotInstalled")
  .sort((a, b) => formatLocalModelDisplayName(a).localeCompare(formatLocalModelDisplayName(b)) || a.modelId.localeCompare(b.modelId))

export interface ModelsListView {
  readonly models: readonly LocalModel[]
  readonly unsupported: readonly UnsupportedLocalModel[]
  readonly hidden: readonly LocalModel[]
  readonly all: boolean
  readonly preparation: LocalModelPreparation
  readonly warnings: readonly string[]
}

const unsupportedReasonLine = (model: UnsupportedLocalModel): string =>
  unsupportedReasonMessages(model.reasons).join(". ")

/** How far the service is in assessing the models it offers on this computer. */
export const renderPreparation = ({ assessment }: LocalModelPreparation): string => {
  const progress = `${assessment.settledModels} of ${plural(assessment.totalModels, "model")} assessed`
  const status = assessment.complete
    ? `Complete - ${progress}`
    : assessment.totalModels === 0
      ? "Not complete - no assessment targets reported"
      : `In progress - ${progress}`
  return `Assessment: ${status}`
}

export const renderModelsList = ({ models, unsupported, hidden, all, preparation, warnings }: ModelsListView): string => {
  const visible = modelsForStatus(models)
  const listed = all ? [...visible, ...hidden] : visible
  const isHidden = (model: LocalModel) => hidden.some(({ modelId }) => modelId === model.modelId)
  const footer = [
    ...(all ? [] : [
      ...catalogComponentHints(visible),
      ...(unsupported.length > 0
        ? [`${plural(unsupported.length, "other model")} on this computer can't run in Magnitude.`]
        : []),
      ...(hidden.length > 0 ? [`${plural(hidden.length, "hidden model")}.`] : []),
      ...(unsupported.length > 0 || hidden.length > 0 ? ["List them: magnitude models list --all"] : []),
    ]),
    ...warnings.map((message) => `Catalog warning: ${message}`),
    renderPreparation(preparation),
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
    "Find one: magnitude models recommend",
    "",
    ...footer,
    ...unsupportedSection,
  ].join("\n"))
  return ensureTrailingNewline([
    "Models on this computer",
    "",
    renderTable(listed, [
      { heading: "MODEL", value: ({ presentation }) => presentation.displayName },
      { heading: "QUANT", value: modelQuant },
      { heading: "SIZE", value: (model) => formatStorageSize(localModelSizeBytes(model)) },
      { heading: "SPEED", value: (model) => assessmentSpeed(modelAssessment(model)) },
      { heading: "CONTEXT", value: modelContext },
      { heading: "SOURCE", value: modelSource },
      { heading: "STATUS", value: (model) => isHidden(model) ? "Hidden" : modelStatus(model) },
      { heading: "MODEL ID", value: ({ modelId }) => modelId },
    ]).trimEnd(),
    "",
    ...footer,
    ...unsupportedSection,
    ...(all && hidden.length > 0 ? ["", "Unhide a model: magnitude models unhide <model-id>"] : []),
  ].join("\n"))
}

export const ModelsListJsonSchema = Schema.Struct({
  initializing: Schema.Boolean,
  preparation: optional(LocalModelPreparationSchema),
  models: Schema.Array(LocalModelJsonSchema),
  unsupported: Schema.Array(UnsupportedLocalModelSchema),
  hidden: Schema.Array(LocalModelJsonSchema),
  warnings: Schema.Array(Schema.String),
})

export const modelsListJson = jsonDocument(ModelsListJsonSchema, (view: ModelsListView) => ({
  initializing: false,
  preparation: Option.some(view.preparation),
  models: modelsForStatus(view.models),
  unsupported: view.unsupported,
  hidden: view.hidden,
  warnings: view.warnings,
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
  if (state._tag === "OptimizationPending") return [
    ["Installation", "Installed"],
    ["Optimization", `Pending - ${MODEL_OPTIMIZATION_PENDING_DETAIL}`],
  ]
  if (state._tag === "InstallFailed" || state._tag === "UpdateFailed") return [
    ["Installation", "Failed"],
    ["Reason", downloadFailureText(state.failure, modelRepository(model.modelId))],
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
  `magnitude models pull ${model.modelId} --only ${component.toLowerCase()}`

const componentSize = ({ downloadBytes }: CatalogMissingComponent): string =>
  downloadBytes > 0 ? ` (${formatStorageSize(downloadBytes)})` : ""

/**
 * One line per catalog component the model lacks, each with the command that downloads only it, and the
 * drafter it runs without because it doesn't fit with it.
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
    case "Found": return model.origin.source._tag === "Folder"
      ? `${model.presentation.sourceLabel} · ${model.origin.source.path}`
      : model.presentation.sourceLabel
  }
}

const attachedDrafterLabel = (model: CustomLocalModel): string => Option.isNone(model.drafter) && model.drafterChoice._tag === "Auto" && !model.drafterPending
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
  readonly hardware: LocalInferenceHardware
  readonly drafters: DrafterChoices
  readonly hidden: boolean
}

const customFields = (model: CustomLocalModel, models: readonly LocalModel[]): readonly (readonly [string, string])[] => [
  ["Source", customSourceDetail(model)],
  ...Option.match(variantOfLabel(model, models), { onNone: () => [], onSome: (label) => [["Variant of", label]] as const }),
  ["Drafter", attachedDrafterLabel(model)],
  ...(drafterUpdateAvailable(model)
    ? [["Update", `A newer drafter is available - magnitude models pull ${model.modelId}`]] as const
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

const formatReleaseDate = (value: string): string => {
  const date = new Date(`${value.length === 7 ? `${value}-01` : value}T00:00:00Z`)
  return Number.isNaN(date.valueOf())
    ? value
    : new Intl.DateTimeFormat("en", { month: "long", year: "numeric", timeZone: "UTC" }).format(date)
}

const catalogFields = (model: CatalogLocalModel): readonly (readonly [string, string])[] => [
  ["Source", modelSource(model)],
  ["Developer", `${model.catalogData.developer.name} (${model.catalogData.developer.country})`],
  ["Released", formatReleaseDate(model.catalogData.releaseDate)],
  ["Parameters", formatModelParameters(model.catalogData.parameterization)],
  ["Intelligence", Option.match(localModelIntelligence(model), { onNone: () => "Not rated", onSome: (score) => `${score}%` })],
  ["Hugging Face", `https://huggingface.co/${model.catalogData.quant.repository}`],
]

/** The quants of the catalog model a catalog model is one rung of, smallest first; none for a custom model. */
const catalogLadder = (model: LocalModel, models: readonly LocalModel[]): readonly LocalModel[] => {
  if (model._tag !== "Catalog") return []
  const base = (modelId: string) => modelId.slice(0, modelId.indexOf(":"))
  return models
    .filter((candidate) => candidate._tag === "Catalog" && base(candidate.modelId) === base(model.modelId))
    .sort((left, right) => left.storageBytes - right.storageBytes)
}

export const otherVersionsCommand = (model: LocalModel): string =>
  `magnitude models search ${JSON.stringify(model.presentation.displayName)}`

export const renderModelDetail = (view: ModelDetailView): string => {
  const { model, models } = view
  const unifiedMemory = deriveHardwareRoles(view.hardware).unifiedMemory
  const ladder = catalogLadder(model, models)
  const description = model._tag === "Catalog" ? model.presentation.description.trim() : ""
  return ensureTrailingNewline([
    formatLocalModelDisplayName(model),
    ...(description.length > 0 ? [description] : []),
    renderFields([
      ["Model ID", model.modelId],
      ["Quant", modelQuant(model)],
      ["Size", formatStorageSize(model.storageBytes)],
      ...(view.hidden ? [["Hidden", `Yes - unhide it: magnitude models unhide ${model.modelId}`]] as const : []),
      ...(model._tag === "Custom" ? customFields(model, models) : catalogFields(model)),
      ...Option.match(model.presentation.license, { onNone: () => [], onSome: (license) => [["License", license] as const] }),
      ...supportFields(model, models),
      ...installationFields(model),
      ...catalogComponentFields(model),
      ...(localModelIsInstalled(model) ? [["Runtime", residencyLabel(model)]] as const : []),
      ...Option.match(localModelAllocation(model), { onNone: () => [], onSome: (allocation) => [promptGroupPlacesFact(allocation)] }),
      ...Option.match(doesNotFitLabel(model), { onNone: () => [], onSome: (label) => [["Fit", label]] as const }),
      ...(modelMemoryBytes(model) === undefined ? [] : [["Memory", modelMemory(model)]] as const),
      ["Speed", assessmentSpeed(modelAssessment(model))],
      ...localModelExpertFacts(model, Option.some(view.hardware)),
      ...memoryBreakdownFields(model),
      ...(modelContext(model) === "—" ? [] : [["Context", modelContext(model)]] as const),
      ...(model._tag === "Catalog" ? [["Acceleration", accelerationLabel(model)]] as const : []),
    ]),
    ...fitDetailSections(modelAssessment(model), unifiedMemory),
    ...(ladder.length > 1 ? ["", "Quants", renderQuantLadder(ladder, Option.some(model.modelId), unifiedMemory).trimEnd()] : []),
    ...drafterSection(view),
    ...(model._tag === "Catalog" ? ["", `Other versions: ${otherVersionsCommand(model)}`] : []),
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

/** A model on this computer that Magnitude can't run, addressed by its ID: where it is and why it can't run. */
export const renderUnsupportedModelDetail = (model: UnsupportedLocalModel): string => ensureTrailingNewline([
  model.displayName,
  renderFields([
    ...Option.match(model.modelId, { onNone: () => [], onSome: (modelId) => [["Model ID", modelId] as const] }),
    ["Source", unsupportedLocalModelSourceTag(model)],
    ["File", model.path],
    ["Can't run", unsupportedReasonLine(model)],
  ]),
].join("\n"))

export const UnsupportedModelDetailJsonSchema = Schema.Struct({ unsupportedModel: UnsupportedLocalModelSchema })

const unsupportedModelDetailJson = jsonDocument(
  UnsupportedModelDetailJsonSchema,
  (unsupportedModel: UnsupportedLocalModel) => ({ unsupportedModel }),
)

export type ModelDetailResult =
  | { readonly _tag: "Detail"; readonly view: ModelDetailView }
  | { readonly _tag: "Unsupported"; readonly model: UnsupportedLocalModel }

export const modelDetailResultJson: JsonDocument<ModelDetailResult> = {
  encode: (result) => result._tag === "Detail"
    ? modelDetailJson.encode(result.view)
    : unsupportedModelDetailJson.encode(result.model),
}

export const renderModelDetailResult = (result: ModelDetailResult): string => result._tag === "Detail"
  ? renderModelDetail(result.view)
  : renderUnsupportedModelDetail(result.model)

/** A model on this computer or in the catalog by ID, with its drafter choices; or a file Magnitude found but can't run. */
export const readModelDetail = (client: CliModelsClient, modelId: ModelId): Effect.Effect<ModelDetailResult, unknown> => Effect.gen(function* () {
  const catalog = yield* readCatalog(client)
  const models = yield* requireLocalModels(catalog)
  const unsupported = catalog._tag === "Initializing"
    ? undefined
    : catalog.unsupportedLocalModels.find((candidate) => Option.contains(candidate.modelId, modelId))
  if (unsupported !== undefined) return { _tag: "Unsupported" as const, model: unsupported }
  const model = yield* findModel(models, modelId)
  const { hardware, drafters } = yield* Effect.all(
    { hardware: readHardware(client), drafters: listModelDrafters(client, model) },
    { concurrency: "unbounded" },
  )
  return {
    _tag: "Detail" as const,
    view: { model, models, hardware, drafters, hidden: isHiddenModel(catalog, model) },
  }
})

const isHiddenModel = (catalog: ModelCatalogState, model: LocalModel): boolean =>
  catalog._tag !== "Initializing" && catalog.hiddenLocalModels.some((hidden) => hidden.modelId === model.modelId)

export type ModelsListResult =
  | { readonly _tag: "Initializing" }
  | { readonly _tag: "List"; readonly view: ModelsListView }

export const modelsListResultJson: JsonDocument<ModelsListResult> = {
  encode: (result) => result._tag === "List"
    ? modelsListJson.encode(result.view)
    : Schema.encode(ModelsListJsonSchema)({ initializing: true, preparation: Option.none(), models: [], unsupported: [], hidden: [], warnings: [] }),
}

export const readModelsList = (client: CliModelsClient, all: boolean): Effect.Effect<ModelsListResult, unknown> =>
  readCatalog(client).pipe(Effect.map((catalog): ModelsListResult => catalog._tag === "Initializing"
    ? { _tag: "Initializing" }
    : {
        _tag: "List",
        view: {
          models: catalogProducts(catalog),
          unsupported: catalog.unsupportedLocalModels,
          hidden: catalog.hiddenLocalModels,
          all,
          preparation: catalog.localModelPreparation,
          warnings: [...new Set(catalog.failures.map(({ message }) => message.trim()).filter(Boolean))],
        },
      }))

export const renderModelsListResult = (result: ModelsListResult): string => result._tag === "List"
  ? renderModelsList(result.view)
  : "Local models are initializing.\n"

export const listModels = (options: JsonOption & { readonly all?: boolean } = {}) => commandEffect({
  effect: withClient((client) => readModelsList(client, options.all === true)),
  render: renderModelsListResult,
  json: asJson(options, modelsListResultJson),
})

export const loadInstance = (modelInput: string, options: JsonOption = {}) => modelMutation(
  modelInput,
  (client, modelId) => client.models.load({ modelId }),
  (modelId) => `Load requested for ${modelId}.\nCheck progress: magnitude models show ${modelId}\n`,
  options,
)

/** Stops the named model, by any ID the other commands accept, or else the active one. */
export const stopInstance = (modelInput?: string, options: JsonOption = {}) => modelInput === undefined
  ? commandEffect({
      effect: withClient((client) => client.models.stop({})),
      render: () => "Stopped the active local model.\n",
      json: asJson(options, jsonDocument(Schema.Struct({}), () => ({}))),
    })
  : commandEffect({
      effect: decodeModelId(modelInput.trim()).pipe(Effect.flatMap((modelId) => withClient((client) =>
        client.models.stopModel({ modelId }).pipe(Effect.map(({ stopped }) => ({ modelId, stopped })))))),
      render: ({ modelId, stopped }) => stopped ? `Stopped ${modelId}.\n` : `${modelId} wasn't running.\n`,
      json: asJson(options, jsonDocument(
        Schema.Struct({ modelId: ModelIdSchema, stopped: Schema.Boolean }),
        (result: { readonly modelId: ModelId; readonly stopped: boolean }) => result,
      )),
    })
