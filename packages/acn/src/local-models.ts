import { Array, Context, Effect, Layer, Option, Schema, Stream } from "effect"
import { dataEquivalence } from "@magnitudedev/utils/data-equivalence"
import {
  LocalModelAssessmentSchema,
  LocalModelDoesNotFitAssessmentSchema,
  CatalogLocalModelSchema,
  CatalogFormModelIdSchema,
  CustomLocalModelSchema,
  CustomFormModelIdSchema,
  CustomModelOriginSchema,
  DrafterVerdictSchema,
  ModelReleaseDateSchema,
  ModelVariantLabelSchema,
  ModelCapabilitiesSchema,
  UnsupportedLocalModelSchema,
  customModelSourceLabel,
  parseModelId,
  type AttachedDrafter,
  type CatalogLocalModel,
  type CustomLocalModel,
  type CustomLocalModelServingState,
  type CustomModelAcquisitionState,
  type CustomModelOrigin,
  type DrafterChoice,
  type DrafterReference,
  type DrafterVerdict,
  type LocalModelAcquisitionState,
  type CatalogLocalModelServingState,
  type LocalModelAssessment,
  type MachineMeasurement,
  type LocalModel,
  type LocalModelsState,
  type LocalModelPreparation,
  type ModelCapabilities,
  type ModelId,
  type ModelFailure,
  type ModelResidency,
  type ResolvedModelInstallation,
  type UnsupportedLocalModel,
} from "@magnitudedev/acn-protocol"
import { IcnCatalogInstallations, IcnDownloads, IcnInstances, IcnModelAssessments } from "@magnitudedev/icn"
import { projectInferenceLoadDevice, projectInferenceResidency } from "@magnitudedev/acn-protocol"
import type {
  CatalogInstallationOperation,
  CatalogModel,
  CatalogModelUpdate,
  ModelDownload,
  DrafterVerdict as NativeDrafterVerdict,
  ModelAssessment,
  ModelAssessmentDomainSnapshot,
  ModelAssessmentsSnapshot,
  MeasurementState,
  ModelInstance,
  ModelPerformance,
  ReadyModel,
  UnsupportedLocalModel as NativeUnsupportedLocalModel,
} from "@magnitudedev/icn-protocol/schemas"
import type { CustomModelsConfig, PersistedDrafterChoice } from "@magnitudedev/storage"
import {
  LocalModelSources,
  type CatalogModelSource,
  type CustomModelSource,
  type LocalModelSourcesState,
} from "./local-model-sources"
import { CustomModels } from "./custom-models"
import {
  drafterReferenceKey,
  drafterReferenceNaming,
  productDrafterCandidate,
  productDrafterReference,
} from "./drafter-references"
import { productUnsupportedReasons } from "./unsupported-reasons"
import {
  acquisitionFailure,
  currentDownload,
  customModelDownloads,
  downloadIsActive,
  transferProgress,
  type RelatedDownload,
} from "./custom-model-downloads"
import { materializeProjection } from "./materialized-projection"
import { modelRankingScores, type LocalModelRankingCandidate } from "./local-model-ranking-policy"
import { LocalModelRemovals, type LocalModelRemovalState } from "./local-model-removals"

// Compiled once: `Schema.validateSync(schema)` rebuilds its parser on every call, which costs
// seconds of event-loop time when the whole list is projected on each inference change.
const decodeLocalModelAssessment = Schema.decodeUnknownSync(LocalModelAssessmentSchema)
const decodeDrafterVerdict = Schema.decodeUnknownOption(DrafterVerdictSchema)
const validateModelCapabilities = Schema.validateOption(ModelCapabilitiesSchema)
const validateDoesNotFitAssessment = Schema.validateSync(LocalModelDoesNotFitAssessmentSchema)
const validateCatalogLocalModel = Schema.validateSync(CatalogLocalModelSchema)
const validateCustomLocalModel = Schema.validateSync(CustomLocalModelSchema)
const validateUnsupportedLocalModel = Schema.validateOption(UnsupportedLocalModelSchema)
const decodeUnsupportedLocalModel = Schema.decodeUnknownOption(UnsupportedLocalModelSchema)
const decodeVariantLabel = Schema.decodeUnknownSync(ModelVariantLabelSchema)
const decodeReleaseDate = Schema.decodeUnknownSync(ModelReleaseDateSchema)
const decodeCustomFormModelId = Schema.decodeUnknownOption(CustomFormModelIdSchema)
const decodeCatalogFormModelId = Schema.decodeUnknownOption(CatalogFormModelIdSchema)
const decodeCustomModelOrigin = Schema.decodeUnknownSync(CustomModelOriginSchema)

const failure = (code: string, message: string, retryable = false): ModelFailure => ({ code, message, retryable })

const assessmentDomainProgress = (
  domain: ModelAssessmentDomainSnapshot,
  sourceRevision: number,
): { readonly complete: boolean; readonly settledModels: number; readonly totalModels: number } => {
  switch (domain._tag) {
    case "Available": {
      const totalModels = domain.entries.length
      const settledModels = domain.entries.filter(({ state }) => state._tag !== "Assessing").length
      return {
        complete: domain.sourceRevision === sourceRevision && settledModels === totalModels,
        settledModels,
        totalModels,
      }
    }
    case "Pending": return { complete: false, settledModels: 0, totalModels: 0 }
  }
}

export const projectLocalModelPreparation = (
  source: LocalModelSourcesState,
  assessments: ModelAssessmentsSnapshot,
): LocalModelPreparation => {
  const discovery = {
    complete: source.reconciliationComplete,
    modelsFound: source.customModels.length,
  }
  const catalog = assessmentDomainProgress(assessments.catalog, source.catalogRevision)
  const custom = assessmentDomainProgress(assessments.custom, source.localModelsRevision)
  return {
    discovery,
    assessment: {
      complete: catalog.complete && custom.complete,
      settledModels: catalog.settledModels + custom.settledModels,
      totalModels: catalog.totalModels + custom.totalModels,
    },
  }
}

type CoordinatedLocalModelAssessment =
  | { readonly _tag: "Assessing" }
  | {
      readonly _tag: "Assessed"
      readonly assessment: LocalModelAssessment
      readonly capabilities: ModelCapabilities
      readonly drafter: DrafterVerdict | undefined
    }
  | { readonly _tag: "Dropped" }

type VisibleCoordinatedLocalModelAssessment = Exclude<CoordinatedLocalModelAssessment, { readonly _tag: "Dropped" }>

export const assessmentTargetVisible = (
  assessment: CoordinatedLocalModelAssessment | undefined,
): assessment is VisibleCoordinatedLocalModelAssessment | undefined => assessment?._tag !== "Dropped"

const projectMeasurement = (measurement: MeasurementState): MachineMeasurement =>
  measurement._tag === "Failed"
    ? { _tag: "Failed", retryAt: measurement.retryAt }
    : { _tag: measurement._tag }

/** A pending speed carries the machine's measurement progress, which is what it waits on. */
const projectPerformance = (performance: ModelPerformance, measurement: MeasurementState) =>
  performance._tag === "Pending"
    ? { _tag: "Pending" as const, measurement: projectMeasurement(measurement) }
    : performance

export const projectAssessment = (
  snapshot: ModelAssessmentsSnapshot,
  assessment: ModelAssessment,
): LocalModelAssessment => {
  const environmentId = snapshot.environmentId
  if (assessment._tag === "Fits") {
    const totalRequiredBytes = assessment.memory.reduce((total, item) => total + item.requiredBytes, 0)
    const requiredSystemMemoryBytes = assessment.memory.filter((item) => item.memoryDomainId === "system")
      .reduce((total, item) => total + item.requiredBytes, 0)
    return decodeLocalModelAssessment({
      _tag: "Fits", assessmentId: assessment.assessmentId, environmentId,
      profile: assessment.profile,
      memory: {
        domains: assessment.memory, totalRequiredBytes, requiredSystemMemoryBytes,
      },
      performance: projectPerformance(assessment.performance, snapshot.measurement),
    })
  }
  if (assessment._tag === "DoesNotFit") {
    return decodeLocalModelAssessment({
      _tag: "DoesNotFit", assessmentId: assessment.assessmentId, environmentId,
      profile: assessment.profile,
      memoryDomains: assessment.memory,
      totalRequiredBytes: assessment.memory.reduce((total, item) => total + item.requiredBytes, 0),
      deficitBytes: assessment.deficitBytes, limitingResource: assessment.limitingResource,
    })
  }
  return decodeLocalModelAssessment({
    _tag: "Unsupported", environmentId,
    ...Option.match(assessment.profile, { onNone: () => ({}), onSome: (profile) => ({ profile }) }),
    reasons: productUnsupportedReasons(assessment.reasons, assessment.detail), detail: assessment.detail,
  })
}

export const coordinatedAssessment = (
  snapshot: ModelAssessmentsSnapshot,
  sourceRevision: number,
  source: "catalog" | "custom",
  modelId: ModelId,
): CoordinatedLocalModelAssessment | undefined => {
  const domain: ModelAssessmentDomainSnapshot = snapshot[source]
  if (domain.sourceRevision !== sourceRevision || domain._tag === "Pending") return undefined
  const entry = domain.entries.find(({ subject }) => subject.modelId === modelId)
  if (entry === undefined || entry.state._tag === "Assessing") return undefined
  if (entry.state._tag === "Dropped") return { _tag: "Dropped" }
  const assessment = entry.state.profiles[0]
  // Capabilities that violate the product contract fail this one assessment, never the whole list.
  const capabilities = validateModelCapabilities(entry.state.capabilities)
  return assessment === undefined || Option.isNone(capabilities)
    ? { _tag: "Dropped" }
    : {
        _tag: "Assessed",
        assessment: projectAssessment(snapshot, assessment),
        capabilities: capabilities.value,
        drafter: assessment._tag === "Unsupported" ? undefined : productDrafterVerdict(assessment.drafter),
      }
}

export const productDrafterVerdict = (verdict: NativeDrafterVerdict): DrafterVerdict | undefined =>
  Option.getOrUndefined(decodeDrafterVerdict(
    verdict._tag === "Rejected" ? { ...verdict, reasons: productUnsupportedReasons(verdict.reasons, verdict.detail) } : verdict,
  ))

const residencyFor = (modelId: string, instances: readonly ModelInstance[]): ModelResidency => {
  const instance = instances.findLast((candidate) => candidate.modelId === modelId)
  return instance === undefined ? { _tag: "Unloaded" } : projectInferenceResidency(instance)
}

const progress = (operation: CatalogInstallationOperation) => {
  if (operation.state._tag !== "Pending" && operation.state._tag !== "Running") return undefined
  return {
    stage: operation.state.progress.stage,
    completedBytes: operation.state.progress.completedBytes,
    totalBytes: operation.state.progress.totalBytes,
    bytesPerSecond: operation.state.progress.bytesPerSecond,
  }
}

type NativeInstallation = Extract<CatalogModel["localState"], { readonly _tag: "Installed" }>["installation"]

export const productInstallation = (installation: NativeInstallation) => installation._tag === "Resolved"
  ? { _tag: "Resolved" as const, installedBytes: installation.installedBytes, primaryPath: installation.primaryPath,
      ownership: installation.ownership }
  : { _tag: "Unresolved" as const, installedBytes: installation.installedBytes, ownership: installation.ownership }

export const catalogAcquisition = (
  model: Omit<CatalogModel, "id">,
  operation: CatalogInstallationOperation | undefined,
  residencyState: ModelResidency,
): LocalModelAcquisitionState => {
  const installed = model.localState._tag === "Installed"
    ? { installation: productInstallation(model.localState.installation), residencyState }
    : undefined
  if (operation?.state._tag === "Pending" || operation?.state._tag === "Running") {
    const transfer = progress(operation)!
    return installed === undefined
      ? { _tag: "Installing", progress: transfer }
      : { _tag: "Updating", ...installed, progress: transfer }
  }
  if (operation?.state._tag === "Optimizing") {
    const { stage, completed, total, device } = operation.state.progress
    // The download is published; until the catalog observes the installation it reads as finishing.
    const storageBytes = model.desired.metadata.storageBytes
    return installed === undefined
      ? { _tag: "Installing", progress: {
          stage: "publishing", completedBytes: storageBytes, totalBytes: storageBytes, bytesPerSecond: Option.none(),
        } }
      : {
          _tag: "Optimizing",
          ...installed,
          progress: { stage, completed, total, device: Option.map(device, projectInferenceLoadDevice) },
        }
  }
  if (operation?.state._tag === "Failed" && !operation.state.acknowledged
    && !(model.localState._tag === "Installed" && model.localState.updateState._tag === "Current")) {
    return installed === undefined
      ? { _tag: "InstallFailed", failure: operation.state.failure }
      : { _tag: "UpdateFailed", ...installed, failure: operation.state.failure }
  }
  if (installed === undefined) return { _tag: "NotInstalled" }
  const updateState = model.localState.updateState as CatalogModelUpdate
  // ICN reports an update only with something to fetch, so its parts are never empty.
  const parts = updateState._tag === "Available" ? updateState.parts.map(part => part === "Projector" ? "ImageInput" as const : part) : []
  return Array.isNonEmptyReadonlyArray(parts)
    ? { _tag: "UpdateAvailable", ...installed, parts }
    : { _tag: "Installed", ...installed }
}

export const catalogModelServingState = (
  ready: ReadyModel | undefined,
  assessment: VisibleCoordinatedLocalModelAssessment | undefined,
  catalogScores: Pick<LocalModelRankingCandidate, "intelligenceScore" | "fidelityRank">,
  unavailableFailure?: ModelFailure,
): CatalogLocalModelServingState => {
  if (ready === undefined) return { _tag: "Failed", profile: Option.none(), failure: unavailableFailure ?? failure(
    "model_unavailable", "This model is not currently runnable.", true,
  ) }
  if (assessment === undefined || assessment._tag === "Assessing") return {
    _tag: "Assessing", profile: ready.profile,
  }
  const assessed = {
    metadata: ready.metadata,
    capabilities: assessment.capabilities,
    speculativeMethod: ready.speculativeMethod,
  }
  return assessment.assessment._tag === "Fits"
    ? {
        _tag: "Assessed", ...assessed, assessment: assessment.assessment,
        rankingScores: modelRankingScores({
          ...catalogScores,
          profile: assessment.assessment.profile,
          performance: assessment.assessment.performance,
        }),
      }
    // The service never reports a catalog model as Unsupported; validation rejects that contract violation.
    : { _tag: "Assessed", ...assessed,
        assessment: validateDoesNotFitAssessment(assessment.assessment) }
}

const catalogSupport = (support: CatalogModel["support"]) => {
  switch (support.level) {
    case "supported": return { _tag: "Supported" as const }
    case "disabled": return { _tag: "Disabled" as const, reason: support.reason }
    case "deprecated": return {
      _tag: "Deprecated" as const,
      since: support.since,
      replacement: support.replacement,
      reason: support.reason,
    }
  }
}

export const catalogRemovalAcquisition = (
  acquisition: LocalModelAcquisitionState,
  removal: LocalModelRemovalState | undefined,
): LocalModelAcquisitionState => removal !== undefined
  && acquisition._tag !== "NotInstalled"
  && acquisition._tag !== "Installing"
  && acquisition._tag !== "InstallFailed"
  && acquisition._tag !== "Updating"
  ? removal._tag === "Removing"
    ? { _tag: "Removing", installation: acquisition.installation, residencyState: acquisition.residencyState }
    : { _tag: "RemoveFailed", installation: acquisition.installation,
        residencyState: acquisition.residencyState, failure: removal.failure }
  : acquisition

const catalogModel = (
  entry: CatalogModelSource,
  operation: CatalogInstallationOperation | undefined,
  removal: LocalModelRemovalState | undefined,
  assessment: VisibleCoordinatedLocalModelAssessment | undefined,
  instances: readonly ModelInstance[],
  hidden: readonly string[],
): Extract<LocalModel, { readonly _tag: "Catalog" }> => {
  const source = entry.source
  const residency = residencyFor(entry.id, instances)
  const acquisition = catalogAcquisition(source, operation, residency)
  const acquisitionState = catalogRemovalAcquisition(acquisition, removal)
  const ready = source.localState._tag === "NotInstalled"
    ? source.desired
    : source.localState.effective._tag === "Ready"
      ? source.localState.effective.model
      : undefined
  const unavailableFailure = source.localState._tag === "Installed"
    && source.localState.effective._tag === "Unavailable"
    ? source.localState.effective.failure
    : undefined
  const catalogScores = { intelligenceScore: source.intelligence, fidelityRank: source.fidelityRank }
  return validateCatalogLocalModel({
    _tag: "Catalog",
    modelId: entry.id,
    storageBytes: source.desired.metadata.storageBytes,
    presentation: {
      displayName: source.displayName,
      variantLabel: decodeVariantLabel(source.variantLabel),
      description: source.description,
      license: Option.some(source.license),
      sourceUrls: source.sourceUrls,
    },
    catalogData: {
      releaseDate: decodeReleaseDate(source.releaseDate),
      parameterization: source.parameterization,
      intelligence: source.intelligence,
      support: catalogSupport(source.support),
      fidelityRank: source.fidelityRank,
      quantizationAware: source.quantizationAware,
      aliases: Option.getOrElse(source.aliases, () => []).filter((alias) => !hidden.includes(alias)).flatMap((alias) =>
        Option.toArray(decodeCustomFormModelId(alias))),
    },
    acquisitionState,
    servingState: catalogModelServingState(ready, assessment, catalogScores, unavailableFailure),
    missingComponents: source.localState._tag === "Installed" ? source.localState.missingComponents : [],
    storedComponents: source.localState._tag === "Installed" ? source.localState.storedComponents : [],
    drafterWithheld: source.drafterWithheld,
  })
}

const unavailableAssessment = failure(
  "model_assessment_failed",
  "Magnitude could not assess this model on this computer.",
  true,
)

export const customModelServingState = (
  source: CustomModelSource["source"],
  assessment: VisibleCoordinatedLocalModelAssessment | undefined,
): CustomLocalModelServingState => {
  const state = source.state
  if (state._tag === "Unavailable") return { _tag: "Failed", profile: source.profile, failure: state.failure }
  if (state._tag === "NotInstalled" || assessment === undefined || assessment._tag === "Assessing") {
    return { _tag: "Assessing", profile: state._tag === "Ready" ? state.model.profile : source.profile }
  }
  return {
    _tag: "Assessed",
    metadata: state.model.metadata,
    capabilities: assessment.capabilities,
    speculativeMethod: state.model.speculativeMethod,
    assessment: assessment.assessment,
  }
}

const sourceUrls = (modelId: ModelId): readonly string[] => {
  const parsed = parseModelId(modelId)
  return parsed._tag === "HuggingFace" ? [`https://huggingface.co/${parsed.repositoryId}`] : []
}

const fileStem = (path: string) => (path.split(/[\\/]/).at(-1) ?? path).replace(/\.gguf$/i, "")

export interface CustomModelInputs {
  readonly entry: CustomModelSource
  readonly assessment: CoordinatedLocalModelAssessment | undefined
  readonly instances: readonly ModelInstance[]
  readonly related: readonly RelatedDownload[]
  readonly catalog: readonly CatalogModelSource[]
  readonly drafterChoice: DrafterChoice
}

/**
 * The drafter attached to a custom model. A found or imported model whose automatic drafter is not
 * on this computer has no drafter yet: its recommended drafter is offered instead. A verdict always
 * belongs to the attached reference: the assessment's only when it names the attached method, else
 * ICN's verdict for the matching recommended candidate.
 */
export const attachedDrafter = (inputs: CustomModelInputs): Option.Option<AttachedDrafter> => {
  const { entry, assessment, related, catalog, drafterChoice } = inputs
  const added = entry.source.origin._tag === "Added"
  return Option.flatMap(entry.source.drafter, (native) => Option.flatMap(productDrafterReference(native.reference), (reference) => {
    const drafterDownload = currentDownload(related, "Drafter")
    const targetDownload = currentDownload(related, "Target")
    const notDownloaded = native.state._tag === "NotInstalled" && Option.isNone(drafterDownload)
    if (notDownloaded && !added && drafterChoice._tag === "Auto") return Option.none()
    const recommended = Option.flatMap(entry.source.recommendedDrafter, (candidate) => productDrafterCandidate(candidate, catalog))
      .pipe(Option.filter((candidate) => drafterReferenceKey(candidate.reference) === drafterReferenceKey(reference)))
    const assessedVerdict = Option.fromNullable(assessment?._tag === "Assessed" && native.state._tag === "Ready"
      ? assessment.drafter
      : undefined).pipe(Option.filter((verdict) => verdict._tag === "Bound" && verdict.method._tag === native.method._tag))
    const verdict: DrafterVerdict = native.state._tag === "Rejected"
      ? productDrafterVerdict(native.state as NativeDrafterVerdict) ?? { _tag: "None" }
      : Option.getOrElse(
          Option.orElse(assessedVerdict, () => Option.map(recommended, (candidate) => candidate.verdict)),
          (): DrafterVerdict => ({ _tag: "Bound", method: native.method, memoryBytes: 0 }),
        )
    const availability: AttachedDrafter["availability"] = Option.match(drafterDownload, {
      onSome: (download) => downloadIsActive(download)
        ? { _tag: "Downloading", progress: Option.getOrThrow(transferProgress(download)) }
        : { _tag: "DownloadFailed", failure: Option.getOrThrow(acquisitionFailure(download)) },
      onNone: () => {
        if (native.state._tag !== "NotInstalled") return { _tag: "Present" }
        // An added model fetches its drafter with its target; a chosen drafter that is missing needs Sync.
        return Option.match(Option.flatMap(targetDownload, transferProgress), {
          onSome: (progress) => ({ _tag: "Downloading", progress }),
          onNone: () => ({ _tag: "DownloadFailed", failure: Option.getOrElse(
            Option.flatMap(targetDownload, acquisitionFailure),
            () => ({ _tag: "Interrupted" as const }),
          ) }),
        })
      },
    })
    const { name, label } = Option.match(recommended, {
      onNone: () => drafterReferenceNaming(reference, catalog, Option.some(native.method)),
      onSome: (candidate) => candidate,
    })
    return Option.some({
      reference,
      name,
      label,
      sizeBytes: Option.match(recommended, { onNone: () => 0, onSome: ({ sizeBytes }) => sizeBytes }),
      verdict,
      availability,
      updateAvailable: native.updateAvailable,
    })
  }))
}

export const customAcquisition = (
  origin: CustomModelOrigin,
  source: CustomModelSource["source"],
  related: readonly RelatedDownload[],
  residencyState: ModelResidency,
): Option.Option<CustomModelAcquisitionState> => {
  const state = source.state
  const installation: ResolvedModelInstallation | undefined = state._tag === "NotInstalled"
    ? undefined
    : productInstallation(state.installation) as ResolvedModelInstallation
  if (origin._tag !== "Added") {
    if (installation === undefined) return Option.none()
    return Option.some(state._tag === "Unavailable"
      ? { _tag: "Unavailable", installation, failure: state.failure }
      : { _tag: "External", installation, residencyState })
  }
  const installed = installation === undefined ? undefined : { installation, residencyState }
  // ICN reports a newer repository commit only when the added model's target has other content there.
  return Option.some(Option.match(currentDownload(related, "Target"), {
    onNone: (): CustomModelAcquisitionState => {
      if (installed === undefined) return { _tag: "NotInstalled" }
      return Option.isSome(source.updateCommit)
        ? { _tag: "UpdateAvailable", ...installed, parts: ["Model"] }
        : { _tag: "Installed", ...installed }
    },
    onSome: (download): CustomModelAcquisitionState => Option.match(transferProgress(download), {
      onSome: (progress): CustomModelAcquisitionState => installed === undefined
        ? { _tag: "Installing", progress }
        : { _tag: "Updating", ...installed, progress },
      onNone: (): CustomModelAcquisitionState => {
        const failed = Option.getOrThrow(acquisitionFailure(download))
        return installed === undefined
          ? { _tag: "InstallFailed", failure: failed }
          : { _tag: "UpdateFailed", ...installed, failure: failed }
      },
    }),
  }))
}

/** The engine's verdict that an external model cannot run; added models keep their row so they stay removable. */
const externalUnsupportedAssessment = (inputs: CustomModelInputs, origin: CustomModelOrigin) => {
  const { assessment } = inputs
  return origin._tag !== "Added" && assessment?._tag === "Assessed" && assessment.assessment._tag === "Unsupported"
    ? Option.some(assessment.assessment)
    : Option.none()
}

/** One custom row, or none when it has nothing to show: absent external files, a failed assessment, or no support. */
export const customModel = (inputs: CustomModelInputs): Option.Option<CustomLocalModel> => {
  const { entry, assessment, instances, related, catalog } = inputs
  const origin = decodeCustomModelOrigin(entry.source.origin)
  if (assessment?._tag === "Dropped" && origin._tag !== "Added") return Option.none()
  if (Option.isSome(externalUnsupportedAssessment(inputs, origin))) return Option.none()
  const visible = assessment?._tag === "Dropped" ? undefined : assessment
  const acquisitionState = customAcquisition(origin, entry.source, related, residencyFor(entry.id, instances))
  if (Option.isNone(acquisitionState)) return Option.none()
  const ready = entry.source.state._tag === "Ready" ? entry.source.state.model : undefined
  const drafter = attachedDrafter(inputs)
  const servingState: CustomLocalModelServingState = assessment?._tag === "Dropped"
    ? { _tag: "Failed", profile: entry.source.profile, failure: unavailableAssessment }
    : customModelServingState(entry.source, visible)
  return Option.some(validateCustomLocalModel({
    _tag: "Custom",
    modelId: entry.id,
    origin,
    presentation: {
      displayName: entry.source.displayName.trim().length > 0
        ? entry.source.displayName
        : fileStem(entry.source.files[0]?.path ?? entry.id),
      variantLabel: decodeVariantLabel(Option.getOrElse(
        Option.filter(entry.source.variantLabel, (label) => label.length > 0),
        () => ready !== undefined && ready.metadata.quantizationName.length > 0 ? ready.metadata.quantizationName : "GGUF",
      )),
      sourceLabel: customModelSourceLabel(origin),
      variantOf: Option.flatMap(entry.source.variantOf, decodeCatalogFormModelId),
      license: Option.filter(Option.map(entry.source.license, (license) => license.trim()), (license) => license.length > 0),
      sourceUrls: sourceUrls(entry.id),
    },
    drafter,
    drafterPending: entry.source.drafterPending,
    drafterChoice: inputs.drafterChoice,
    recommendedDrafter: Option.flatMap(entry.source.recommendedDrafter, (candidate) => Option.flatMap(
      productDrafterReference(candidate.reference),
      (reference) => productDrafterCandidate(candidate, catalog),
    )),
    acquisitionState: acquisitionState.value,
    servingState,
    storageBytes: entry.source.sizeBytes,
  }))
}

/** A found or imported model whose assessment is `Unsupported`, listed with every engine reason. */
export const unsupportedCustomModel = (inputs: CustomModelInputs): Option.Option<UnsupportedLocalModel> => {
  const { entry } = inputs
  const origin = decodeCustomModelOrigin(entry.source.origin)
  return Option.flatMap(externalUnsupportedAssessment(inputs, origin), (unsupported) => {
    const path = entry.source.files[0]?.path ?? (origin._tag === "Imported" ? origin.path : entry.id)
    return validateUnsupportedLocalModel({
      modelId: Option.some(entry.id),
      displayName: entry.source.displayName.trim().length > 0 ? entry.source.displayName : fileStem(path),
      origin,
      path,
      reasons: unsupported.reasons,
      detail: unsupported.detail,
    })
  })
}

export const unsupportedLocalModel = (model: NativeUnsupportedLocalModel): Option.Option<UnsupportedLocalModel> =>
  decodeUnsupportedLocalModel({
    ...model,
    displayName: model.displayName.trim().length > 0 ? model.displayName : fileStem(model.path),
    reasons: productUnsupportedReasons(model.reasons, model.detail),
  })
export interface LocalModelsApi {
  readonly state: Effect.Effect<LocalModelsState>
  readonly changes: Stream.Stream<void>
  /** The current state, then every change, with no change lost between the two. */
  readonly states: Stream.Stream<LocalModelsState>
  readonly refresh: Effect.Effect<void>
}
export class LocalModels extends Context.Tag("LocalModels")<LocalModels, LocalModelsApi>() {}

export const LocalModelsLive: Layer.Layer<
  LocalModels,
  never,
  | LocalModelSources
  | IcnModelAssessments
  | IcnCatalogInstallations
  | IcnInstances
  | IcnDownloads
  | LocalModelRemovals
  | CustomModels
> = Layer.scoped(LocalModels, Effect.gen(function* () {
  const sources = yield* LocalModelSources
  const assessments = yield* IcnModelAssessments
  const installations = yield* IcnCatalogInstallations
  const instances = yield* IcnInstances
  const downloads = yield* IcnDownloads
  const removalService = yield* LocalModelRemovals
  const customModels = yield* CustomModels
  const project = Effect.gen(function* () {
    const source = yield* sources.state
    const assessment = (yield* assessments.get).state
    const operations = (yield* installations.get).state.operations
    const runtime = (yield* instances.get).instances
    const removals = yield* removalService.state
    const config = yield* customModels.get
    const nativeDownloads = (yield* downloads.get).state.downloads
    return projectLocalModels({ source, assessment, operations, runtime, removals, config, downloads: nativeDownloads })
  })
  const projection = yield* materializeProjection({
    project,
    invalidations: Stream.mergeAll([
      sources.changes.pipe(Stream.map(() => undefined)), assessments.changes.pipe(Stream.map(() => undefined)),
      installations.changes.pipe(Stream.map(() => undefined)), instances.changes.pipe(Stream.map(() => undefined)),
      downloads.changes.pipe(Stream.map(() => undefined)), customModels.changes.pipe(Stream.map(() => undefined)),
      removalService.changes,
    ], { concurrency: "unbounded" }),
    equivalent: dataEquivalence(),
  })
  return LocalModels.of({
    state: projection.get,
    changes: projection.changes.pipe(Stream.map(() => undefined)),
    states: projection.values,
    refresh: sources.refreshLocalModels.pipe(Effect.orDie),
  })
}))

export interface LocalModelsInputs {
  readonly source: LocalModelSourcesState
  readonly assessment: ModelAssessmentsSnapshot
  readonly operations: readonly CatalogInstallationOperation[]
  readonly runtime: readonly ModelInstance[]
  readonly removals: ReadonlyMap<string, LocalModelRemovalState>
  readonly config: CustomModelsConfig
  readonly downloads: readonly ModelDownload[]
}

/**
 * Catalog models stay ranked; custom models are never ranked. ICN lists hidden models apart and never
 * assesses them; a hidden external copy of a catalog model is one of them, under the ID its files have as
 * a found model. A found or imported model the engine cannot run joins the unsupported list.
 */
/** A persisted drafter choice as clients see it; the persisted method is ICN's to receive. */
const drafterChoice = (persisted: PersistedDrafterChoice | undefined): DrafterChoice =>
  persisted === undefined ? { _tag: "Auto" }
    : persisted._tag === "Reference" ? { _tag: "Reference", reference: persisted.reference }
    : persisted

export const projectLocalModels = (inputs: LocalModelsInputs): LocalModelsState => {
  const { source, assessment, operations, runtime, removals, config, downloads } = inputs
  const latestOperation = new Map(operations.map((operation) => [operation.modelId, operation]))
  const customInputs = (entry: CustomModelSource, assessed: CoordinatedLocalModelAssessment | undefined): CustomModelInputs => ({
    entry,
    assessment: assessed,
    instances: runtime,
    related: customModelDownloads(entry.id, config, downloads),
    catalog: source.catalogModels,
    drafterChoice: drafterChoice(config.drafters[entry.id]),
  })
  const custom = source.customModels.map((entry) => customInputs(
    entry,
    coordinatedAssessment(assessment, source.localModelsRevision, "custom", entry.id),
  ))
  const catalog = source.catalogModels.flatMap((entry) => {
    const coordinated = coordinatedAssessment(assessment, source.catalogRevision, "catalog", entry.id)
    return assessmentTargetVisible(coordinated)
      ? [catalogModel(entry, latestOperation.get(entry.id), removals.get(entry.id), coordinated, runtime, config.hidden)]
      : []
  })
  return {
    preparation: projectLocalModelPreparation(source, assessment),
    models: [
      ...catalog,
      ...custom.flatMap((inputs) => Option.toArray(customModel(inputs))),
    ],
    unsupported: [
      ...source.unsupported.flatMap((model) => Option.toArray(unsupportedLocalModel(model))),
      ...custom.flatMap((inputs) => Option.toArray(unsupportedCustomModel(inputs))),
    ],
    hidden: [
      ...source.hiddenModels.flatMap((entry) => Option.toArray(customModel(customInputs(entry, undefined)))),
    ],
  }
}
