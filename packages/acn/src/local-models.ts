import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import {
  LocalModelsStateSchema,
  LocalModelAssessmentSchema,
  LocalModelDoesNotFitAssessmentSchema,
  CatalogLocalModelSchema,
  CatalogFormModelIdSchema,
  CustomLocalModelSchema,
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
  type DrafterReference,
  type DrafterVerdict,
  type LocalModelAcquisitionState,
  type CatalogLocalModelServingState,
  type LocalModelRankingScores,
  type LocalModelAssessment,
  type LocalModel,
  type LocalModelsState,
  type LocalModelPreparation,
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
  ModelCapabilities,
  ModelInstance,
  ReadyModel,
  UnsupportedLocalModel as NativeUnsupportedLocalModel,
} from "@magnitudedev/icn-protocol/schemas"
import type { CustomModelsConfig } from "@magnitudedev/storage"
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
import { modelRankingScores } from "./local-model-ranking-policy"
import { LocalModelRemovals, type LocalModelRemovalState } from "./local-model-removals"

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

export const projectAssessment = (environmentId: string, assessment: ModelAssessment): LocalModelAssessment => {
  if (assessment._tag === "Fits") {
    const totalRequiredBytes = assessment.memory.reduce((total, item) => total + item.requiredBytes, 0)
    const requiredSystemMemoryBytes = assessment.memory.filter((item) => item.memoryDomainId === "system")
      .reduce((total, item) => total + item.requiredBytes, 0)
    return Schema.decodeUnknownSync(LocalModelAssessmentSchema)({
      _tag: "Fits", assessmentId: assessment.assessmentId, environmentId,
      profile: assessment.profile,
      memory: {
        domains: assessment.memory, totalRequiredBytes, requiredSystemMemoryBytes,
      },
      performance: assessment.performance,
    })
  }
  if (assessment._tag === "DoesNotFit") {
    return Schema.decodeUnknownSync(LocalModelAssessmentSchema)({
      _tag: "DoesNotFit", assessmentId: assessment.assessmentId, environmentId,
      profile: assessment.profile,
      memoryDomains: assessment.memory,
      totalRequiredBytes: assessment.memory.reduce((total, item) => total + item.requiredBytes, 0),
      deficitBytes: assessment.deficitBytes, limitingResource: assessment.limitingResource,
    })
  }
  return Schema.decodeUnknownSync(LocalModelAssessmentSchema)({
    _tag: "Unsupported", environmentId, profile: assessment.profile,
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
  return assessment === undefined
    ? { _tag: "Dropped" }
    : {
        _tag: "Assessed",
        assessment: projectAssessment(snapshot.environmentId, assessment),
        capabilities: entry.state.capabilities,
        drafter: assessment._tag === "Unsupported" ? undefined : productDrafterVerdict(assessment.drafter),
      }
}

export const productDrafterVerdict = (verdict: NativeDrafterVerdict): DrafterVerdict | undefined =>
  Option.getOrUndefined(Schema.decodeUnknownOption(DrafterVerdictSchema)(
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
  return updateState._tag === "Available"
    ? { _tag: "UpdateAvailable", ...installed }
    : { _tag: "Installed", ...installed }
}

export const catalogModelServingState = (
  ready: ReadyModel | undefined,
  assessment: VisibleCoordinatedLocalModelAssessment | undefined,
  rankingScores: Option.Option<LocalModelRankingScores>,
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
    capabilities: Schema.validateSync(ModelCapabilitiesSchema)(assessment.capabilities),
    speculativeMethod: ready.speculativeMethod,
  }
  return assessment.assessment._tag === "Fits"
    ? { _tag: "Assessed", ...assessed, assessment: assessment.assessment, rankingScores }
    // The service never reports a catalog model as Unsupported; validation rejects that contract violation.
    : { _tag: "Assessed", ...assessed,
        assessment: Schema.validateSync(LocalModelDoesNotFitAssessmentSchema)(assessment.assessment) }
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
  const rankingScores = assessment?._tag === "Assessed" && assessment.assessment._tag === "Fits"
    ? modelRankingScores({
        intelligenceScore: source.intelligence,
        fidelityRank: source.fidelityRank,
        profile: ready?.profile ?? source.desired.profile,
        performance: assessment.assessment.performance,
      })
    : Option.none()
  return Schema.validateSync(CatalogLocalModelSchema)({
    _tag: "Catalog",
    modelId: entry.id,
    storageBytes: source.desired.metadata.storageBytes,
    presentation: {
      displayName: source.displayName,
      variantLabel: Schema.decodeUnknownSync(ModelVariantLabelSchema)(source.variantLabel),
      description: source.description,
      license: Option.some(source.license),
      sourceUrls: source.sourceUrls,
    },
    catalogData: {
      releaseDate: Schema.decodeUnknownSync(ModelReleaseDateSchema)(source.releaseDate),
      parameterization: source.parameterization,
      intelligence: source.intelligence,
      support: catalogSupport(source.support),
      fidelityRank: source.fidelityRank,
      quantizationAware: source.quantizationAware,
    },
    acquisitionState,
    servingState: catalogModelServingState(ready, assessment, rankingScores, unavailableFailure),
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
    capabilities: Schema.validateSync(ModelCapabilitiesSchema)(assessment.capabilities),
    speculativeMethod: state.model.speculativeMethod,
    assessment: assessment.assessment,
  }
}

const sourceUrls = (modelId: ModelId): readonly string[] => {
  const parsed = parseModelId(modelId)
  return parsed._tag === "HuggingFace" ? [`https://huggingface.co/${parsed.repositoryId}`] : []
}

const fileStem = (path: string) => (path.split(/[\\/]/).at(-1) ?? path).replace(/\.gguf$/i, "")

const catalogUpdateAvailable = (catalog: readonly CatalogModelSource[], reference: DrafterReference): boolean =>
  reference._tag === "Catalog" && catalog.some(({ id, source }) => id === reference.modelId
    && source.localState._tag === "Installed" && source.localState.updateState._tag === "Available")

export interface CustomModelInputs {
  readonly entry: CustomModelSource
  readonly assessment: CoordinatedLocalModelAssessment | undefined
  readonly instances: readonly ModelInstance[]
  readonly related: readonly RelatedDownload[]
  readonly catalog: readonly CatalogModelSource[]
}

export const attachedDrafter = (inputs: CustomModelInputs): Option.Option<AttachedDrafter> => {
  const { entry, assessment, related, catalog } = inputs
  return Option.flatMap(entry.source.drafter, (native) => Option.map(productDrafterReference(native.reference), (reference) => {
    const recommended = Option.flatMap(entry.source.recommendedDrafter, (candidate) => productDrafterCandidate(candidate, catalog))
      .pipe(Option.filter((candidate) => drafterReferenceKey(candidate.reference) === drafterReferenceKey(reference)))
    const assessedVerdict = assessment?._tag === "Assessed" ? assessment.drafter : undefined
    const verdict: DrafterVerdict = native.state._tag === "Rejected"
      ? productDrafterVerdict(native.state as NativeDrafterVerdict) ?? { _tag: "None" }
      : assessedVerdict ?? { _tag: "Bound", method: native.method, memoryBytes: 0 }
    const drafterDownload = currentDownload(related, "Drafter")
    const targetDownload = currentDownload(related, "Target")
    const availability: AttachedDrafter["availability"] = Option.match(drafterDownload, {
      onSome: (download) => downloadIsActive(download)
        ? { _tag: "Downloading", progress: Option.getOrThrow(transferProgress(download)) }
        : { _tag: "DownloadFailed", failure: Option.getOrThrow(acquisitionFailure(download)) },
      onNone: () => {
        if (native.state._tag !== "NotInstalled") return { _tag: "Present" }
        // An added model fetches its drafter with its target; any other missing drafter needs Sync.
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
      onNone: () => drafterReferenceNaming(reference, catalog),
      onSome: (candidate) => candidate,
    })
    return {
      reference,
      name,
      label,
      sizeBytes: Option.match(recommended, { onNone: () => 0, onSome: ({ sizeBytes }) => sizeBytes }),
      verdict,
      availability,
      updateAvailable: catalogUpdateAvailable(catalog, reference),
    }
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
  return Option.some(Option.match(currentDownload(related, "Target"), {
    onNone: (): CustomModelAcquisitionState => installed === undefined
      ? { _tag: "NotInstalled" }
      : { _tag: "Installed", ...installed },
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
  const origin = Schema.decodeUnknownSync(CustomModelOriginSchema)(entry.source.origin)
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
  return Option.some(Schema.validateSync(CustomLocalModelSchema)({
    _tag: "Custom",
    modelId: entry.id,
    origin,
    presentation: {
      displayName: entry.source.displayName.trim().length > 0
        ? entry.source.displayName
        : fileStem(entry.source.files[0]?.path ?? entry.id),
      variantLabel: Schema.decodeUnknownSync(ModelVariantLabelSchema)(Option.getOrElse(
        Option.filter(entry.source.variantLabel, (label) => label.length > 0),
        () => ready !== undefined && ready.metadata.quantizationName.length > 0 ? ready.metadata.quantizationName : "GGUF",
      )),
      sourceLabel: customModelSourceLabel(origin),
      variantOf: Option.flatMap(entry.source.variantOf, Schema.decodeUnknownOption(CatalogFormModelIdSchema)),
      license: Option.none(),
      sourceUrls: sourceUrls(entry.id),
    },
    drafter,
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
  const origin = Schema.decodeUnknownSync(CustomModelOriginSchema)(entry.source.origin)
  return Option.flatMap(externalUnsupportedAssessment(inputs, origin), (unsupported) => {
    const path = entry.source.files[0]?.path ?? (origin._tag === "Imported" ? origin.path : entry.id)
    return Schema.validateOption(UnsupportedLocalModelSchema)({
      displayName: entry.source.displayName.trim().length > 0 ? entry.source.displayName : fileStem(path),
      origin,
      path,
      reasons: unsupported.reasons,
      detail: unsupported.detail,
    })
  })
}

export const unsupportedLocalModel = (model: NativeUnsupportedLocalModel): Option.Option<UnsupportedLocalModel> =>
  Schema.decodeUnknownOption(UnsupportedLocalModelSchema)({
    ...model,
    displayName: model.displayName.trim().length > 0 ? model.displayName : fileStem(model.path),
    reasons: productUnsupportedReasons(model.reasons, model.detail),
  })
export interface LocalModelsApi {
  readonly state: Effect.Effect<LocalModelsState>
  readonly changes: Stream.Stream<void>
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
    equivalent: Schema.equivalence(LocalModelsStateSchema),
  })
  return LocalModels.of({
    state: projection.get,
    changes: projection.changes.pipe(Stream.map(() => undefined)),
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

/** A catalog model whose files are all in an external location: a found copy, hidden rather than removed. */
export const catalogFilesExternal = (model: CatalogLocalModel): boolean =>
  "installation" in model.acquisitionState && model.acquisitionState.installation.ownership === "External"

/**
 * Catalog models stay ranked; custom models are never ranked. A catalog model found only in an external
 * location that the user hid is listed apart with hidden custom models. ICN lists hidden models apart and never
 * assesses them. A found or imported model the engine cannot run joins the unsupported list.
 */
export const projectLocalModels = (inputs: LocalModelsInputs): LocalModelsState => {
  const { source, assessment, operations, runtime, removals, config, downloads } = inputs
  const latestOperation = new Map(operations.map((operation) => [operation.modelId, operation]))
  const customInputs = (entry: CustomModelSource, assessed: CoordinatedLocalModelAssessment | undefined): CustomModelInputs => ({
    entry,
    assessment: assessed,
    instances: runtime,
    related: customModelDownloads(entry.id, config, downloads),
    catalog: source.catalogModels,
  })
  const custom = source.customModels.map((entry) => customInputs(
    entry,
    coordinatedAssessment(assessment, source.localModelsRevision, "custom", entry.id),
  ))
  const catalog = source.catalogModels.flatMap((entry) => {
    const coordinated = coordinatedAssessment(assessment, source.catalogRevision, "catalog", entry.id)
    return assessmentTargetVisible(coordinated)
      ? [catalogModel(entry, latestOperation.get(entry.id), removals.get(entry.id), coordinated, runtime)]
      : []
  })
  const catalogHidden = (model: CatalogLocalModel) => config.hidden.includes(model.modelId) && catalogFilesExternal(model)
  return {
    preparation: projectLocalModelPreparation(source, assessment),
    models: [
      ...catalog.filter((model) => !catalogHidden(model)),
      ...custom.flatMap((inputs) => Option.toArray(customModel(inputs))),
    ],
    unsupported: [
      ...source.unsupported.flatMap((model) => Option.toArray(unsupportedLocalModel(model))),
      ...custom.flatMap((inputs) => Option.toArray(unsupportedCustomModel(inputs))),
    ],
    hidden: [
      ...catalog.filter(catalogHidden),
      ...source.hiddenModels.flatMap((entry) => Option.toArray(customModel(customInputs(entry, undefined)))),
    ],
  }
}
