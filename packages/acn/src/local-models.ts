import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import {
  LocalModelsStateSchema,
  LocalModelAssessmentSchema,
  CatalogLocalModelSchema,
  DiscoveredLocalModelSchema,
  LocalModelPresentationSchema,
  ModelReleaseDateSchema,
  ModelVariantLabelSchema,
  ModelCapabilitiesSchema,
  parseModelId,
  type LocalModelAcquisitionState,
  type CatalogLocalModelServingState,
  type DiscoveredLocalModelServingState,
  type LocalModelRankingScores,
  type LocalModelAssessment,
  type LocalModel,
  type LocalModelsState,
  type LocalModelPreparation,
  type ModelId,
  type ModelFailure,
  type ModelResidency,
} from "@magnitudedev/acn-protocol"
import { IcnCatalogInstallations, IcnInstances, IcnModelAssessments } from "@magnitudedev/icn"
import { projectInferenceResidency } from "@magnitudedev/sdk"
import type {
  CatalogInstallationOperation,
  CatalogModel,
  CatalogModelUpdate,
  ModelAssessment,
  ModelAssessmentDomainSnapshot,
  ModelAssessmentsSnapshot,
  ModelCapabilities,
  ModelInstance,
  ReadyModel,
} from "@magnitudedev/icn-protocol/schemas"
import {
  LocalModelSources,
  type CatalogModelSource,
  type DiscoveredModelSource,
  type LocalModelSourcesState,
} from "./local-model-sources"
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
    case "Pending":
    case "Failed": return { complete: false, settledModels: 0, totalModels: 0 }
  }
}

export const projectLocalModelPreparation = (
  source: LocalModelSourcesState,
  assessments: ModelAssessmentsSnapshot,
): LocalModelPreparation => {
  const discovery = {
    complete: source.reconciliationComplete,
    modelsFound: source.discoveredModels.length,
  }
  if (assessments.state._tag !== "Ready") {
    return {
      discovery,
      assessment: { complete: false, settledModels: 0, totalModels: 0 },
    }
  }
  const catalog = assessmentDomainProgress(assessments.state.catalog, source.catalogRevision)
  const discovered = assessmentDomainProgress(assessments.state.discovered, source.discoveryRevision)
  return {
    discovery,
    assessment: {
      complete: catalog.complete && discovered.complete,
      settledModels: catalog.settledModels + discovered.settledModels,
      totalModels: catalog.totalModels + discovered.totalModels,
    },
  }
}

type CoordinatedLocalModelAssessment =
  | { readonly _tag: "Assessing" }
  | {
      readonly _tag: "Assessed"
      readonly assessment: LocalModelAssessment
      readonly capabilities: ModelCapabilities
    }
  | { readonly _tag: "Dropped" }

type VisibleCoordinatedLocalModelAssessment = Exclude<CoordinatedLocalModelAssessment, { readonly _tag: "Dropped" }>

export const assessmentTargetVisible = (
  assessment: CoordinatedLocalModelAssessment | undefined,
): assessment is VisibleCoordinatedLocalModelAssessment | undefined => assessment?._tag !== "Dropped"

const projectAssessment = (environmentId: string, assessment: ModelAssessment): LocalModelAssessment => {
  if (assessment._tag === "Fits") {
    const totalRequiredBytes = assessment.memory.reduce((total, item) => total + item.requiredBytes, 0)
    const requiredSystemMemoryBytes = assessment.memory.filter((item) => item.memoryDomainId === "system")
      .reduce((total, item) => total + item.requiredBytes, 0)
    return Schema.decodeUnknownSync(LocalModelAssessmentSchema)({
      _tag: "Fits", assessmentId: assessment.assessmentId, environmentId,
      profile: assessment.profile,
      memory: {
        domains: assessment.memory, totalRequiredBytes, requiredSystemMemoryBytes,
        systemUseState: { _tag: "NotObserved" },
        currentHeadroomState: { _tag: "NotObserved" },
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
    _tag: "Incompatible", environmentId, profile: assessment.profile, failure: assessment.failure,
  })
}

export const coordinatedAssessment = (
  snapshot: ModelAssessmentsSnapshot,
  sourceRevision: number,
  source: "catalog" | "discovered",
  modelId: ModelId,
): CoordinatedLocalModelAssessment | undefined => {
  if (snapshot.state._tag !== "Ready") return undefined
  const domain: ModelAssessmentDomainSnapshot = snapshot.state[source]
  if (domain.sourceRevision !== sourceRevision || domain._tag === "Pending") return undefined
  if (domain._tag === "Failed") return undefined
  const entry = domain.entries.find(({ subject }) => subject.modelId === modelId)
  if (entry === undefined || entry.state._tag === "Assessing") return undefined
  if (entry.state._tag === "Dropped") return { _tag: "Dropped" }
  const assessment = entry.state.profiles[0]
  return assessment === undefined
    ? { _tag: "Dropped" }
    : {
        _tag: "Assessed",
        assessment: projectAssessment(snapshot.state.environmentId, assessment),
        capabilities: entry.state.capabilities,
      }
}

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

export const catalogAcquisition = (
  model: Omit<CatalogModel, "id">,
  operation: CatalogInstallationOperation | undefined,
  residencyState: ModelResidency,
): LocalModelAcquisitionState => {
  const installed = model.localState._tag === "Installed"
    ? { installation: model.localState.installation, residencyState }
    : undefined
  if (operation?.state._tag === "Pending" || operation?.state._tag === "Running") {
    const transfer = progress(operation)!
    return installed === undefined
      ? { _tag: "Installing", progress: transfer }
      : { _tag: "Updating", ...installed, progress: transfer }
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
  const fits = assessment.assessment._tag === "Fits"
  const assessed = {
    metadata: ready.metadata,
    capabilities: Schema.validateSync(ModelCapabilitiesSchema)(assessment.capabilities),
    speculativeMethod: ready.speculativeMethod,
  }
  return fits
    ? { _tag: "Assessed", ...assessed, assessment: assessment.assessment, rankingScores }
    : { _tag: "Assessed", ...assessed, assessment: assessment.assessment }
}

export const discoveredModelServingState = (
  ready: ReadyModel,
  assessment: VisibleCoordinatedLocalModelAssessment | undefined,
): DiscoveredLocalModelServingState => {
  if (assessment === undefined || assessment._tag === "Assessing") {
    return { _tag: "Assessing", profile: ready.profile }
  }
  const assessed = {
    metadata: ready.metadata,
    capabilities: Schema.validateSync(ModelCapabilitiesSchema)(assessment.capabilities),
    speculativeMethod: ready.speculativeMethod,
  }
  return { _tag: "Assessed", ...assessed, assessment: assessment.assessment }
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
        intelligenceScore: source.intelligence.score,
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
      fidelityRank: source.fidelityRank,
      quantizationAware: source.quantizationAware,
    },
    acquisitionState,
    servingState: catalogModelServingState(ready, assessment, rankingScores, unavailableFailure),
  })
}

const discoveredPresentation = (id: ModelId, model: ReadyModel | undefined) => {
  const parsed = parseModelId(id)
  const selector = parsed._tag === "HuggingFace" ? parsed.artifactSelector : id
  return Schema.validateSync(LocalModelPresentationSchema)({
    displayName: (selector.split("/").at(-1) ?? selector).replace(/\.gguf$/i, ""),
    variantLabel: Schema.decodeUnknownSync(ModelVariantLabelSchema)(model?.metadata.quantizationName ?? "GGUF"),
    description: `Discovered in the Hugging Face cache`,
    license: Option.none<string>(),
    sourceUrls: parsed._tag === "HuggingFace"
      ? [`https://huggingface.co/${parsed.repositoryId}`]
      : [],
  })
}

const discoveredModel = (
  entry: DiscoveredModelSource,
  assessment: VisibleCoordinatedLocalModelAssessment | undefined,
  instances: readonly ModelInstance[],
): Extract<LocalModel, { readonly _tag: "Discovered" }> => {
  const state = entry.source.state
  const ready = state._tag === "Ready" ? state.model : undefined
  return Schema.validateSync(DiscoveredLocalModelSchema)({
    _tag: "Discovered",
    modelId: entry.id,
    presentation: discoveredPresentation(entry.id, ready),
    state: state._tag === "Ready"
      ? { _tag: "Ready" as const, installation: state.installation,
          residencyState: residencyFor(entry.id, instances),
          catalogAttribution: state.catalogAttribution._tag === "NotInCatalog"
            ? { _tag: "NotInCatalog" as const }
            : { _tag: "AttributionFailed" as const, failure: state.catalogAttribution.failure },
          servingState: discoveredModelServingState(state.model, assessment) }
      : { _tag: "Unavailable" as const, installation: state.installation, failure: state.failure },
  })
}

export interface LocalModelsApi {
  readonly state: Effect.Effect<LocalModelsState>
  readonly changes: Stream.Stream<void>
  readonly refresh: Effect.Effect<void>
}
export class LocalModels extends Context.Tag("LocalModels")<LocalModels, LocalModelsApi>() {}

export const LocalModelsLive: Layer.Layer<
  LocalModels,
  never,
  LocalModelSources | IcnModelAssessments | IcnCatalogInstallations | IcnInstances | LocalModelRemovals
> = Layer.scoped(LocalModels, Effect.gen(function* () {
  const sources = yield* LocalModelSources
  const assessments = yield* IcnModelAssessments
  const installations = yield* IcnCatalogInstallations
  const instances = yield* IcnInstances
  const removalService = yield* LocalModelRemovals
  const project = Effect.gen(function* () {
    const source = yield* sources.state
    const assessment = (yield* assessments.get).state
    const operations = (yield* installations.get).state.operations
    const runtime = (yield* instances.get).instances
    const removals = yield* removalService.state
    const latestOperation = new Map(operations.map((operation) => [operation.modelId, operation]))
    return {
      preparation: projectLocalModelPreparation(source, assessment),
      models: [
        ...source.catalogModels.flatMap((entry) => {
          const coordinated = coordinatedAssessment(assessment, source.catalogRevision, "catalog", entry.id)
          return assessmentTargetVisible(coordinated)
            ? [catalogModel(entry, latestOperation.get(entry.id), removals.get(entry.id), coordinated, runtime)]
            : []
        }),
        ...source.discoveredModels.flatMap((entry) => {
          const coordinated = coordinatedAssessment(assessment, source.discoveryRevision, "discovered", entry.id)
          return assessmentTargetVisible(coordinated) ? [discoveredModel(entry, coordinated, runtime)] : []
        }),
      ],
    } satisfies LocalModelsState
  })
  const projection = yield* materializeProjection({
    project,
    invalidations: Stream.mergeAll([
      sources.changes.pipe(Stream.map(() => undefined)), assessments.changes.pipe(Stream.map(() => undefined)),
      installations.changes.pipe(Stream.map(() => undefined)), instances.changes.pipe(Stream.map(() => undefined)),
      removalService.changes,
    ], { concurrency: "unbounded" }),
    equivalent: Schema.equivalence(LocalModelsStateSchema),
  })
  return LocalModels.of({
    state: projection.get,
    changes: projection.changes.pipe(Stream.map(() => undefined)),
    refresh: sources.refreshDiscovery.pipe(Effect.orDie),
  })
}))
