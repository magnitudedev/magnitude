import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import {
  LocalModelsStateSchema,
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
  type LocalModel,
  type LocalModelsState,
  type ModelId,
  type ModelFailure,
  type ModelResidency,
} from "@magnitudedev/acn-protocol"
import { IcnCatalogInstallations, IcnInstances } from "@magnitudedev/icn"
import { projectInferenceResidency } from "@magnitudedev/sdk"
import type { CatalogInstallationOperation, CatalogModel, CatalogModelUpdate, ReadyModel } from "@magnitudedev/icn-protocol/schemas"
import {
  LocalModelSources,
  type CatalogModelSource,
  type DiscoveredModelSource,
} from "./local-model-sources"
import { LocalModelAssessor, type CoordinatedLocalModelAssessment } from "./local-model-assessor"
import { materializeProjection } from "./materialized-projection"
import { modelRankingScores } from "./local-model-ranking-policy"
import { LocalModelRemovals, type LocalModelRemovalState } from "./local-model-removals"

const failure = (code: string, message: string, retryable = false): ModelFailure => ({ code, message, retryable })

const residencyFor = (modelId: string, instances: readonly import("@magnitudedev/icn-protocol/schemas").ModelInstance[]): ModelResidency => {
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
  assessment: CoordinatedLocalModelAssessment | undefined,
  rankingScores: Option.Option<LocalModelRankingScores>,
  unavailableFailure?: ModelFailure,
): CatalogLocalModelServingState => {
  if (ready === undefined) return { _tag: "Failed", profile: Option.none(), failure: unavailableFailure ?? failure(
    "model_unavailable", "This model is not currently runnable.", true,
  ) }
  if (assessment === undefined || assessment._tag === "Assessing") return {
    _tag: "Assessing", profile: ready.profile,
  }
  if (assessment._tag === "Failed") return {
    _tag: "Failed", profile: Option.some(ready.profile), failure: assessment.failure,
  }
  const fits = assessment.assessment._tag === "Fits"
  const assessed = {
    metadata: ready.metadata,
    capabilities: Schema.validateSync(ModelCapabilitiesSchema)(ready.capabilities),
    speculativeMethod: ready.speculativeMethod,
  }
  return fits
    ? { _tag: "Assessed", ...assessed, assessment: assessment.assessment, rankingScores }
    : { _tag: "Assessed", ...assessed, assessment: assessment.assessment }
}

export const discoveredModelServingState = (
  ready: ReadyModel,
  assessment: CoordinatedLocalModelAssessment | undefined,
): DiscoveredLocalModelServingState => {
  if (assessment === undefined || assessment._tag === "Assessing") {
    return { _tag: "Assessing", profile: ready.profile }
  }
  if (assessment._tag === "Failed") {
    return { _tag: "Failed", profile: ready.profile, failure: assessment.failure }
  }
  const assessed = {
    metadata: ready.metadata,
    capabilities: Schema.validateSync(ModelCapabilitiesSchema)(ready.capabilities),
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
  assessment: CoordinatedLocalModelAssessment | undefined,
  instances: readonly import("@magnitudedev/icn-protocol/schemas").ModelInstance[],
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
  assessment: CoordinatedLocalModelAssessment | undefined,
  instances: readonly import("@magnitudedev/icn-protocol/schemas").ModelInstance[],
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
      : state._tag === "Unavailable"
        ? { _tag: "Unavailable" as const, installation: state.installation, failure: state.failure }
        : { _tag: "Ambiguous" as const, failure: state.failure },
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
  LocalModelSources | LocalModelAssessor | IcnCatalogInstallations | IcnInstances | LocalModelRemovals
> = Layer.scoped(LocalModels, Effect.gen(function* () {
  const sources = yield* LocalModelSources
  const assessor = yield* LocalModelAssessor
  const installations = yield* IcnCatalogInstallations
  const instances = yield* IcnInstances
  const removalService = yield* LocalModelRemovals
  const project = Effect.gen(function* () {
    const source = yield* sources.state
    const assessment = yield* assessor.snapshot
    const operations = (yield* installations.get).state.operations
    const runtime = (yield* instances.get).instances
    const removals = yield* removalService.state
    const latestOperation = new Map(operations.map((operation) => [operation.modelId, operation]))
    return {
      reconciliationComplete: source.reconciliationComplete,
      models: [
        ...source.catalogModels.map((entry) => catalogModel(entry, latestOperation.get(entry.id),
          removals.get(entry.id),
          assessment.assessments.get(entry.id), runtime)),
        ...source.discoveredModels.map((entry) => discoveredModel(entry,
          assessment.assessments.get(entry.id), runtime)),
      ],
    } satisfies LocalModelsState
  })
  const projection = yield* materializeProjection({
    project,
    invalidations: Stream.mergeAll([
      sources.changes.pipe(Stream.map(() => undefined)), assessor.changes.pipe(Stream.map(() => undefined)),
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
