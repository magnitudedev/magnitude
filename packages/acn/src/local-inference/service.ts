import { Cause, Context, Effect, Layer, Option, Ref, Scope, Stream } from "effect"
import { createId } from "@magnitudedev/generate-id"
import { IcnApiClient, Generated } from "@magnitudedev/icn"
import {
  LocalInferenceError,
  type LocalInferenceHostProfile,
  type LocalInferenceOperationSnapshot,
  type LocalInferenceRecommendationState,
  type LocalInferenceSnapshot,
  type LocalModelChoice,
  type LocalModelFitAssessment,
  type LocalModelRecommendation,
  LocalInferenceMirror,
} from "@magnitudedev/protocol"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import { makeMirroredState, MirroredStateChanges } from "../mirrored-state"
import {
  LOCAL_MODEL_CATALOG_OVERLAY,
  catalogSourcePageUrl,
  resolveCatalogArtifact,
} from "./catalog"
import type { LocalModelCatalogEntry } from "./types"
import { LocalModelInventoryChanges } from "./inventory-changes"
import { LocalModelConfiguration } from "./model-configuration"
import { reconcileSelectedServingConfiguration } from "./serving-configuration"
import { AcnActivityTracker } from "../activity-tracker"

const PREVIEW_REQUEST_CONCURRENCY = 12
const PROFILE_CONTEXTS = [200_000, 100_000] as const
const PROFILE_PARALLEL_SEQUENCES = 1
const MAX_RECOMMENDATIONS = 4
const MATERIALLY_LIGHTER_RATIO = 0.8
const TERMINAL_OPERATION_HISTORY = 24

interface RecommendationResult {
  readonly recommendations: readonly LocalModelRecommendation[]
  readonly failureCount: number
}

interface ResolvedCatalog {
  readonly entries: readonly LocalModelCatalogEntry[]
  readonly commits: readonly string[]
  readonly failureCount: number
}

export interface LocalInferenceApi {
  readonly state: Effect.Effect<LocalInferenceSnapshot, LocalInferenceError>
  readonly downloadModel: (configurationId: string, requestId: string) => Effect.Effect<{ readonly operationId: string }, LocalInferenceError>
  readonly activateModel: (selectionId: string, requestId: string) => Effect.Effect<{ readonly operationId: string }, LocalInferenceError>
  readonly deleteModel: (selectionId: string) => Effect.Effect<void, LocalInferenceError>
  readonly restart: (requestId: string) => Effect.Effect<{ readonly operationId: string }, LocalInferenceError>
  readonly disable: Effect.Effect<void, LocalInferenceError>
}

export class LocalInference extends Context.Tag("LocalInference")<LocalInference, LocalInferenceApi>() {}

const localError = (code: LocalInferenceError["code"], operation: string, cause: unknown, retryable = true) =>
  new LocalInferenceError({
    code,
    operation,
    message: cause instanceof Error ? cause.message : String(cause),
    retryable,
  })

const profileFor = (
  entry: LocalModelCatalogEntry,
  contextLength: number,
) => ({
  id: `${entry.id}:p${PROFILE_PARALLEL_SEQUENCES}:ctx${contextLength}`,
  context_length: contextLength,
  parallel_sequences: PROFILE_PARALLEL_SEQUENCES,
})

const sourceFor = (entry: LocalModelCatalogEntry): Generated.ModelPreviewSourceSchema => ({
  repository: entry.repo,
  revision: entry.revision,
  primary_gguf: entry.primaryGguf,
  additional_components: entry.additionalComponents,
})

const isStoredArtifactStatus = (availability: Generated.ModelAvailabilitySchema): boolean =>
  availability.type === "available"

const fitAssessment = (assessment: Generated.HardwareAssessmentSchema): LocalModelFitAssessment => {
  if (assessment.type !== "fits" && assessment.type !== "does_not_fit") {
    return { _tag: "NotAssessed" }
  }
  const memory = assessment.memory
  return {
    _tag: "Assessed",
    requiredTotalBytes: memory.required_bytes,
    domains: memory.domains.map((domain) => ({
      memoryDomainId: domain.memory_domain,
      requiredBytes: domain.required_bytes,
      stableCapacityBytes: domain.available_bytes,
      marginBytes: domain.margin_bytes,
    })),
    result: assessment.type === "fits" ? "fits" : "does_not_fit",
  }
}

const displayBackendName = (backend: string): string =>
  backend.toUpperCase() === "MTL" ? "Metal" : backend

export const hostToWire = (hardware: Generated.HardwareSnapshotSchema): LocalInferenceHostProfile => ({
  platform: hardware.platform,
  architecture: hardware.architecture,
  systemMemoryBytes: hardware.system_memory.total_bytes,
  cpuModel: Option.getOrNull(hardware.cpu_model),
  logicalCores: Math.max(1, hardware.logical_cores),
  memoryDomains: hardware.memory_domains.map((domain) => ({
    id: domain.id,
    kind: domain.kind,
    totalCapacityBytes: domain.total_capacity_bytes,
    stableCapacityBytes: domain.stable_capacity_bytes,
    currentFreeBytes: Option.getOrNull(domain.current_free_bytes),
    sharesSystemMemory: domain.shares_system_memory,
    backendNames: [...new Set(domain.devices
      .filter((device) => device.kind !== "cpu")
      .map((device) => displayBackendName(device.backend)))],
    deviceNames: domain.devices
      .filter((device) => device.kind !== "cpu")
      .map((device) => device.description),
    splitGroupId: null,
  })),
})

const recommendationFrom = (
  entry: LocalModelCatalogEntry,
  preview: Generated.ModelPreviewSchema,
  assessment: Generated.ModelPreviewAssessmentSchema,
): LocalModelRecommendation | null => {
  if (assessment.assessment.type !== "fits") return null
  const properties = preview.properties.type === "inspected" ? preview.properties : undefined
  const profile = assessment.assessment.profile
  const memory = assessment.assessment.memory
  const totalDownloadBytes = preview.components.reduce((total, component) => total + component.size_bytes, 0)
  const quantization = properties ? Option.getOrUndefined(properties.quantization) : undefined
  const architectureName = properties ? Option.getOrUndefined(properties.architecture) : undefined
  const trainingContext = properties ? Option.getOrUndefined(properties.training_context_length) : undefined
  const totalParameters = properties ? Option.getOrNull(properties.parameter_count) : null
  const activeParameters = properties ? Option.getOrNull(properties.active_parameter_count) : null
  const selectedProfile = preview.assessments.find((candidate) => candidate.profile_id === assessment.profile_id)
  if (!selectedProfile) return null
  const contextTokens = Number(assessment.profile_id.match(/:ctx(\d+)$/)?.[1] ?? trainingContext ?? 1)
  const fitClass = profile.acceleration.toLowerCase().includes("hybrid")
    ? "hybrid" as const
    : profile.acceleration.toLowerCase().includes("cpu")
      ? "cpu_or_unified" as const
      : "full_accelerator" as const
  return {
    configurationId: assessment.profile_id,
    catalogModelId: entry.id,
    badge: "alternative",
    displayName: entry.displayName,
    family: entry.family,
    architecture: architectureName?.toLowerCase().includes("moe") ? "moe" : "dense",
    ...(totalParameters === null ? {} : { totalParametersBillions: totalParameters / 1_000_000_000 }),
    ...(activeParameters === null ? {} : { activeParametersBillions: activeParameters / 1_000_000_000 }),
    quantization: {
      format: quantization ?? entry.quantTag,
      quantAwareCheckpoint: entry.quantization.quantAwareCheckpoint,
      fidelityLabel: entry.quantization.fidelityLabel,
      fidelityEvidence: entry.quantization.fidelityEvidence,
      fidelitySourceUrl: entry.quantization.fidelitySourceUrl,
    },
    quantTag: quantization ?? entry.quantTag,
    repo: preview.repository,
    revision: preview.commit,
    files: preview.components.map((component) => ({
      path: component.path,
      role: component.role,
      sizeBytes: component.size_bytes,
      sha256: component.content.type === "sha256" ? component.content.value : "",
    })),
    totalDownloadBytes,
    sourcePageUrl: catalogSourcePageUrl(entry),
    license: entry.license,
    contextTokens,
    modelMaximumContextTokens: trainingContext ?? contextTokens,
    estimatedRuntimeBytes: memory.required_bytes,
    stableCapacityBudgetBytes: memory.available_bytes,
    fitMarginBytes: memory.headroom_bytes,
    fitClass,
    constrainedContext: assessment.assessment.recommendation === "constrained",
    explanation: `${entry.quantization.fidelityLabel}; ${profile.acceleration} placement at ${Math.round(contextTokens / 1_000)}K context.`,
  }
}

export interface RankedRecommendationCandidate {
  readonly value: LocalModelRecommendation
  readonly modelId: string
  readonly modelQualityRank: number
  readonly fidelityRank: number
}

const compareConfiguration = (
  left: RankedRecommendationCandidate,
  right: RankedRecommendationCandidate,
): number =>
  right.fidelityRank - left.fidelityRank
    || right.value.contextTokens - left.value.contextTokens
    || right.value.fitMarginBytes - left.value.fitMarginBytes

const compareProductRank = (
  left: RankedRecommendationCandidate,
  right: RankedRecommendationCandidate,
): number =>
  right.modelQualityRank - left.modelQualityRank
    || right.fidelityRank - left.fidelityRank
    || right.value.contextTokens - left.value.contextTokens
    || right.value.fitMarginBytes - left.value.fitMarginBytes

/**
 * Builds a small set of meaningfully different choices instead of treating every
 * context profile and quantization of one checkpoint as a separate recommendation.
 */
export const selectRecommendationPortfolio = (
  candidates: readonly RankedRecommendationCandidate[],
): readonly LocalModelRecommendation[] => {
  const bestByModel = new Map<string, RankedRecommendationCandidate>()
  for (const candidate of candidates) {
    const current = bestByModel.get(candidate.modelId)
    if (!current || compareConfiguration(candidate, current) < 0) {
      bestByModel.set(candidate.modelId, candidate)
    }
  }

  const ranked = [...bestByModel.values()].sort(compareProductRank)
  const primary = ranked.shift()
  if (!primary) return []

  const selected: Array<{ candidate: RankedRecommendationCandidate; badge: LocalModelRecommendation["badge"] }> = [
    { candidate: primary, badge: "recommended" },
  ]
  const remaining = new Map(ranked.map((candidate) => [candidate.modelId, candidate]))
  const take = (
    badge: LocalModelRecommendation["badge"],
    predicate: (candidate: RankedRecommendationCandidate) => boolean,
  ): void => {
    const candidate = [...remaining.values()].find(predicate)
    if (!candidate || selected.length >= MAX_RECOMMENDATIONS) return
    remaining.delete(candidate.modelId)
    selected.push({ candidate, badge })
  }

  take(
    "lighter",
    (candidate) => candidate.value.estimatedRuntimeBytes
      <= primary.value.estimatedRuntimeBytes * MATERIALLY_LIGHTER_RATIO,
  )
  take("higher_fidelity", (candidate) => candidate.fidelityRank > primary.fidelityRank)
  take("alternative", (candidate) => candidate.value.family !== primary.value.family)

  for (const candidate of remaining.values()) {
    if (selected.length >= MAX_RECOMMENDATIONS) break
    selected.push({ candidate, badge: "alternative" })
  }

  return selected.map(({ candidate, badge }) => ({ ...candidate.value, badge }))
}

const operationStage = (event: Generated.ModelDownloadEventSchema | Generated.ModelLoadEvent): LocalInferenceOperationSnapshot["stage"] => {
  if (event.type === "resolving") return "resolving"
  if (event.type === "checking_space") return "checking_space"
  if (event.type === "ready") return "ready"
  if (event.type === "failed") return "queued"
  return event.stage === "publishing" ? "publishing"
    : event.stage === "downloading" ? "downloading"
    : event.stage === "assessing" ? "assessing"
    : event.stage === "unloading" ? "unloading"
    : event.stage === "loading" ? "loading"
    : event.stage === "verifying" ? "verifying"
    : "queued"
}

const boundOperationHistory = (
  operations: readonly LocalInferenceOperationSnapshot[],
): readonly LocalInferenceOperationSnapshot[] => {
  const active = operations.filter((operation) => operation.status === "running")
  const terminal = operations.filter((operation) => operation.status !== "running").slice(-TERMINAL_OPERATION_HISTORY)
  return [...terminal, ...active]
}

export const LocalInferenceLive: Layer.Layer<
  LocalInference,
  never,
  IcnApiClient | LocalModelInventoryChanges | LocalModelConfiguration | AcnActivityTracker | MirroredStateChanges
> = Layer.scoped(LocalInference, Effect.gen(function* () {
  const client = yield* IcnApiClient
  const changes = yield* LocalModelInventoryChanges
  const configuration = yield* LocalModelConfiguration
  const activity = yield* AcnActivityTracker
  const serviceScope = yield* Scope.Scope
  const recommendationCache = yield* Ref.make<ReadonlyMap<string, RecommendationResult>>(new Map())
  const resolvedCatalog = yield* Ref.make<ReadonlyMap<string, LocalModelCatalogEntry>>(new Map())
  const lock = yield* Effect.makeSemaphore(1)
  const recommendationRefreshLock = yield* Effect.makeSemaphore(1)

  const resolveHubRepository = (repository: string) =>
    client.huggingFace.resolveHuggingFaceRepository({
      payload: { repository, revision: "main" },
    })

  const resolveCanonicalCatalog = Effect.gen(function* () {
    const repositories = [...new Set(LOCAL_MODEL_CATALOG_OVERLAY.models.flatMap((model) =>
      model.artifacts.map((candidate) => candidate.repository)))]
    const results = yield* Effect.forEach(
      repositories,
      (repository) => resolveHubRepository(repository).pipe(
        Effect.map((snapshot) => [repository, snapshot] as const),
        Effect.either,
      ),
      { concurrency: PREVIEW_REQUEST_CONCURRENCY },
    )
    const snapshots = new Map(results.flatMap((result) => result._tag === "Right" ? [result.right] : []))
    const entries = LOCAL_MODEL_CATALOG_OVERLAY.models.flatMap((model) => model.artifacts.flatMap((candidate) => {
      const snapshot = snapshots.get(candidate.repository)
      if (!snapshot) return []
      const entry = resolveCatalogArtifact(model, candidate, {
        repository: snapshot.repository,
        commit: snapshot.commit,
        license: Option.getOrNull(snapshot.license),
        license_url: Option.getOrNull(snapshot.license_url),
        gguf_files: snapshot.gguf_files,
      })
      return entry ? [entry] : []
    }))
    yield* Ref.set(resolvedCatalog, new Map(entries.map((entry) => [entry.id, entry])))
    return {
      entries,
      commits: [...snapshots.entries()].map(([repository, snapshot]) => `${repository}@${snapshot.commit}`).sort(),
      failureCount: LOCAL_MODEL_CATALOG_OVERLAY.models.reduce((count, model) => count + model.artifacts.length, 0)
        - entries.length,
    } satisfies ResolvedCatalog
  })

  yield* reconcileSelectedServingConfiguration(client, configuration).pipe(
    Effect.catchAll((cause) => Effect.logWarning("Unable to migrate legacy local model selection").pipe(
      Effect.annotateLogs({ cause: String(cause) }),
    )),
  )

  const previews = (entries: readonly LocalModelCatalogEntry[]) => Effect.forEach(
    entries,
    (entry) => {
      const contexts = PROFILE_CONTEXTS.filter((context) => entry.supportedContextTokens.includes(context))
      return client.models.previewModel({
        payload: {
          source: sourceFor(entry),
          profiles: contexts.map((context) => profileFor(entry, context)),
        },
      }).pipe(
        Effect.map((preview) => ({ entry, preview })),
        Effect.either,
      )
    },
    { concurrency: PREVIEW_REQUEST_CONCURRENCY },
  )

  const recommendations = (
    onLoading: Effect.Effect<void, LocalInferenceError> = Effect.void,
  ) => Effect.gen(function* () {
    const hardware = yield* client.system.getHardware({})
    yield* onLoading
    const catalog = yield* resolveCanonicalCatalog
    const key = [
      hardware.native_build,
      hardware.topology_fingerprint,
      ...catalog.commits,
    ].join(":")
    const cached = (yield* Ref.get(recommendationCache)).get(key)
    if (cached) return cached
    const totalStableCapacity = hardware.memory_domains.reduce(
      (total, domain) => total + domain.stable_capacity_bytes,
      0,
    )
    const plausibleEntries = catalog.entries.filter((entry) => entry.publishedWeightBytes <= totalStableCapacity)
    const previewResults = yield* previews(plausibleEntries)
    const items = previewResults.flatMap((result) => result._tag === "Right" ? [result.right] : [])
    const candidates = items.flatMap(({ entry, preview }) => preview.assessments.flatMap((assessment) => {
      const value = recommendationFrom(entry, preview, assessment)
      return value ? [{
        value,
        modelId: entry.modelId,
        modelQualityRank: entry.modelQualityRank,
        fidelityRank: entry.quantization.fidelityRank,
      }] : []
    }))
    const recommendations = selectRecommendationPortfolio(candidates)
    const result = {
      recommendations,
      failureCount: catalog.failureCount + previewResults.length - items.length,
    }
    yield* Ref.update(recommendationCache, (current) => new Map(current).set(key, result))
    return result
  })

  const resolveStoredModel = (selectionId: string) => Effect.gen(function* () {
    const models = (yield* client.models.listModels({})).data
    const direct = models.find((model) => model.id === selectionId)
    if (direct) return direct
    const config = yield* configuration.get.pipe(
      Effect.mapError((cause) => localError("configuration_failed", "read local model configuration", cause)),
    )
    const selected = config.selectedProfile
    return selected?.configurationId === selectionId && selected.providerModelId
      ? models.find((model) => model.id === selected.providerModelId)
      : undefined
  })

  const stateValue = (
    recommendationState: LocalInferenceRecommendationState,
    recommendationFailureCount = 0,
    currentOperations: readonly LocalInferenceOperationSnapshot[] = [],
  ) => Effect.gen(function* () {
    const [hardware, inventory] = yield* Effect.all([
      client.system.getHardware({}),
      client.models.listModels({}),
    ], { concurrency: 2 })
    const active = inventory.data.find((model) => model.residency.type === "loaded")
    const activeId = active?.id
    const choices: LocalModelChoice[] = inventory.data
      .filter((model) => isStoredArtifactStatus(model.availability))
      .map((model) => {
        const contextTokens = Option.getOrUndefined(model.serving_configuration)?.profile.context_length
        const properties = model.properties.type === "inspected" ? model.properties : undefined
        const quantization = properties ? Option.getOrUndefined(properties.quantization) : undefined
        return {
          _tag: activeId === model.id ? "Running" as const : "Stored" as const,
          choiceId: model.id,
          displayName: Option.getOrUndefined(model.name) ?? model.id,
          providerModelId: ProviderModelIdSchema.make(model.id),
          ...(contextTokens ? { contextTokens } : {}),
          fitClass: model.hardware.type === "fits"
            ? model.hardware.profile.acceleration.toLowerCase().includes("hybrid") ? "hybrid" as const : "full_accelerator" as const
            : "unknown" as const,
          availability: { _tag: "Available" as const },
          fitAssessment: fitAssessment(model.hardware),
          explanation: model.hardware.type === "fits" ? `${model.hardware.profile.acceleration} placement` : "Stored local model",
          residency: activeId === model.id ? "loaded" as const : "unloaded" as const,
          ...(quantization ? { quantization: {
            format: quantization,
            quantAwareCheckpoint: false,
            fidelityLabel: "Inspected artifact",
            fidelityEvidence: "Artifact properties inspected by ICN.",
            fidelitySourceUrl: "",
          } } : {}),
          ...(model.location.type === "magnitude_cache" || model.location.type === "hugging_face_cache" || model.location.type === "directory"
            ? { sizeBytes: model.location.total_bytes }
            : { sizeBytes: model.location.component.size_bytes }),
        }
      })
    return {
      activeBinding: active && active.residency.type === "loaded" ? {
        selectionId: active.id,
        providerModelId: ProviderModelIdSchema.make(active.id),
        contextTokens: Option.getOrUndefined(active.serving_configuration)?.profile.context_length
          ?? active.residency.context_length,
      } : null,
      host: { _tag: "Available" as const, profile: hostToWire(hardware) },
      choices,
      operations: currentOperations,
      recommendationState,
      warnings: recommendationState._tag === "Ready" && recommendationFailureCount > 0
          ? [{ code: "preview_failed", message: "ICN could not assess the local model catalog. Try again when the inference service is available." }]
          : [],
    }
  }).pipe(Effect.mapError((cause) => localError("icn_unavailable", "read local inference state", cause)))

  const initialRecommendation: LocalInferenceRecommendationState = { _tag: "Loading" }
  const mirror = yield* makeMirroredState(
    LocalInferenceMirror,
    yield* stateValue(initialRecommendation).pipe(Effect.orDie),
  )
  const publishState = (
    recommendationState: LocalInferenceRecommendationState,
    recommendationFailureCount = 0,
  ) => mirror.get.pipe(
    Effect.flatMap((current) => stateValue(
      recommendationState,
      recommendationFailureCount,
      current.state.operations,
    )),
    Effect.flatMap((next) => mirror.setIfChanged(next, (left, right) => JSON.stringify(left) === JSON.stringify(right))),
    Effect.asVoid,
  )
  const refresh = recommendationRefreshLock.withPermits(1)(Effect.gen(function* () {
    const result = yield* recommendations(
      publishState({ _tag: "Loading" }),
    ).pipe(Effect.either)
    yield* (result._tag === "Right"
      ? publishState(
        { _tag: "Ready", recommendations: result.right.recommendations },
        result.right.failureCount,
      )
      : publishState({
        _tag: "Failed",
        message: "ICN could not assess the local model catalog. Try again when the inference service is available.",
      }))
  }).pipe(Effect.catchAll(() => Effect.void)))
  yield* refresh.pipe(Effect.forkScoped)

  const reconcileState = mirror.get.pipe(
    Effect.flatMap((current) => stateValue(
      current.state.recommendationState,
      current.state.warnings.some((warning) => warning.code === "preview_failed") ? 1 : 0,
      current.state.operations,
    )),
    Effect.flatMap((next) => mirror.setIfChanged(next, (left, right) => JSON.stringify(left) === JSON.stringify(right))),
    Effect.asVoid,
  )
  yield* changes.stream.pipe(
    Stream.runForEach(() => reconcileState),
    Effect.catchAll(() => Effect.void),
    Effect.forkScoped,
  )

  const setOperation = (snapshot: LocalInferenceOperationSnapshot) => mirror.update((state) => {
    const previous = state.operations.find((operation) => operation.operationId === snapshot.operationId)
    const next = snapshot.status === "failed" && previous
      ? { ...snapshot, stage: previous.stage }
      : snapshot
    const operations = previous
      ? state.operations.map((operation) => operation.operationId === snapshot.operationId ? next : operation)
      : [...state.operations, next]
    return { ...state, operations: boundOperationHistory(operations) }
  }).pipe(Effect.asVoid)

  const sameTarget = (
    left: LocalInferenceOperationSnapshot["target"],
    right: LocalInferenceOperationSnapshot["target"],
  ): boolean => JSON.stringify(left) === JSON.stringify(right)

  interface AcceptedOperation {
    readonly operation: LocalInferenceOperationSnapshot
    readonly accepted: boolean
  }

  const acceptOperation = (
    requestId: string,
    kind: LocalInferenceOperationSnapshot["kind"],
    target: LocalInferenceOperationSnapshot["target"],
    providerModelId: LocalInferenceOperationSnapshot["providerModelId"],
  ) => mirror.modify<AcceptedOperation>((state) => {
    const existing = state.operations.find((operation) => operation.requestId === requestId)
      ?? state.operations.find((operation) => operation.status === "running" && operation.kind === kind && sameTarget(operation.target, target))
    if (existing) return {
      state,
      result: { operation: existing, accepted: false as const },
      changed: false,
    }
    const now = new Date().toISOString()
    const operation: LocalInferenceOperationSnapshot = {
      operationId: createId(),
      requestId,
      kind,
      target,
      providerModelId,
      status: "running",
      stage: "queued",
      startedAt: now,
      updatedAt: now,
    }
    return {
      state: { ...state, operations: [...state.operations, operation] },
      result: { operation, accepted: true as const },
    }
  }).pipe(Effect.map(({ result }) => result))

  const runAccepted = (
    accepted: Effect.Effect<{ readonly operation: LocalInferenceOperationSnapshot; readonly accepted: boolean }, LocalInferenceError>,
    run: (operation: LocalInferenceOperationSnapshot) => Effect.Effect<void, LocalInferenceError>,
  ) => Effect.gen(function* () {
    const result = yield* accepted
    if (result.accepted) {
      yield* Effect.forkIn(activity.withActiveWork(`local-inference:${result.operation.kind}`, run(result.operation)).pipe(
        Effect.catchAllCause((cause) => setOperation({
          ...result.operation,
          status: "failed",
          failure: Option.match(Cause.failureOption(cause), {
            onNone: () => ({ code: "unexpected_failure", message: Cause.pretty(cause).slice(0, 1_000), retryable: true }),
            onSome: (error) => ({ code: error.code, message: error.message, retryable: error.retryable }),
          }),
          updatedAt: new Date().toISOString(),
        })),
      ), serviceScope)
    }
    return { operationId: result.operation.operationId }
  })

  const downloadModel = (configurationId: string, requestId: string) => Effect.gen(function* () {
    const current = (yield* mirror.get).state
    const recommendation = current.recommendationState._tag === "Ready"
      ? current.recommendationState.recommendations.find((item) => item.configurationId === configurationId)
      : undefined
    if (!recommendation) return yield* localError("invalid_selection", "download local model", "The selected recommendation is no longer available.", false)
    yield* configuration.selectProfile({
      configurationId,
      catalogModelId: recommendation.catalogModelId,
      contextTokens: recommendation.contextTokens,
    }).pipe(Effect.mapError((cause) => localError("configuration_failed", "save local model selection", cause)))
    return yield* runAccepted(
      acceptOperation(requestId, "download", { _tag: "configuration", configurationId }, ProviderModelIdSchema.make(recommendation.catalogModelId)),
      (operation) => {
        const request: Generated.DownloadModelRequestSchema = {
          source: { type: "hugging_face", repository: recommendation.repo, revision: recommendation.revision },
          components: recommendation.files.map((file, index) => ({
            path: file.path,
            role: file.role,
            expected_sha256: file.sha256 ? Option.some(file.sha256) : Option.none(),
            shard_index: index === 0 ? Option.none() : Option.some(index),
          })),
          relationships: [],
          serving_profile: {
            context_length: recommendation.contextTokens,
            parallel_sequences: PROFILE_PARALLEL_SEQUENCES,
          },
        }
        return client.models.downloadModel({ payload: request }).pipe(
          Effect.flatMap(({ events }) => Stream.runForEach(events, (event) => {
            const modelId = event.type === "ready"
              ? event.model.id
              : event.type === "checking_space" || event.type === "progress"
                ? event.model_id
                : event.type === "failed"
                  ? Option.getOrNull(event.model_id) ?? recommendation.catalogModelId
                  : recommendation.catalogModelId
            const total = event.type === "checking_space" || event.type === "progress" || event.type === "failed" ? event.total_bytes : 0
            const completed = event.type === "checking_space" || event.type === "progress" || event.type === "failed" ? event.completed_bytes : 0
            const update = setOperation({
              ...operation,
              upstreamOperationId: event.operation_id,
              providerModelId: ProviderModelIdSchema.make(modelId),
              status: event.type === "failed" ? "failed" : event.type === "ready" ? "completed" : "running",
              stage: operationStage(event),
              ...(total > 0 ? { progress: { completedBytes: completed, totalBytes: total } } : {}),
              ...(event.type === "failed" ? { failure: {
                code: event.error.code,
                message: event.error.message,
                retryable: event.error.retryable,
              } } : {}),
              updatedAt: new Date().toISOString(),
            })
            return event.type === "ready"
              ? update.pipe(Effect.zipRight(configuration.selectProfile({
                  configurationId,
                  catalogModelId: recommendation.catalogModelId,
                  contextTokens: recommendation.contextTokens,
                  providerModelId: event.model.id,
                }).pipe(Effect.orDie)))
              : update
          })),
          Effect.mapError((cause) => localError("configuration_failed", "download local model", cause)),
          Effect.zipRight(reconcileState),
          Effect.zipRight(changes.publish),
        )
      },
    )
  })

  const activateModel = (selectionId: string, requestId: string) => runAccepted(
    acceptOperation(requestId, "activate", { _tag: "model", selectionId }, ProviderModelIdSchema.make(selectionId)),
    (operation) => Effect.gen(function* () {
      const model = yield* resolveStoredModel(selectionId).pipe(
        Effect.mapError((cause) => localError("artifact_unavailable", "activate local model", cause)),
      )
      if (!model) {
        return yield* localError(
          "artifact_unavailable",
          "activate local model",
          "The selected model is not available in ICN.",
          false,
        )
      }
      yield* reconcileSelectedServingConfiguration(client, configuration, [model]).pipe(
        Effect.mapError((cause) => localError("configuration_failed", "configure local model serving", cause)),
      )
      yield* client.models.loadModel({ path: { model_id: model.id } }).pipe(
        Effect.flatMap(({ events }) => Stream.runForEach(events, (event) => setOperation({
          ...operation,
          upstreamOperationId: event.operation_id,
          providerModelId: ProviderModelIdSchema.make(event.model_id),
          status: event.type === "failed" ? "failed" : event.type === "ready" ? "completed" : "running",
          stage: operationStage(event),
          ...(event.type === "failed" ? { failure: {
            code: event.code,
            message: event.message,
            retryable: event.retryable,
          } } : {}),
          updatedAt: new Date().toISOString(),
        }))),
        Effect.mapError((cause) => localError("runtime_start_failed", "activate local model", cause)),
      )
      yield* reconcileState
      yield* changes.publish
    }),
  )

  const deleteModel = (selectionId: string) => lock.withPermits(1)(Effect.gen(function* () {
    const model = yield* resolveStoredModel(selectionId).pipe(Effect.mapError((cause) => localError("artifact_unavailable", "delete local model", cause)))
    if (!model) return yield* localError("artifact_unavailable", "delete local model", "The selected model is not available in ICN.", false)
    yield* client.models.deleteModel({ path: { model_id: model.id }, urlParams: { dry_run: Option.none() } }).pipe(
      Effect.mapError((cause) => localError("artifact_active", "delete local model", cause)),
    )
    yield* reconcileState
    yield* changes.publish
  }))

  const disable = lock.withPermits(1)(Effect.gen(function* () {
    const models = yield* configuration.getModels.pipe(
      Effect.mapError((cause) => localError("configuration_failed", "read model slots", cause)),
    )
    const slots = models?.slots ?? {}
    const updates = Object.fromEntries(
      (["primary", "secondary"] as const)
        .filter((slotId) => slots[slotId]?.providerId === "local")
        .map((slotId) => [slotId, {}]),
    )
    if (Object.keys(updates).length > 0) {
      yield* configuration.updateSlots(updates).pipe(
        Effect.mapError((cause) => localError("configuration_failed", "clear local model slots", cause)),
      )
    }
    const inventory = yield* client.models.listModels({}).pipe(
      Effect.mapError((cause) => localError("icn_unavailable", "read local models", cause)),
    )
    yield* Effect.forEach(
      inventory.data.filter((model) => model.residency.type === "loaded"),
      (model) => client.models.unloadModel({ path: { model_id: model.id } }),
      { discard: true },
    ).pipe(Effect.mapError((cause) => localError("configuration_failed", "disable local inference", cause)))
    yield* reconcileState
    yield* changes.publish
  }))

  const restart = (requestId: string) => runAccepted(
    acceptOperation(requestId, "restart", { _tag: "runtime" }, ProviderModelIdSchema.make("local-runtime")),
    (operation) => Effect.gen(function* () {
      const inventory = yield* client.models.listModels({}).pipe(
        Effect.mapError((cause) => localError("icn_unavailable", "read local models", cause)),
      )
      const current = inventory.data.find((model) => model.residency.type === "loaded")
      if (!current) {
        yield* setOperation({ ...operation, status: "completed", stage: "ready", updatedAt: new Date().toISOString() })
        return
      }
      yield* client.models.unloadModel({ path: { model_id: current.id } }).pipe(
        Effect.mapError((cause) => localError("runtime_start_failed", "unload local inference for restart", cause)),
      )
      yield* client.models.loadModel({ path: { model_id: current.id } }).pipe(
        Effect.flatMap(({ events }) => Stream.runForEach(events, (event) => setOperation({
          ...operation,
          upstreamOperationId: event.operation_id,
          providerModelId: ProviderModelIdSchema.make(event.model_id),
          status: event.type === "failed" ? "failed" : event.type === "ready" ? "completed" : "running",
          stage: operationStage(event),
          ...(event.type === "failed" ? { failure: {
            code: event.code,
            message: event.message,
            retryable: event.retryable,
          } } : {}),
          updatedAt: new Date().toISOString(),
        }))),
        Effect.mapError((cause) => localError("runtime_start_failed", "restart local inference", cause)),
      )
      yield* reconcileState
      yield* changes.publish
    }),
  )

  return LocalInference.of({
    state: mirror.get,
    downloadModel,
    activateModel,
    deleteModel,
    restart,
    disable,
  })
}))
