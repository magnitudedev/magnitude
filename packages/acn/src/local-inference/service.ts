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
import { LOCAL_MODEL_CATALOG, catalogSourcePageUrl } from "./catalog"
import type { LocalModelCatalogEntry } from "./types"
import { LocalModelInventoryChanges } from "./inventory-changes"
import { LocalModelConfiguration } from "./model-configuration"
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
  policy: string,
) => ({
  id: `${entry.id}:p${PROFILE_PARALLEL_SEQUENCES}:ctx${contextLength}`,
  policy,
  context_length: contextLength,
  parallel_sequences: PROFILE_PARALLEL_SEQUENCES,
})

const sourceFor = (entry: LocalModelCatalogEntry): Generated.ModelPreviewSourceSchema => ({
  repository: entry.repo,
  revision: entry.revision,
  primary_gguf: entry.primaryGguf,
  additional_components: entry.additionalComponents,
})

const isStoredArtifactStatus = (status: Generated.ModelStatusSchema): boolean =>
  status.type === "available"
    || status.type === "loading"
    || status.type === "loaded"
    || status.type === "unloading"
    || status.type === "load_failed"

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

const operationStage = (event: Generated.ModelDownloadEventSchema | Generated.RuntimeModelEvent): LocalInferenceOperationSnapshot["stage"] => {
  if (event.type === "resolving") return "resolving"
  if (event.type === "checking_space") return "checking_space"
  if (event.type === "ready") return "ready"
  if (event.type === "failed") return "verifying"
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
  const lock = yield* Effect.makeSemaphore(1)
  const recommendationRefreshLock = yield* Effect.makeSemaphore(1)

  const previews = (hardware: Generated.HardwareSnapshotSchema) => Effect.forEach(
    LOCAL_MODEL_CATALOG,
    (entry) => {
      const contexts = PROFILE_CONTEXTS.filter((context) => entry.supportedContextTokens.includes(context))
      return client.models.previewModel({
        payload: {
          source: sourceFor(entry),
          profiles: contexts.map((context) => profileFor(entry, context, hardware.assessment_policy)),
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
    const key = [
      hardware.native_build,
      hardware.topology_fingerprint,
      hardware.capacity_policy,
      hardware.assessment_policy,
    ].join(":")
    const cached = (yield* Ref.get(recommendationCache)).get(key)
    if (cached) return cached
    yield* onLoading
    const previewResults = yield* previews(hardware)
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
      failureCount: previewResults.length - items.length,
    }
    yield* Ref.update(recommendationCache, (current) => new Map(current).set(key, result))
    return result
  })

  const resolveCatalogSelection = (selectionId: string) => Effect.gen(function* () {
    const config = yield* configuration.get.pipe(Effect.mapError((cause) => localError("configuration_failed", "read local model configuration", cause)))
    const recommendationList = yield* recommendations().pipe(
      Effect.mapError((cause) => localError("icn_unavailable", "preview local model recommendations", cause)),
    )
    const recommendation = recommendationList.recommendations.find((item) => item.configurationId === selectionId || item.catalogModelId === selectionId)
    const catalogId = recommendation?.catalogModelId ?? config.selectedProfile?.catalogModelId
    return catalogId ? LOCAL_MODEL_CATALOG.find((entry) => entry.id === catalogId) : undefined
  })

  const resolveStoredModel = (selectionId: string) => Effect.gen(function* () {
    const models = (yield* client.models.listModels({})).data
    const direct = models.find((model) => model.id === selectionId)
    if (direct) return direct
    const entry = yield* resolveCatalogSelection(selectionId)
    if (!entry) return undefined
    return models.find((model) => {
      if (model.source.type !== "hugging_face" || model.source.repository !== entry.repo) return false
      const components = model.location.type === "file" ? [model.location.component] : model.location.components
      return components.some((component) => component.path === entry.primaryGguf)
    })
  })

  const stateValue = (
    recommendationState: LocalInferenceRecommendationState,
    recommendationFailureCount = 0,
    currentOperations: readonly LocalInferenceOperationSnapshot[] = [],
  ) => Effect.gen(function* () {
    const [hardware, inventory, runtime] = yield* Effect.all([
      client.system.getHardware({}),
      client.models.listModels({}),
      client.runtime.getRuntimeState({}),
    ], { concurrency: 3 })
    const activeId = runtime.status.type === "ready" ? runtime.status.model_id : undefined
    const choices: LocalModelChoice[] = inventory.data
      .filter((model) => isStoredArtifactStatus(model.status))
      .map((model) => {
        const contextTokens = activeId === model.id && runtime.status.type === "ready"
          ? runtime.status.profile.context_length
          : model.hardware.type === "fits"
            ? model.hardware.profile.context_length
            : undefined
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
      activeBinding: runtime.status.type === "ready" ? {
        selectionId: runtime.status.model_id,
        providerModelId: ProviderModelIdSchema.make(runtime.status.model_id),
        contextTokens: runtime.status.profile.context_length,
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

  const setOperation = (snapshot: LocalInferenceOperationSnapshot) => mirror.update((state) => {
    const operations = state.operations.some((operation) => operation.operationId === snapshot.operationId)
      ? state.operations.map((operation) => operation.operationId === snapshot.operationId ? snapshot : operation)
      : [...state.operations, snapshot]
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
    const entry = LOCAL_MODEL_CATALOG.find((candidate) => candidate.id === recommendation.catalogModelId)
    if (!entry) return yield* localError("invalid_selection", "download local model", "Unknown catalog model.", false)
    yield* configuration.selectProfile({
      configurationId,
      catalogModelId: entry.id,
      contextTokens: recommendation.contextTokens,
    }).pipe(Effect.mapError((cause) => localError("configuration_failed", "save local model selection", cause)))
    return yield* runAccepted(
      acceptOperation(requestId, "download", { _tag: "configuration", configurationId }, ProviderModelIdSchema.make(entry.id)),
      (operation) => {
        const request: Generated.DownloadModelRequestSchema = {
          source: { type: "hugging_face", repository: entry.repo, revision: entry.revision },
          components: recommendation.files.map((file, index) => ({
            path: file.path,
            role: file.role,
            expected_sha256: file.sha256 ? Option.some(file.sha256) : Option.none(),
            shard_index: index === 0 ? Option.none() : Option.some(index),
          })),
          relationships: [],
        }
        return client.models.downloadModel({ payload: request }).pipe(
          Stream.runForEach((event) => {
            const modelId = event.type === "ready"
              ? event.model.id
              : event.type === "checking_space" || event.type === "progress"
                ? event.model_id
                : event.type === "failed"
                  ? Option.getOrNull(event.model_id) ?? entry.id
                  : entry.id
            const total = event.type === "checking_space" || event.type === "progress" || event.type === "failed" ? event.total_bytes : 0
            const completed = event.type === "checking_space" || event.type === "progress" || event.type === "failed" ? event.completed_bytes : 0
            return setOperation({
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
          }),
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
      const config = yield* configuration.get.pipe(
        Effect.mapError((cause) => localError("configuration_failed", "read local model configuration", cause)),
      )
      const contextLength = config.selectedProfile?.contextTokens
        ?? (model.hardware.type === "fits" ? model.hardware.profile.context_length : 100_000)
      const hardware = yield* client.system.getHardware({}).pipe(
        Effect.mapError((cause) => localError("icn_unavailable", "read ICN execution policy", cause)),
      )
      yield* client.runtime.loadRuntimeModel({ payload: {
        model_id: model.id,
        profile: {
          policy: hardware.assessment_policy,
          context_length: contextLength,
          parallel_sequences: PROFILE_PARALLEL_SEQUENCES,
        },
      } }).pipe(
        Stream.runForEach((event) => setOperation({
          ...operation,
          upstreamOperationId: event.operation_id,
          providerModelId: ProviderModelIdSchema.make(event.type === "ready" ? model.id : event.model_id),
          status: event.type === "failed" ? "failed" : event.type === "ready" ? "completed" : "running",
          stage: operationStage(event),
          ...(event.type === "failed" ? { failure: {
            code: event.code,
            message: event.message,
            retryable: event.retryable,
          } } : {}),
          updatedAt: new Date().toISOString(),
        })),
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

  const disable = lock.withPermits(1)(client.runtime.unloadRuntimeModel({}).pipe(
    Effect.mapError((cause) => localError("configuration_failed", "disable local inference", cause)),
    Effect.zipRight(reconcileState),
    Effect.zipRight(changes.publish),
  ))

  const restart = (requestId: string) => runAccepted(
    acceptOperation(requestId, "restart", { _tag: "runtime" }, ProviderModelIdSchema.make("local-runtime")),
    (operation) => Effect.gen(function* () {
      const runtime = yield* client.runtime.getRuntimeState({}).pipe(Effect.mapError((cause) => localError("icn_unavailable", "read local runtime", cause)))
      if (runtime.status.type !== "ready") {
        yield* setOperation({ ...operation, status: "completed", stage: "ready", updatedAt: new Date().toISOString() })
        return
      }
      const current = runtime.status
      yield* client.runtime.unloadRuntimeModel({}).pipe(Effect.mapError((cause) => localError("runtime_start_failed", "restart local inference", cause)))
      yield* client.runtime.loadRuntimeModel({ payload: { model_id: current.model_id, profile: current.profile } }).pipe(
        Stream.runForEach((event) => setOperation({
          ...operation,
          upstreamOperationId: event.operation_id,
          providerModelId: ProviderModelIdSchema.make(event.type === "ready" ? current.model_id : event.model_id),
          status: event.type === "failed" ? "failed" : event.type === "ready" ? "completed" : "running",
          stage: operationStage(event),
          ...(event.type === "failed" ? { failure: {
            code: event.code,
            message: event.message,
            retryable: event.retryable,
          } } : {}),
          updatedAt: new Date().toISOString(),
        })),
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
