import { Context, Effect, Layer, Option, Ref, Stream } from "effect"
import { IcnApiClient, Generated } from "@magnitudedev/icn"
import {
  LocalInferenceError,
  type LocalInferenceHostProfile,
  type LocalInferenceOperationSnapshot,
  type LocalInferenceRecommendationState,
  type LocalInferenceSnapshot,
  type LocalInferenceUsageSelection,
  type LocalModelChoice,
  type LocalModelFitAssessment,
  type LocalModelRecommendation,
  type MirroredResourceInvalidation,
} from "@magnitudedev/protocol"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import { makeMirroredResource } from "../mirrored-resource"
import { LOCAL_MODEL_CATALOG, catalogSourcePageUrl } from "./catalog"
import type { LocalModelCatalogEntry } from "./types"
import { LocalInferenceChanges } from "./gateway"
import { LocalModelConfiguration } from "./model-configuration"

const PREVIEW_REQUEST_CONCURRENCY = 12

interface RecommendationResult {
  readonly recommendations: readonly LocalModelRecommendation[]
  readonly failureCount: number
}

export interface LocalInferenceApi {
  readonly state: Effect.Effect<LocalInferenceSnapshot, LocalInferenceError>
  readonly watchState: Stream.Stream<MirroredResourceInvalidation, LocalInferenceError>
  readonly configureUsage: (selection: LocalInferenceUsageSelection) => Effect.Effect<void, LocalInferenceError>
  readonly downloadModel: (configurationId: string) => Effect.Effect<void, LocalInferenceError>
  readonly activateModel: (selectionId: string) => Effect.Effect<void, LocalInferenceError>
  readonly deleteModel: (selectionId: string) => Effect.Effect<void, LocalInferenceError>
  readonly restart: Effect.Effect<void, LocalInferenceError>
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

const parallelSequences = (usage: LocalInferenceUsageSelection): number =>
  usage.sessionConcurrency === "one" ? 1 : 3

const profileFor = (
  entry: LocalModelCatalogEntry,
  usage: LocalInferenceUsageSelection,
  contextLength: number,
  policy: string,
) => ({
  id: `${entry.id}:p${parallelSequences(usage)}:ctx${contextLength}`,
  policy,
  context_length: contextLength,
  parallel_sequences: parallelSequences(usage),
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
  badge: LocalModelRecommendation["badge"],
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
  const parallelSlots = Number(assessment.profile_id.match(/:p(\d+):/)?.[1] ?? 1)
  const fitClass = profile.acceleration.toLowerCase().includes("hybrid")
    ? "hybrid" as const
    : profile.acceleration.toLowerCase().includes("cpu")
      ? "cpu_or_unified" as const
      : "full_accelerator" as const
  return {
    configurationId: assessment.profile_id,
    catalogModelId: entry.id,
    badge,
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
    servingProfile: {
      sessionConcurrency: parallelSlots === 1 ? "one" : "up_to_three",
      parallelSlots,
      contextTokensPerSlot: contextTokens,
      totalContextCapacityTokens: contextTokens * parallelSlots,
      slotAllocation: "uniform",
      runtimeProfileId: assessment.profile_id,
    },
    modelMaximumContextTokens: trainingContext ?? contextTokens,
    estimatedRuntimeBytes: memory.required_bytes,
    stableCapacityBudgetBytes: memory.available_bytes,
    fitMarginBytes: memory.headroom_bytes,
    fitClass,
    constrainedContext: assessment.assessment.recommendation === "constrained",
    explanation: `${entry.quantization.fidelityLabel}; ${profile.acceleration} placement at ${Math.round(contextTokens / 1_000)}K context across ${parallelSlots} local slot${parallelSlots === 1 ? "" : "s"}.`,
  }
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

export const LocalInferenceLive: Layer.Layer<
  LocalInference,
  never,
  IcnApiClient | LocalInferenceChanges | LocalModelConfiguration
> = Layer.scoped(LocalInference, Effect.gen(function* () {
  const client = yield* IcnApiClient
  const changes = yield* LocalInferenceChanges
  const configuration = yield* LocalModelConfiguration
  const operations = yield* Ref.make<ReadonlyMap<string, LocalInferenceOperationSnapshot>>(new Map())
  const recommendationCache = yield* Ref.make<ReadonlyMap<string, RecommendationResult>>(new Map())
  const lock = yield* Effect.makeSemaphore(1)
  const recommendationRefreshLock = yield* Effect.makeSemaphore(1)

  const setOperation = (snapshot: LocalInferenceOperationSnapshot) => Ref.update(
    operations,
    (current) => new Map(current).set(snapshot.operationId, snapshot),
  ).pipe(Effect.zipRight(changes.publish))

  const previews = (usage: LocalInferenceUsageSelection, hardware: Generated.HardwareSnapshotSchema) => Effect.forEach(
    LOCAL_MODEL_CATALOG,
    (entry) => {
      const contexts = [200_000, 100_000, 64_000].filter((context) => entry.supportedContextTokens.includes(context))
      return client.models.previewModel({
        payload: {
          source: sourceFor(entry),
          profiles: contexts.map((context) => profileFor(entry, usage, context, hardware.assessment_policy)),
        },
      }).pipe(
        Effect.map((preview) => ({ entry, preview })),
        Effect.either,
      )
    },
    { concurrency: PREVIEW_REQUEST_CONCURRENCY },
  )

  const recommendations = (
    usage: LocalInferenceUsageSelection | undefined,
    onLoading: Effect.Effect<void, LocalInferenceError> = Effect.void,
  ) => usage
    ? Effect.gen(function* () {
        const hardware = yield* client.system.getHardware({})
        const key = [
          usage.sessionConcurrency,
          hardware.native_build,
          hardware.topology_fingerprint,
          hardware.capacity_policy,
          hardware.assessment_policy,
        ].join(":")
        const cached = (yield* Ref.get(recommendationCache)).get(key)
        if (cached) return cached
        yield* onLoading
        const previewResults = yield* previews(usage, hardware)
        const items = previewResults.flatMap((result) => result._tag === "Right" ? [result.right] : [])
        const candidates = items.flatMap(({ entry, preview }) => preview.assessments.flatMap((assessment) => {
          const value = recommendationFrom(entry, preview, assessment, "alternative")
          return value ? [{ value, rank: entry.modelQualityRank * 1_000 + entry.quantization.fidelityRank }] : []
        })).sort((left, right) => right.rank - left.rank || right.value.contextTokens - left.value.contextTokens)
        const recommendations = candidates.slice(0, 4).map(({ value }, index) => ({
          ...value,
          badge: index === 0 ? "recommended" as const : index === 1 ? "lighter" as const : "alternative" as const,
        }))
        const result = {
          recommendations,
          failureCount: previewResults.length - items.length,
        }
        yield* Ref.update(recommendationCache, (current) => new Map(current).set(key, result))
        return result
      })
    : Effect.succeed({ recommendations: [], failureCount: 0 } satisfies RecommendationResult)

  const resolveCatalogSelection = (selectionId: string) => Effect.gen(function* () {
    const config = yield* configuration.get.pipe(Effect.mapError((cause) => localError("configuration_failed", "read local model configuration", cause)))
    const recommendationList = yield* recommendations(config.usage).pipe(
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
  ) => Effect.gen(function* () {
    const [hardware, inventory, runtime, config] = yield* Effect.all([
      client.system.getHardware({}),
      client.models.listModels({}),
      client.runtime.getRuntimeState({}),
      configuration.get,
    ], { concurrency: 4 })
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
      usage: config.usage ?? null,
      activeBinding: runtime.status.type === "ready" ? {
        selectionId: runtime.status.model_id,
        providerModelId: ProviderModelIdSchema.make(runtime.status.model_id),
        contextTokens: runtime.status.profile.context_length,
      } : null,
      host: { _tag: "Available" as const, profile: hostToWire(hardware) },
      choices,
      operations: [...(yield* Ref.get(operations)).values()],
      recommendationState,
      warnings: recommendationState._tag === "Ready" && recommendationFailureCount > 0
          ? [{ code: "preview_failed", message: "ICN could not assess the local model catalog. Try again when the inference service is available." }]
          : [],
    }
  }).pipe(Effect.mapError((cause) => localError("icn_unavailable", "read local inference state", cause)))

  const initialConfig = yield* configuration.get.pipe(Effect.orDie)
  const initialRecommendation: LocalInferenceRecommendationState = initialConfig.usage
    ? { _tag: "Loading" }
    : { _tag: "NotRequested" }
  const resource = yield* makeMirroredResource(yield* stateValue(initialRecommendation).pipe(Effect.orDie))
  const publishState = (
    recommendationState: LocalInferenceRecommendationState,
    recommendationFailureCount = 0,
  ) => stateValue(recommendationState, recommendationFailureCount).pipe(
    Effect.flatMap((next) => resource.setIfChanged(next, (left, right) => JSON.stringify(left) === JSON.stringify(right))),
    Effect.asVoid,
  )
  const refresh = recommendationRefreshLock.withPermits(1)(Effect.gen(function* () {
    const config = yield* configuration.get.pipe(
      Effect.mapError((cause) => localError("configuration_failed", "read local model configuration", cause)),
    )
    if (!config.usage) {
      yield* publishState({ _tag: "NotRequested" })
      return
    }
    const result = yield* recommendations(
      config.usage,
      publishState({ _tag: "Loading" }),
    ).pipe(Effect.either)
    const latestConfig = yield* configuration.get.pipe(
      Effect.mapError((cause) => localError("configuration_failed", "read local model configuration", cause)),
    )
    if (latestConfig.usage?.sessionConcurrency !== config.usage.sessionConcurrency) return
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
  yield* changes.stream.pipe(Stream.runForEach(() => refresh), Effect.forkScoped)
  yield* refresh.pipe(Effect.forkScoped)

  const configureUsage = (selection: LocalInferenceUsageSelection) => lock.withPermits(1)(
    Ref.set(recommendationCache, new Map()).pipe(
      Effect.zipRight(configuration.updateUsage(selection)),
      Effect.mapError((cause) => localError("configuration_failed", "configure local inference usage", cause)),
      Effect.zipRight(changes.publish),
    ),
  )

  const downloadModel = (configurationId: string) => lock.withPermits(1)(Effect.gen(function* () {
    const config = yield* configuration.get.pipe(Effect.mapError((cause) => localError("configuration_failed", "read local model configuration", cause)))
    const recommendationList = yield* recommendations(config.usage).pipe(
      Effect.mapError((cause) => localError("icn_unavailable", "preview local model recommendations", cause)),
    )
    const recommendation = recommendationList.recommendations.find((item) => item.configurationId === configurationId)
    if (!recommendation) return yield* localError("invalid_selection", "download local model", "The selected recommendation is no longer available.", false)
    const entry = LOCAL_MODEL_CATALOG.find((candidate) => candidate.id === recommendation.catalogModelId)
    if (!entry) return yield* localError("invalid_selection", "download local model", "Unknown catalog model.", false)
    yield* configuration.selectProfile({
      configurationId,
      catalogModelId: entry.id,
      contextTokens: recommendation.contextTokens,
      parallelSlots: recommendation.servingProfile.parallelSlots,
    }).pipe(Effect.mapError((cause) => localError("configuration_failed", "save local model selection", cause)))
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
    yield* client.models.downloadModel({ payload: request }).pipe(Stream.runForEach((event) => {
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
        operationId: event.operation_id,
        providerModelId: ProviderModelIdSchema.make(modelId),
        status: event.type === "failed" ? "failed" : event.type === "ready" ? "completed" : "running",
        stage: operationStage(event),
        ...(total > 0 ? { progress: completed / total } : {}),
        ...(event.type === "failed" ? { message: event.error.message } : {}),
      })
    }), Effect.mapError((cause) => localError("configuration_failed", "download local model", cause)))
    yield* changes.publish
  }))

  const activateModel = (selectionId: string) => lock.withPermits(1)(Effect.gen(function* () {
    const model = yield* resolveStoredModel(selectionId).pipe(Effect.mapError((cause) => localError("artifact_unavailable", "activate local model", cause)))
    if (!model) return yield* localError("artifact_unavailable", "activate local model", "The selected model is not available in ICN.", false)
    const config = yield* configuration.get.pipe(Effect.mapError((cause) => localError("configuration_failed", "read local model configuration", cause)))
    const contextLength = config.selectedProfile?.contextTokens
      ?? (model.hardware.type === "fits" ? model.hardware.profile.context_length : 100_000)
    const sequences = config.selectedProfile?.parallelSlots ?? parallelSequences(config.usage ?? { sessionConcurrency: "one" })
    const hardware = yield* client.system.getHardware({}).pipe(
      Effect.mapError((cause) => localError("icn_unavailable", "read ICN execution policy", cause)),
    )
    yield* client.runtime.loadRuntimeModel({ payload: {
      model_id: model.id,
      profile: { policy: hardware.assessment_policy, context_length: contextLength, parallel_sequences: sequences },
    } }).pipe(Stream.runForEach((event) => setOperation({
      operationId: event.operation_id,
      providerModelId: ProviderModelIdSchema.make(event.type === "ready" ? model.id : event.model_id),
      status: event.type === "failed" ? "failed" : event.type === "ready" ? "completed" : "running",
      stage: operationStage(event),
      ...(event.type === "failed" ? { message: event.message } : {}),
    })), Effect.mapError((cause) => localError("runtime_start_failed", "activate local model", cause)))
    yield* changes.publish
  }))

  const deleteModel = (selectionId: string) => lock.withPermits(1)(Effect.gen(function* () {
    const model = yield* resolveStoredModel(selectionId).pipe(Effect.mapError((cause) => localError("artifact_unavailable", "delete local model", cause)))
    if (!model) return yield* localError("artifact_unavailable", "delete local model", "The selected model is not available in ICN.", false)
    yield* client.models.deleteModel({ path: { model_id: model.id }, urlParams: { dry_run: Option.none() } }).pipe(
      Effect.mapError((cause) => localError("artifact_active", "delete local model", cause)),
    )
    yield* changes.publish
  }))

  const disable = lock.withPermits(1)(client.runtime.unloadRuntimeModel({}).pipe(
    Effect.mapError((cause) => localError("configuration_failed", "disable local inference", cause)),
    Effect.zipRight(changes.publish),
  ))

  const restart = lock.withPermits(1)(Effect.gen(function* () {
    const runtime = yield* client.runtime.getRuntimeState({}).pipe(Effect.mapError((cause) => localError("icn_unavailable", "read local runtime", cause)))
    if (runtime.status.type !== "ready") return
    const current = runtime.status
    yield* client.runtime.unloadRuntimeModel({}).pipe(Effect.mapError((cause) => localError("runtime_start_failed", "restart local inference", cause)))
    yield* client.runtime.loadRuntimeModel({ payload: { model_id: current.model_id, profile: current.profile } }).pipe(
      Stream.runDrain,
      Effect.mapError((cause) => localError("runtime_start_failed", "restart local inference", cause)),
    )
    yield* changes.publish
  }))

  return LocalInference.of({
    state: resource.get,
    watchState: resource.changes,
    configureUsage,
    downloadModel,
    activateModel,
    deleteModel,
    restart,
    disable,
  })
}))
