import { Context, Data, Effect, Layer, Match, Option, Predicate, Schema, Stream } from "effect"
import {
  LocalInferenceMemoryDomainIdSchema,
  LocalModelAvailableForDownload,
  LocalModelDownloaded,
  LocalModelDownloading,
  LocalModelDownloadFailed,
  LocalModelIdSchema,
  LocalModelInventoryEntryLifecycle,
  LocalModelInventoryLifecycle,
  LocalModelInventoryLoading,
  LocalModelInventoryMirror,
  LocalModelInventoryReady,
  LocalModelNotFound,
  LocalModelMutationFailed,
  PRIMARY_SLOT_ID,
  SECONDARY_SLOT_ID,
  type LocalModelId,
  type LocalModelInventoryEntry,
  type LocalModelInventoryEntryDetails,
  type LocalModelInventoryState,
  type ModelCapabilities,
  type ProviderModelCatalogEntry,
} from "@magnitudedev/protocol"
import { ReasoningEffortSchema, type ProviderModelId } from "@magnitudedev/sdk"
import {
  IcnClient,
  IcnInventory,
  IcnRecipes,
  type Generated,
  type ModelRecipeRecommendation,
} from "@magnitudedev/icn"
import {
  NativeIcnModelIdSchema,
  PrivateLocalModelIdSchema,
  PROVIDER_ID as LOCAL_PROVIDER_ID,
  candidateLocalModelId,
  localProviderModelId,
  nativeLocalModelId,
  type NativeIcnModelId,
} from "@magnitudedev/icn/provider"
import { makeMirroredState, MirroredStateChanges } from "./mirrored-state"

class InventoryProjectionFailure extends Data.TaggedError("InventoryProjectionFailure")<{
  readonly message: string
}> {}

const candidateId = (recommendation: ModelRecipeRecommendation): LocalModelId =>
  LocalModelIdSchema.make(candidateLocalModelId(recommendation))

const nativeId = (modelId: NativeIcnModelId): LocalModelId => LocalModelIdSchema.make(nativeLocalModelId(modelId))

const providerModelId = (localModelId: LocalModelId) =>
  localProviderModelId(PrivateLocalModelIdSchema.make(localModelId))

const percentage = (completed: number, total: number) =>
  Math.max(0, Math.min(100, total === 0 ? 0 : Math.round(completed / total * 100)))

const locationBytes = (model: Generated.Model): number =>
  typeof model.location.total_bytes === "number" ? model.location.total_bytes : 0

const text = (value: unknown, fallback: string): string =>
  typeof value === "string" ? value : fallback

const resolvedReasoning = (properties: Generated.InventoryPropertiesSchema): ModelCapabilities["reasoning"] => {
  if (properties.type !== "inspected" || properties.reasoning.type !== "supported") {
    return { supported: false, efforts: [], defaultEffort: Option.none() }
  }
  const control = properties.reasoning.control
  const levels = control.type === "effort" || control.type === "effort_and_budget"
    ? control.levels
    : control.type === "toggle"
      ? ["none", "medium"]
      : ["medium"]
  const efforts = [...new Set(levels)].map((level) => ReasoningEffortSchema.make(level))
  const requested = control.type === "effort"
    ? Option.filter(Option.map(
        Option.filter(control.default, Predicate.isNotNull),
        ReasoningEffortSchema.make,
      ), (effort) => efforts.includes(effort))
    : control.type === "effort_and_budget"
      ? Option.filter(Option.map(
          Option.filter(control.default_effort, Predicate.isNotNull),
          ReasoningEffortSchema.make,
        ), (effort) => efforts.includes(effort))
      : Option.fromNullable(efforts[control.type === "toggle" && !control.default ? 0 : efforts.length - 1])
  return {
    supported: true,
    efforts,
    defaultEffort: Option.orElse(requested, () => Option.fromNullable(efforts[0])),
  }
}

const capabilitiesFromNative = (properties: Generated.InventoryPropertiesSchema): ModelCapabilities => ({
  vision: properties.type === "inspected" && properties.modalities.includes("vision"),
  tools: properties.type === "inspected" && properties.tools.type === "supported",
  structuredOutput: properties.type === "inspected" && properties.structured_output.type === "supported",
  reasoning: resolvedReasoning(properties),
})

const candidateCapabilities = (recommendation: ModelRecipeRecommendation): ModelCapabilities => {
  const efforts = [...new Set(recommendation.capabilities.reasoningEfforts)].map((effort) => ReasoningEffortSchema.make(effort))
  const requestedDefault = Option.map(recommendation.capabilities.defaultReasoningEffort, ReasoningEffortSchema.make)
  return {
    vision: recommendation.capabilities.vision,
    tools: recommendation.capabilities.tools,
    structuredOutput: recommendation.capabilities.structuredOutput,
    reasoning: efforts.length === 0
      ? { supported: false, efforts: [], defaultEffort: Option.none() }
      : {
          supported: true,
          efforts,
          defaultEffort: Option.orElse(
            Option.filter(requestedDefault, (effort) => efforts.includes(effort)),
            () => Option.fromNullable(efforts[0]),
          ),
        },
  }
}

const fitFromNative = (model: Generated.Model): LocalModelInventoryEntryDetails["fit"] => {
  if (model.hardware.type === "fits") return {
    _tag: "Fits",
    requiredBytes: model.hardware.memory.required_bytes,
    availableBytes: model.hardware.memory.available_bytes,
    memoryDomainIds: model.hardware.memory.domains.map((domain) =>
      LocalInferenceMemoryDomainIdSchema.make(domain.memory_domain)),
  }
  if (model.hardware.type === "does_not_fit") return {
    _tag: "DoesNotFit",
    requiredBytes: model.hardware.memory.required_bytes,
    availableBytes: model.hardware.memory.available_bytes,
    limitingResource: model.hardware.limiting_resource,
    memoryDomainIds: model.hardware.memory.domains.map((domain) =>
      LocalInferenceMemoryDomainIdSchema.make(domain.memory_domain)),
  }
  return {
    _tag: "DoesNotFit",
    requiredBytes: locationBytes(model),
    availableBytes: 0,
    limitingResource: text(model.hardware.message, "model compatibility"),
    memoryDomainIds: [],
  }
}

const nativeDetails = (
  model: Generated.Model,
  localModelId: LocalModelId,
): Effect.Effect<LocalModelInventoryEntryDetails, InventoryProjectionFailure> => Effect.gen(function* () {
  const fail = (message: string) => new InventoryProjectionFailure({ message })
  if (model.properties.type !== "inspected") {
    return yield* fail(`ICN inventory model ${model.id} has incomplete properties (${model.properties.type})`)
  }
  const inspected = model.properties
  const contextWindow = yield* Option.match(
    Option.filter(inspected.training_context_length, (value): value is number => value !== null && value > 0), {
    onNone: () => fail(`ICN inventory model ${model.id} has no authoritative context window`),
    onSome: Effect.succeed,
  })
  const architecture = yield* Option.match(
    Option.filter(inspected.architecture, Predicate.isNotNull),
    {
      onNone: () => fail(`ICN inventory model ${model.id} has no inspected architecture`),
      onSome: Effect.succeed,
    },
  )
  const quantization = yield* Option.match(
    Option.filter(inspected.quantization, Predicate.isNotNull),
    {
      onNone: () => fail(`ICN inventory model ${model.id} has no inspected quantization`),
      onSome: Effect.succeed,
    },
  )
  const displayName = Option.getOrElse(
    Option.filter(Option.flatMap(model.name, Option.fromNullable), (name) => name.trim().length > 0),
    () => "Local model",
  )
  return {
    localModelId,
    providerModelId: providerModelId(localModelId),
    modelFamilyId: Option.none(),
    displayName,
    family: architecture,
    architecture: architecture.toLowerCase().includes("moe") ? "MixtureOfExperts" : "Dense",
    capabilities: capabilitiesFromNative(model.properties),
    contextWindow,
    maxOutputTokens: Math.max(1, Math.min(32_768, contextWindow)),
    quantization,
    downloadBytes: locationBytes(model),
    fit: fitFromNative(model),
    recommendation: Option.none(),
  }
})

export const candidateDetails = (recommendation: ModelRecipeRecommendation): LocalModelInventoryEntryDetails => ({
  localModelId: candidateId(recommendation),
  providerModelId: providerModelId(candidateId(recommendation)),
  modelFamilyId: Option.none(),
  displayName: recommendation.displayName,
  family: recommendation.family,
  architecture: recommendation.architecture === "moe" ? "MixtureOfExperts" : "Dense",
  capabilities: candidateCapabilities(recommendation),
  contextWindow: recommendation.contextWindow,
  maxOutputTokens: Math.max(1, Math.min(32_768, recommendation.contextWindow)),
  quantization: recommendation.quantTag,
  downloadBytes: recommendation.totalDownloadBytes,
  fit: recommendation.fitMarginBytes >= 0 ? {
    _tag: "Fits",
    requiredBytes: recommendation.estimatedRuntimeBytes,
    availableBytes: recommendation.stableCapacityBudgetBytes,
    memoryDomainIds: [],
  } : {
    _tag: "DoesNotFit",
    requiredBytes: recommendation.estimatedRuntimeBytes,
    availableBytes: recommendation.stableCapacityBudgetBytes,
    limitingResource: "memory",
    memoryDomainIds: [],
  },
  recommendation: Option.some({
    intent: recommendation.intent,
    explanation: recommendation.explanation,
    fidelityLabel: recommendation.quantization.fidelityLabel,
    fidelityEvidence: recommendation.quantization.fidelityEvidence,
    repository: recommendation.repo,
    revision: recommendation.revision,
    files: recommendation.files.map(({ path, sha256 }) => ({ path, sha256 })),
    sourcePageUrl: recommendation.sourcePageUrl,
    estimatedRuntimeBytes: recommendation.estimatedRuntimeBytes,
    fitMarginBytes: recommendation.fitMarginBytes,
    estimatedGeneration: recommendation.estimatedGeneration,
  }),
})

export type LocalInventoryEntryTarget =
  | { readonly kind: "AvailableForDownload"; readonly model: LocalModelInventoryEntryDetails }
  | { readonly kind: "Downloading"; readonly model: LocalModelInventoryEntryDetails; readonly percentage: number; readonly completedBytes: number; readonly totalBytes: number }
  | { readonly kind: "Downloaded"; readonly model: LocalModelInventoryEntryDetails; readonly downloadedBytes: number }
  | { readonly kind: "DownloadFailed"; readonly model: LocalModelInventoryEntryDetails; readonly completedBytes: number; readonly totalBytes: number; readonly error: { readonly code: string; readonly message: string; readonly retryable: boolean } }

const initialInventoryEntry = (target: LocalInventoryEntryTarget): LocalModelInventoryEntry => {
  switch (target.kind) {
    case "AvailableForDownload": return new LocalModelAvailableForDownload({ model: target.model })
    case "Downloading": return new LocalModelDownloading(target)
    case "Downloaded": return new LocalModelDownloaded(target)
    case "DownloadFailed": return new LocalModelDownloadFailed(target)
  }
}

export const transitionLocalInventoryEntry = (
  previous: Option.Option<LocalModelInventoryEntry>,
  desired: LocalInventoryEntryTarget,
): Effect.Effect<LocalModelInventoryEntry, InventoryProjectionFailure> => Effect.gen(function* () {
  if (Option.isNone(previous)) return initialInventoryEntry(desired)
  const current = previous.value
  if (current._tag === desired.kind) {
    switch (current._tag) {
      case "AvailableForDownload": if (desired.kind === "AvailableForDownload") return LocalModelInventoryEntryLifecycle.hold(current, desired)
        break
      case "Downloading": if (desired.kind === "Downloading") return LocalModelInventoryEntryLifecycle.hold(current, {
        ...desired,
        percentage: Math.max(current.percentage, desired.percentage),
        completedBytes: Math.max(current.completedBytes, desired.completedBytes),
        totalBytes: Math.max(current.totalBytes, desired.totalBytes),
      })
        break
      case "Downloaded": if (desired.kind === "Downloaded") return LocalModelInventoryEntryLifecycle.hold(current, desired)
        break
      case "DownloadFailed": if (desired.kind === "DownloadFailed") return LocalModelInventoryEntryLifecycle.hold(current, desired)
        break
    }
  }
  if (current._tag === "AvailableForDownload" && desired.kind === "Downloading") {
    return LocalModelInventoryEntryLifecycle.transition(current, "Downloading", desired)
  }
  if (current._tag === "DownloadFailed" && desired.kind === "Downloading") {
    return LocalModelInventoryEntryLifecycle.transition(current, "Downloading", desired)
  }
  if (current._tag === "Downloading" && desired.kind === "Downloaded") {
    return LocalModelInventoryEntryLifecycle.transition(current, "Downloaded", desired)
  }
  if (current._tag === "Downloading" && desired.kind === "DownloadFailed") {
    return LocalModelInventoryEntryLifecycle.transition(current, "DownloadFailed", desired)
  }
  if (current._tag === "Downloaded" && desired.kind === "AvailableForDownload") {
    return LocalModelInventoryEntryLifecycle.transition(current, "AvailableForDownload", desired)
  }
  return yield* new InventoryProjectionFailure({
    message: `Invalid observed local inventory transition ${current._tag} -> ${desired.kind}`,
  })
})

interface LocalAssociation {
  readonly localModelId: LocalModelId
  readonly nativeModelId?: NativeIcnModelId
  readonly recommendation?: ModelRecipeRecommendation
  readonly nativeModel?: Generated.Model
}

export interface LocalModelInventoryApi {
  readonly snapshot: Effect.Effect<{ readonly revision: number; readonly state: LocalModelInventoryState }>
  readonly changes: Stream.Stream<{ readonly revision: number; readonly state: LocalModelInventoryState }>
  readonly localCatalog: Effect.Effect<readonly ProviderModelCatalogEntry[]>
  readonly providerModelId: (localModelId: LocalModelId) => ProviderModelId
  readonly nativeModelId: (providerModelId: ProviderModelId) => Effect.Effect<NativeIcnModelId, LocalModelMutationFailed>
  readonly download: (localModelId: LocalModelId) => Effect.Effect<void, LocalModelNotFound | LocalModelMutationFailed>
  readonly delete: (localModelId: LocalModelId) => Effect.Effect<void, LocalModelNotFound | LocalModelMutationFailed>
}

export class LocalModelInventory extends Context.Tag("LocalModelInventory")<
  LocalModelInventory,
  LocalModelInventoryApi
>() {}

export const LocalModelInventoryLive: Layer.Layer<
  LocalModelInventory,
  never,
  IcnClient | IcnInventory | IcnRecipes | MirroredStateChanges
> = Layer.scoped(LocalModelInventory, Effect.gen(function* () {
  const client = yield* IcnClient
  const inventory = yield* IcnInventory
  const recipes = yield* IcnRecipes
  const scope = yield* Effect.scope
  const inventoryMirror = yield* makeMirroredState(LocalModelInventoryMirror, new LocalModelInventoryLoading({}))
  const lock = yield* Effect.makeSemaphore(1)
  const equivalent = Schema.equivalence(LocalModelInventoryMirror.stateSchema)
  const updateInventory = (update: (state: LocalModelInventoryState) => LocalModelInventoryState) =>
    inventoryMirror.modify((state) => {
      const next = update(state)
      return { state: next, result: undefined, changed: !equivalent(state, next) }
    })

  const beginRecovery = updateInventory((state) => state._tag === "Failed"
    ? LocalModelInventoryLifecycle.transition(state, "Loading", {})
    : state)

  const associations = Effect.gen(function* () {
    const nativeModels = (yield* inventory.get).state.data
    const recipeState = (yield* recipes.get).state
    const recommendations = recipeState._tag === "Ready" ? recipeState.recommendations : []
    const byNativeId = new Map(recommendations.flatMap((recommendation) =>
      Option.match(recommendation.modelId, {
        onNone: () => [],
        onSome: (modelId) => [[modelId, recommendation] as const],
      })))
    const values = new Map<LocalModelId, LocalAssociation>()
    for (const recommendation of recommendations) {
      const id = candidateId(recommendation)
      values.set(id, { localModelId: id, recommendation })
    }
    for (const model of nativeModels) {
      const recommendation = byNativeId.get(NativeIcnModelIdSchema.make(model.id))
      const id = recommendation ? candidateId(recommendation) : nativeId(NativeIcnModelIdSchema.make(model.id))
      values.set(id, {
        localModelId: id,
        nativeModelId: NativeIcnModelIdSchema.make(model.id),
        ...(recommendation ? { recommendation } : {}),
        nativeModel: model,
      })
    }
    return [...values.values()]
  })

  const publishProjectionFailure = (error: InventoryProjectionFailure) => updateInventory((state) => state._tag === "Loading"
    ? LocalModelInventoryLifecycle.transition(state, "Failed", {
        error: { code: "inventory_projection_failed", message: error.message, retryable: true },
      })
    : state._tag === "Ready"
      ? LocalModelInventoryLifecycle.transition(state, "Failed", {
          error: { code: "inventory_projection_failed", message: error.message, retryable: true },
        })
    : state._tag === "Failed"
      ? LocalModelInventoryLifecycle.hold(state, {
          error: { code: "inventory_projection_failed", message: error.message, retryable: true },
        })
      : state).pipe(
    Effect.andThen(Effect.logWarning("Unable to project local model inventory").pipe(
      Effect.annotateLogs({ cause: error.message }),
    )),
  )

  const rebuildInventory = lock.withPermits(1)(Effect.gen(function* () {
    if ((yield* recipes.get).state._tag !== "Ready") return
    const current = (yield* inventoryMirror.get).state
    const previousEntries = current._tag === "Ready"
      ? new Map(current.entries.map((entry) => [entry.model.localModelId, entry]))
      : new Map<LocalModelId, LocalModelInventoryEntry>()
    const currentAssociations = yield* associations
    const entries = yield* Effect.forEach(currentAssociations, (association): Effect.Effect<LocalModelInventoryEntry, InventoryProjectionFailure> => Effect.gen(function* () {
        if (!association.nativeModel) {
          if (!association.recommendation) {
            return yield* new InventoryProjectionFailure({
              message: `Local model ${association.localModelId} has neither a native model nor a recommendation`,
            })
          }
          const details = candidateDetails(association.recommendation)
          return yield* transitionLocalInventoryEntry(Option.fromNullable(previousEntries.get(association.localModelId)), {
            kind: "AvailableForDownload",
            model: details,
          })
        }
        const model = association.nativeModel
        const details = association.recommendation
          ? { ...candidateDetails(association.recommendation), fit: fitFromNative(model) }
          : yield* nativeDetails(model, association.localModelId)
        const desired = Match.value(model.availability).pipe(
          Match.when({ type: "downloading" }, (availability): LocalInventoryEntryTarget => ({
            kind: "Downloading",
            model: details,
            percentage: percentage(availability.completed_bytes, availability.total_bytes),
            completedBytes: availability.completed_bytes,
            totalBytes: availability.total_bytes,
          })),
          Match.when({ type: "interrupted" }, (availability): LocalInventoryEntryTarget => ({
            kind: "DownloadFailed",
            model: details,
            completedBytes: availability.completed_bytes,
            totalBytes: availability.total_bytes,
            error: { code: "download_interrupted", message: availability.last_error, retryable: availability.resumable },
          })),
          Match.when({ type: "available" }, (): LocalInventoryEntryTarget => ({
            kind: "Downloaded",
            model: details,
            downloadedBytes: locationBytes(model),
          })),
          Match.when({ type: "invalid_artifact" }, (availability): LocalInventoryEntryTarget => ({
            kind: "DownloadFailed",
            model: details,
            completedBytes: locationBytes(model),
            totalBytes: locationBytes(model),
            error: { code: availability.code, message: availability.message, retryable: false },
          })),
          Match.when({ type: "incompatible_artifact" }, (availability): LocalInventoryEntryTarget => ({
            kind: "DownloadFailed",
            model: details,
            completedBytes: locationBytes(model),
            totalBytes: locationBytes(model),
            error: { code: availability.code, message: availability.message, retryable: false },
          })),
          Match.exhaustive,
        )
        return yield* transitionLocalInventoryEntry(Option.fromNullable(previousEntries.get(association.localModelId)), desired)
      })).pipe(Effect.map((projected) => projected.sort((left, right) => left.model.displayName.localeCompare(right.model.displayName))))
    yield* updateInventory((state) => state._tag === "Loading"
      ? LocalModelInventoryLifecycle.transition(state, "Ready", { entries })
      : state._tag === "Ready"
        ? LocalModelInventoryLifecycle.hold(state, { entries })
        : state)
  }).pipe(Effect.catchAll(publishProjectionFailure)))

  const initialNativeInventorySnapshot = yield* inventory.get
  const initialRecipesSnapshot = yield* recipes.get
  yield* rebuildInventory
  yield* Effect.forkIn(Stream.merge(
    inventory.changes.pipe(
      Stream.dropWhile((snapshot) => snapshot.revision <= initialNativeInventorySnapshot.revision),
    ),
    recipes.changes.pipe(
      Stream.dropWhile((snapshot) => snapshot.revision <= initialRecipesSnapshot.revision),
    ),
  ).pipe(
    Stream.runForEach(() => beginRecovery.pipe(Effect.zipRight(rebuildInventory))),
  ), scope)

  const findAssociation = (localModelId: LocalModelId) => associations.pipe(
    Effect.flatMap((values) => Option.match(
      Option.fromNullable(values.find((value) => value.localModelId === localModelId)),
      { onNone: () => Effect.fail(new LocalModelNotFound({ localModelId })), onSome: Effect.succeed },
    )),
  )
  const mapMutationError = (error: unknown) => new LocalModelMutationFailed({
    code: "local_model_mutation_failed",
    message: error instanceof Error ? error.message : String(error),
    retryable: true,
  })

  return LocalModelInventory.of({
    snapshot: inventoryMirror.get,
    changes: inventoryMirror.changes,
    localCatalog: inventoryMirror.get.pipe(Effect.map(({ state }) => state._tag !== "Ready" ? [] : state.entries.flatMap((entry) => {
      if (entry._tag !== "Downloaded") return []
      return [{
        providerId: LOCAL_PROVIDER_ID,
        providerModelId: entry.model.providerModelId,
        modelFamilyId: entry.model.modelFamilyId,
        displayName: entry.model.displayName,
        supportedSlots: [PRIMARY_SLOT_ID, SECONDARY_SLOT_ID],
        contextWindow: entry.model.contextWindow,
        maxOutputTokens: entry.model.maxOutputTokens,
        capabilities: entry.model.capabilities,
        availability: entry.model.fit._tag === "Fits" ? { _tag: "Available" as const } : {
          _tag: "Disabled" as const,
          reason: "insufficient_resources" as const,
        },
        pricing: Option.none(),
      }]
    }))),
    providerModelId,
    nativeModelId: (publicId) => associations.pipe(Effect.flatMap((values) => {
      const found = values.find((association) => providerModelId(association.localModelId) === publicId)
      return found?.nativeModelId
        ? Effect.succeed(found.nativeModelId)
        : Effect.fail(new LocalModelMutationFailed({
            code: "local_model_identity_not_found",
            message: `No native model identity is associated with provider model ${publicId}`,
            retryable: false,
          }))
    })),
    download: (localModelId) => Effect.gen(function* () {
      const association = yield* findAssociation(localModelId)
      if (!association.recommendation) return yield* new LocalModelNotFound({ localModelId })
      const current = (yield* inventoryMirror.get).state
      const currentEntry = current._tag === "Ready"
        ? current.entries.find((entry) => entry.model.localModelId === localModelId)
        : undefined
      if (currentEntry?._tag === "Downloading" || currentEntry?._tag === "Downloaded") return
      if (association.nativeModel?.availability.type === "available"
        || association.nativeModel?.availability.type === "downloading") return
      const recommendation = association.recommendation
      yield* updateInventory((state) => state._tag !== "Ready" ? state : LocalModelInventoryLifecycle.hold(state, {
        entries: state.entries.map((entry) => entry.model.localModelId !== localModelId
          ? entry
          : entry._tag === "Downloading"
            ? entry
            : entry._tag === "AvailableForDownload" || entry._tag === "DownloadFailed"
              ? LocalModelInventoryEntryLifecycle.transition(entry, "Downloading", {
                percentage: 0,
                completedBytes: 0,
                totalBytes: entry.model.downloadBytes,
                })
              : entry),
      }))
      const request: Generated.DownloadModelRequestSchema = {
        source: { type: "hugging_face", repository: recommendation.repo, revision: recommendation.revision },
        components: recommendation.files.map((file, index) => ({
          path: file.path,
          role: file.role,
          expected_sha256: Option.fromNullable(file.sha256),
          shard_index: index === 0 ? Option.none() : Option.some(index),
        })),
        relationships: [],
        serving_profile: { context_length: recommendation.contextWindow, parallel_sequences: 1 },
      }
      const response = yield* client.models.downloadModel({ payload: request }).pipe(Effect.mapError(mapMutationError))
      yield* response.events.pipe(
        Stream.runForEach((event) => event.type === "progress"
          ? updateInventory((state) => {
              if (state._tag !== "Ready") return state
              return LocalModelInventoryLifecycle.hold(state, {
                entries: state.entries.map((entry) => entry.model.localModelId !== localModelId ? entry
                  : entry._tag === "Downloading"
                    ? LocalModelInventoryEntryLifecycle.hold(entry, {
                        percentage: Math.max(entry.percentage, percentage(event.completed_bytes, event.total_bytes)),
                        completedBytes: Math.max(entry.completedBytes, event.completed_bytes),
                        totalBytes: Math.max(entry.totalBytes, event.total_bytes, entry.completedBytes, event.completed_bytes),
                      })
                    : entry._tag === "AvailableForDownload" || entry._tag === "DownloadFailed"
                      ? LocalModelInventoryEntryLifecycle.transition(entry, "Downloading", {
                          percentage: percentage(event.completed_bytes, event.total_bytes),
                          completedBytes: event.completed_bytes,
                          totalBytes: event.total_bytes,
                        })
                      : entry),
              })
            }).pipe(Effect.asVoid)
          : Effect.void),
        Effect.mapError(mapMutationError),
      )
      yield* inventory.refresh.pipe(Effect.mapError(mapMutationError))
      yield* rebuildInventory
    }).pipe(Effect.tapError((error) => updateInventory((state) => state._tag !== "Ready" ? state
      : LocalModelInventoryLifecycle.hold(state, {
          entries: state.entries.map((entry) => entry.model.localModelId !== localModelId || entry._tag !== "Downloading"
            ? entry
            : LocalModelInventoryEntryLifecycle.transition(entry, "DownloadFailed", {
                completedBytes: entry.completedBytes,
                totalBytes: entry.totalBytes,
                error: { code: error._tag, message: error.message, retryable: "retryable" in error ? error.retryable : true },
              })),
        })).pipe(Effect.asVoid))),
    delete: (localModelId) => Effect.gen(function* () {
      const association = yield* associations.pipe(
        Effect.map((values) => values.find((value) => value.localModelId === localModelId)),
      )
      if (!association?.nativeModel || !association.nativeModelId) return
      yield* client.models.deleteModel({
        path: { model_id: association.nativeModelId },
        urlParams: { dry_run: Option.none() },
      }).pipe(Effect.mapError(mapMutationError))
      yield* inventory.refresh.pipe(Effect.mapError(mapMutationError))
      yield* rebuildInventory
    }),
  })
}))
