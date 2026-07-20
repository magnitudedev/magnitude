import { FetchHttpClient } from "@effect/platform"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import { Cause, Chunk, Context, Effect, Layer, Option, Ref, Schema, Scope, Stream, SubscriptionRef } from "effect"
import { buildConfigStateFromSlots, type ConfigState } from "@magnitudedev/agent"
import {
  ProviderClient,
  type UsageQuery,
  type CloudUsageResponse,
  type ProviderModel,
  type ProviderClientShape,
  type ProviderRegistryInfo,
  MagnitudeModelListResponseSchema,
  toMagnitudeModelInfo,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  type ReasoningEffort,
  type ProviderId,
  type ProviderModelId,
} from "@magnitudedev/sdk"
import { isEnvFlagOn } from "@magnitudedev/utils"
import { ProviderClientRegistry } from "./shared-client"
import {
  MagnitudeStorage,
  type MagnitudeStorageShape,
} from "@magnitudedev/storage"
import {
  SessionOperationFailed,
  type SessionError,
  type SlotProfiles,
  type SlotStates,
  type ModelSummary,
  type ModelCatalog,
  type ModelSlots,
  type ModelConfigResponse,
  type SlotModelConfig,
  type ProviderInfo,
  type ProviderAuth,
  type SlotId,
  ModelCatalogLifecycle,
  ModelCatalogLoading,
  type ModelCatalogState,
  ModelSlotsLifecycle,
  ModelSlotsLoading,
  ModelSlotConfigurationUnavailable,
  type ModelSlotsState,
  type ProviderCatalogFailure,
  type ModelSlotsFailure,
  ProviderCatalogStale,
  ProviderCatalogUnavailable,
  SlotUnassigned,
  SlotPending,
  SlotReady,
  SlotBlocked,
  ModelCatalogMirror,
  ModelSlotsMirror,
} from "@magnitudedev/protocol"
import { SessionStore } from "./session-store"
import { LocalModelProviderSource } from "./local-inference/provider-source"
import { LocalModelConfiguration } from "./local-inference/model-configuration"
import { foldProviderCatalogOutcomes, type FoldedProviderCatalogs } from "./model-catalog-snapshot"
import { makeMirroredState, MirroredStateChanges } from "./mirrored-state"

import {
  SLOT_IDS,
} from "@magnitudedev/roles"

export interface AccountApi {
  readonly updateProviderAuth: (providerId: ProviderId, auth: ProviderAuth) => Effect.Effect<void, SessionError>
  readonly getProviderAuth: (providerId: ProviderId) => Effect.Effect<ProviderAuth | null, SessionError>
  readonly listProviderAuth: Effect.Effect<Readonly<Record<string, ProviderAuth>>, SessionError>
  readonly listPublicSlotProfiles: Effect.Effect<SlotProfiles | null, SessionError>
  readonly modelCatalog: Effect.Effect<ModelCatalog>
  readonly refreshModelCatalog: (providerId: Option.Option<ProviderId>) => Effect.Effect<void>
  readonly modelSlots: Effect.Effect<ModelSlots>
  /** Replaying, coherent model configuration consumed by agent ambients. */
  readonly agentModelConfigurations: Stream.Stream<ConfigState>
  readonly updateModelSlots: (slots: Partial<Record<SlotId, SlotModelConfig>>) => Effect.Effect<void, SessionError>
  readonly applyReasoningEffortFallback: (input: {
    readonly slotId: SlotId
    readonly providerId: ProviderId
    readonly providerModelId: ProviderModelId
    readonly requested: ReasoningEffort
    readonly fallback: ReasoningEffort
  }) => Effect.Effect<void, SessionError>
  readonly getCloudUsage: (query?: UsageQuery) => Effect.Effect<CloudUsageResponse, SessionError>
}

export class Account extends Context.Tag("Account")<
  Account,
  AccountApi
>() {}

const toAccountError = (operation: string) => (cause: unknown): SessionError =>
  new SessionOperationFailed({
    operation,
    reason: Cause.pretty(Cause.fail(cause)),
  })

const noApiKey = (operation: string): SessionError =>
  new SessionOperationFailed({
    operation,
    reason: "No Magnitude API key found",
  })

// =============================================================================
// Pure mapping helpers
// =============================================================================

const isSlotId = (value: unknown): value is SlotId =>
  value === "primary" || value === "secondary"

const storedProviderId = (value: string | undefined) =>
  value !== undefined && Schema.is(ProviderIdSchema)(value) ? value : undefined

const storedProviderModelId = (value: string | undefined) =>
  value !== undefined && Schema.is(ProviderModelIdSchema)(value) ? value : undefined

const slotsForModel = (model: ProviderModel): readonly SlotId[] => {
  if (!("slots" in model) || !Array.isArray(model.slots)) return []
  return model.slots.filter(isSlotId)
}

/**
 * Map a `ProviderModel` → `ModelSummary` for the protocol layer.
 */
function toModelSummary(model: ProviderModel): ModelSummary {
  const slots = slotsForModel(model)
  return {
    providerId: model.providerId,
    providerModelId: model.providerModelId,
    ...(model.modelFamilyId ? { modelFamilyId: model.modelFamilyId } : {}),
    displayName: model.displayName,
    ...(slots.length > 0 ? { slots: [...slots] } : {}),
    contextWindow: model.contextWindow,
    maxOutputTokens: model.maxOutputTokens,
    defaultReasoningEffort: model.defaultReasoningEffort,
    properties: model.properties,
    availability: model.availability,
    pricing: model.pricing ? {
      input: model.pricing.input,
      output: model.pricing.output,
      ...(model.pricing.cached_input !== null ? { cachedInput: model.pricing.cached_input } : {}),
    } : undefined,
  }
}

interface BuiltCatalogState {
  readonly models: readonly ModelSummary[]
  readonly providers: readonly ProviderInfo[]
}

interface BuiltSlotState {
  readonly slots: SlotStates
  readonly modelConfig: ModelConfigResponse
}

function buildCatalogState(
  models: readonly ProviderModel[],
  providerInfos: readonly ProviderRegistryInfo[],
): BuiltCatalogState {
  return {
    models: models.map(toModelSummary),
    providers: providerInfos.map((provider) => ({
      id: provider.id,
      displayName: provider.displayName,
      authStatus: provider.authStatus._tag,
      ...(provider.status ? { status: provider.status } : {}),
      ...(provider.message ? { message: provider.message } : {}),
      ...(provider.hint ? { hint: provider.hint } : {}),
    })),
  }
}

function buildSlotState(
  models: readonly ProviderModel[],
  storage: MagnitudeStorageShape,
  operation: string,
): Effect.Effect<BuiltSlotState, SessionError> {
  return Effect.gen(function* () {
    const modelConfig = yield* storage.config.getModelConfig().pipe(
      Effect.mapError(toAccountError(operation)),
    )
    return {
      slots: slotStatesFromModels(models, modelConfig),
      modelConfig: {
        slots: Object.fromEntries(Object.entries(modelConfig?.slots ?? {}).map(([slotId, slot]) => [slotId, {
          ...(storedProviderId(slot?.providerId) ? { providerId: storedProviderId(slot?.providerId)! } : {}),
          ...(storedProviderModelId(slot?.providerModelId) ? { providerModelId: storedProviderModelId(slot?.providerModelId)! } : {}),
          ...(slot?.reasoningEffort ? { reasoningEffort: slot.reasoningEffort } : {}),
        }])),
        localSlotIntent: modelConfig?.localSlotIntent ?? {},
      },
    }
  })
}

/** Build authoritative slot state from provider models and persisted intent. */
export function slotStatesFromModels(
  models: readonly ProviderModel[],
  userConfig: {
    readonly slots?: Partial<Record<SlotId, { readonly providerId?: string; readonly providerModelId?: string; readonly reasoningEffort?: string }>>
    readonly localSlotIntent?: Partial<Record<SlotId, "local" | "cloud">>
  } | null,
): SlotStates {
  const routableModels = models.map((model) => ({
    ...model,
    slots: slotsForModel(model),
  }))
  const states = {} as Record<SlotId, SlotStates[SlotId]>
  for (const slotId of SLOT_IDS) {
    const userSlotConfig = userConfig?.slots?.[slotId]
    const intent = userConfig?.localSlotIntent?.[slotId]
    const eligibleModels = intent === "local"
      ? routableModels.filter((model) => model.providerId === "local")
      : intent === "cloud"
        ? routableModels.filter((model) => model.providerId !== "local")
        : routableModels
    const hasOverride = userSlotConfig?.providerId !== undefined && userSlotConfig.providerModelId !== undefined
    const selected = hasOverride
      ? routableModels.find((candidate) => candidate.providerId === userSlotConfig.providerId && candidate.providerModelId === userSlotConfig.providerModelId)
      : eligibleModels.find((model) => model.availability._tag === "Available" && model.slots.includes(slotId))
        ?? eligibleModels.find((model) => model.availability._tag === "Available")
    if (!selected) {
      states[slotId] = new SlotUnassigned({ slotId, reason: models.length === 0 ? "provider_unavailable" : "no_candidate" })
      continue
    }
    const requestedEffort = userSlotConfig?.reasoningEffort
      ? ReasoningEffortSchema.make(userSlotConfig.reasoningEffort)
      : selected.defaultReasoningEffort
    const reasoning = selected.properties.reasoning
    const hasKnownEfforts = reasoning._tag === "Cached" || reasoning._tag === "Resolved" || reasoning._tag === "Refreshing"
    const knownEfforts = hasKnownEfforts
      ? reasoning.value
      : []
    const waitingForReasoning = requestedEffort !== selected.defaultReasoningEffort && !hasKnownEfforts
    const reasoningEffort = requestedEffort === selected.defaultReasoningEffort
      || knownEfforts.includes(requestedEffort)
      || waitingForReasoning
      ? requestedEffort
      : selected.defaultReasoningEffort
    const selection = {
      providerId: selected.providerId,
      providerModelId: selected.providerModelId,
      reasoningEffort,
    }
    const source = hasOverride ? "user" as const : "automatic" as const
    if (selected.availability._tag === "Disabled") {
      const reason = selected.availability.reason === "model_unavailable"
        ? "model_unavailable" as const
        : selected.availability.reason === "installation_unavailable"
          ? "installation_unavailable" as const
        : selected.availability.reason === "incompatible_runtime"
          ? "incompatible_runtime" as const
          : "invalid_configuration" as const
      states[slotId] = new SlotBlocked({ slotId, selection, source, reason })
      continue
    }
    if (requestedEffort !== selected.defaultReasoningEffort && reasoning._tag === "Failed") {
      states[slotId] = new SlotBlocked({ slotId, selection, source, reason: "property_discovery_failed" })
      continue
    }
    if (waitingForReasoning) {
      states[slotId] = new SlotPending({ slotId, selection, source, waitingFor: ["reasoning"] })
      continue
    }
    states[slotId] = new SlotReady({
      slotId,
      selection,
      source,
      modelDisplayName: selected.displayName,
      contextWindow: selected.contextWindow,
      maxOutputTokens: selected.maxOutputTokens,
    })
  }
  return states as SlotStates
}

function compatibilityProfiles(states: SlotStates): SlotProfiles {
  let profiles: SlotProfiles = {}
  for (const slotId of SLOT_IDS) {
    const state = states[slotId]
    if (state._tag !== "Ready") continue
    profiles = { ...profiles, [slotId]: {
        slotId,
        providerId: state.selection.providerId,
        providerModelId: state.selection.providerModelId,
        modelDisplayName: state.modelDisplayName,
        contextWindow: state.contextWindow,
        maxOutputTokens: state.maxOutputTokens,
        reasoningEffort: state.selection.reasoningEffort,
        isUserOverride: state.source === "user",
      } }
  }
  return profiles
}

// =============================================================================
// Layer
// =============================================================================

export const AccountLive: Layer.Layer<Account, never, SessionStore | ProviderClient | MagnitudeStorage | ProviderClientRegistry | LocalModelProviderSource | LocalModelConfiguration | MirroredStateChanges> =
  Layer.scoped(
    Account,
    Effect.gen(function* () {
      const store = yield* SessionStore
      const sharedClient = yield* ProviderClient
      const storage = yield* MagnitudeStorage
      const providerClients = yield* ProviderClientRegistry
      const localModels = yield* LocalModelProviderSource
      const modelConfiguration = yield* LocalModelConfiguration
      const scope = yield* Scope.Scope
      const snapshotLock = yield* Effect.makeSemaphore(1)
      const slotMutationLock = yield* Effect.makeSemaphore(1)
      const initialCatalog: ModelCatalogState = new ModelCatalogLoading({})
      const initialSlots: ModelSlotsState = new ModelSlotsLoading({})
      const catalogMirror = yield* makeMirroredState(ModelCatalogMirror, initialCatalog)
      const slotMirror = yield* makeMirroredState(ModelSlotsMirror, initialSlots)
      const agentConfiguration = yield* SubscriptionRef.make<ConfigState>({
        revision: 0,
        bySlot: {
          primary: { _tag: "Unavailable", slotId: "primary", reason: "not_loaded" },
          secondary: { _tag: "Unavailable", slotId: "secondary", reason: "not_loaded" },
        },
        catalogLoaded: false,
      })
      const providerCatalogs = yield* Ref.make<Pick<FoldedProviderCatalogs, "byProvider" | "failuresByProvider">>({
        byProvider: new Map(),
        failuresByProvider: new Map(),
      })

      const catalogModels = Ref.get(providerCatalogs).pipe(
        Effect.map((catalogs) => [...catalogs.byProvider.values()].flat()),
      )

      const invalidCatalogTransition = (state: ModelCatalogState, operation: string): never => {
        throw new Error(`Invalid model catalog transition during ${operation}: ${state._tag}`)
      }
      const invalidSlotTransition = (state: ModelSlotsState, operation: string): never => {
        throw new Error(`Invalid model slots transition during ${operation}: ${state._tag}`)
      }

      const markCatalogRefreshing = Effect.gen(function* () {
        yield* catalogMirror.update((state) => ModelCatalogLifecycle.match(state, {
          loading: (current) => current,
          ready: (current) => ModelCatalogLifecycle.transition(current, "refreshing", { failures: [] }),
          refreshing: (current) => current,
          degraded: (current) => ModelCatalogLifecycle.transition(current, "refreshing", {}),
          unavailable: (current) => ModelCatalogLifecycle.transition(current, "refreshing", { models: [] }),
        }))
        yield* slotMirror.update((state) => ModelSlotsLifecycle.match(state, {
          loading: (current) => current,
          ready: (current) => ModelSlotsLifecycle.transition(current, "refreshing", { failures: [] }),
          refreshing: (current) => current,
          degraded: (current) => ModelSlotsLifecycle.transition(current, "refreshing", {}),
          unavailable: (current) => ModelSlotsLifecycle.transition(current, "refreshing", {}),
        }))
      })

      const publishCatalog = (list: BuiltCatalogState, failures: readonly ProviderCatalogFailure[]) =>
        catalogMirror.update((state) => {
          if (failures.length === 0) {
            if (ModelCatalogLifecycle.is(state, "loading") || ModelCatalogLifecycle.is(state, "refreshing")) {
              return ModelCatalogLifecycle.transition(state, "ready", list)
            }
            return invalidCatalogTransition(state, "publish ready")
          }
          if (list.models.length > 0) {
            if (ModelCatalogLifecycle.is(state, "loading") || ModelCatalogLifecycle.is(state, "refreshing")) {
              return ModelCatalogLifecycle.transition(state, "degraded", { ...list, failures })
            }
            return invalidCatalogTransition(state, "publish degraded")
          }
          const unavailable = failures.filter((failure): failure is ProviderCatalogUnavailable => failure._tag === "unavailable")
          if (ModelCatalogLifecycle.is(state, "loading") || ModelCatalogLifecycle.is(state, "refreshing")) {
            return ModelCatalogLifecycle.transition(state, "unavailable", { providers: list.providers, failures: unavailable })
          }
          return invalidCatalogTransition(state, "publish unavailable")
        })

      const publishSlots = (list: BuiltSlotState, failures: readonly ModelSlotsFailure[]) =>
        slotMirror.update((state) => {
          if (failures.length === 0) {
            if (ModelSlotsLifecycle.is(state, "loading") || ModelSlotsLifecycle.is(state, "refreshing")) {
              return ModelSlotsLifecycle.transition(state, "ready", { slots: list.slots, config: list.modelConfig })
            }
            return invalidSlotTransition(state, "publish ready")
          }
          if (Object.values(list.slots).some((slot) => slot._tag === "Ready")) {
            if (ModelSlotsLifecycle.is(state, "loading") || ModelSlotsLifecycle.is(state, "refreshing")) {
              return ModelSlotsLifecycle.transition(state, "degraded", { slots: list.slots, config: list.modelConfig, failures })
            }
            return invalidSlotTransition(state, "publish degraded")
          }
          if (ModelSlotsLifecycle.is(state, "loading") || ModelSlotsLifecycle.is(state, "refreshing")) {
            return ModelSlotsLifecycle.transition(state, "unavailable", { slots: list.slots, config: list.modelConfig, failures })
          }
          return invalidSlotTransition(state, "publish unavailable")
        })

      const publishAgentConfiguration = (
        models: readonly ModelSummary[],
        slots: SlotStates,
      ) => Effect.gen(function* () {
        const policy = yield* storage.config.getContextLimitPolicy()
        const current = yield* SubscriptionRef.get(agentConfiguration)
        const candidate = buildConfigStateFromSlots(models, slots, policy, current.revision)
        const unchanged = current.catalogLoaded === candidate.catalogLoaded
          && SLOT_IDS.every((slotId) => JSON.stringify(current.bySlot[slotId]) === JSON.stringify(candidate.bySlot[slotId]))
        if (!unchanged) {
          yield* SubscriptionRef.set(agentConfiguration, {
            ...candidate,
            revision: current.revision + 1,
          })
        }
      })

      const markSlotConfigurationFailed = (cause: unknown) => {
        const failure = new ModelSlotConfigurationUnavailable({
          message: cause instanceof Error ? cause.message : String(cause),
        })
        return slotMirror.update((state) => {
          if (ModelSlotsLifecycle.is(state, "loading")) {
            return ModelSlotsLifecycle.transition(state, "unavailable", {
              config: { slots: {}, localSlotIntent: {} },
              slots: slotStatesFromModels([], null),
              failures: [failure],
            })
          }
          if (ModelSlotsLifecycle.is(state, "refreshing")) {
            const failures = [...state.failures, failure]
            return Object.values(state.slots).some((slot) => slot._tag === "Ready")
              ? ModelSlotsLifecycle.transition(state, "degraded", { failures })
              : ModelSlotsLifecycle.transition(state, "unavailable", { failures })
          }
          return invalidSlotTransition(state, "publish configuration failure")
        })
      }

      const observeResourceDefects = <A, E, R>(operation: string, effect: Effect.Effect<A, E, R>) => effect.pipe(
        Effect.tapErrorCause((cause) => Chunk.isEmpty(Cause.defects(cause))
          ? Effect.void
          : Effect.logFatal("Model resource defect").pipe(
            Effect.annotateLogs({ operation, defect: Cause.pretty(cause) }),
          )),
      )

      const resolveApiKey = Effect.gen(function* () {
        const envKey = process.env.MAGNITUDE_API_KEY
        if (envKey?.trim()) return envKey

        const stored = yield* storage.auth.get("magnitude").pipe(
          Effect.catchAll(() => Effect.void),
        )
        if (stored?.type === "api" && stored.key.trim()) return stored.key

        return null
      })

      const magnitudeAuthenticatedClient = (operation: string) =>
        Effect.gen(function* () {
          const apiKey = yield* resolveApiKey
          if (!apiKey) return yield* noApiKey(operation)
          return sharedClient
        })

      const persistRemovedEffortFallbacks = (models: readonly ProviderModel[]) => Effect.gen(function* () {
        const configured = yield* modelConfiguration.getModels
        const updates: Partial<Record<SlotId, SlotModelConfig>> = {}
        for (const slotId of SLOT_IDS) {
          const slot = configured?.slots?.[slotId]
          if (!slot?.providerId || !slot.providerModelId || !slot.reasoningEffort) continue
          const model = models.find((candidate) => candidate.providerId === slot.providerId
            && candidate.providerModelId === slot.providerModelId)
          if (!model || model.properties.reasoning._tag !== "Resolved") continue
          if (slot.reasoningEffort === model.defaultReasoningEffort
            || model.properties.reasoning.value.includes(slot.reasoningEffort)) continue
          updates[slotId] = {
            providerId: model.providerId,
            providerModelId: model.providerModelId,
            reasoningEffort: model.defaultReasoningEffort,
          }
        }
        if (Object.keys(updates).length > 0) yield* modelConfiguration.updateSlots(updates)
      })

      const discoverSelectedProperties = (
        models: readonly ProviderModel[],
        slots: SlotStates,
        retryFailed: boolean,
      ) => Effect.forEach(
        [...new Map(Object.values(slots).flatMap((slot) => slot._tag === "Unassigned"
          ? []
          : [[`${slot.selection.providerId}\0${slot.selection.providerModelId}`, slot.selection] as const])).values()],
        (selection) => {
          const model = models.find((candidate) => candidate.providerId === selection.providerId
            && candidate.providerModelId === selection.providerModelId)
          if (!model || model.availability._tag !== "Available") return Effect.void
          const properties = (["vision", "reasoning"] as const).filter((property) => {
            const state = model.properties[property]
            return state._tag === "Deferred" || retryFailed && state._tag === "Failed"
          })
          return properties.length === 0
            ? Effect.void
            : sharedClient.discoverModelProperties(selection.providerId, {
                providerModelId: selection.providerModelId,
                properties,
              }).pipe(Effect.ignore)
        },
        { discard: true },
      )

      const publishFoldedCatalog = (folded: FoldedProviderCatalogs) => Effect.gen(function* () {
        yield* Ref.set(providerCatalogs, folded)
        yield* persistRemovedEffortFallbacks(folded.models)
        const providers = yield* sharedClient.listProviders
        yield* publishCatalog(buildCatalogState(folded.models, providers), folded.failures)
        const slots = yield* buildSlotState(folded.models, storage, "build model slots").pipe(
          Effect.tapError(markSlotConfigurationFailed),
        )
        yield* publishSlots(slots, folded.failures)
        yield* publishAgentConfiguration(folded.models, slots.slots)
        yield* discoverSelectedProperties(folded.models, slots.slots, false)
      }).pipe(Effect.provide(FetchHttpClient.layer))

      const rebuildSnapshots = (force: boolean, providerId: Option.Option<ProviderId> = Option.none()) => snapshotLock.withPermits(1)(
        Effect.gen(function* () {
          yield* markCatalogRefreshing
          const outcomes = yield* (force
            ? sharedClient.catalogs.refresh(Option.getOrUndefined(providerId))
            : sharedClient.catalogs.list).pipe(
            Effect.provide(FetchHttpClient.layer),
          )
          const folded = foldProviderCatalogOutcomes(yield* Ref.get(providerCatalogs), outcomes)
          yield* publishFoldedCatalog(folded)
        }).pipe(Effect.catchAll((cause) => Effect.logWarning("Model resource refresh failed").pipe(
          Effect.annotateLogs({ operation: "rebuild-model-resources", cause: String(cause) }),
        ))),
      )

      const rebuildLocalSnapshots = snapshotLock.withPermits(1)(Effect.gen(function* () {
        yield* markCatalogRefreshing
        const local = yield* localModels.catalog.list.pipe(
          Effect.provide(FetchHttpClient.layer),
          Effect.either,
        )
        const current = yield* Ref.get(providerCatalogs)
        const providerId = ProviderIdSchema.make("local")
        const byProvider = new Map(current.byProvider)
        const failuresByProvider = new Map(current.failuresByProvider)
        if (local._tag === "Right") {
          byProvider.set(providerId, local.right)
          failuresByProvider.delete(providerId)
        } else {
          const retained = byProvider.get(providerId) ?? []
          const message = local.left.message
          failuresByProvider.set(providerId, retained.length > 0
            ? new ProviderCatalogStale({ providerId, message })
            : new ProviderCatalogUnavailable({ providerId, message }))
        }
        yield* publishFoldedCatalog({
          byProvider,
          failuresByProvider,
          models: [...byProvider.values()].flat(),
          failures: [...failuresByProvider.values()],
        })
      }).pipe(Effect.catchAll((cause) => Effect.logWarning("Local model resource refresh failed").pipe(
        Effect.annotateLogs({ operation: "rebuild-local-model-resources", cause: String(cause) }),
      ))))

      const rebuildSlotSnapshot = snapshotLock.withPermits(1)(Effect.gen(function* () {
        yield* slotMirror.update((state) => ModelSlotsLifecycle.match(state, {
          loading: (current) => current,
          ready: (current) => ModelSlotsLifecycle.transition(current, "refreshing", { failures: [] }),
          refreshing: (current) => current,
          degraded: (current) => ModelSlotsLifecycle.transition(current, "refreshing", {}),
          unavailable: (current) => ModelSlotsLifecycle.transition(current, "refreshing", {}),
        }))
        const models = yield* catalogModels
        const list = yield* buildSlotState(models, storage, "build model slots").pipe(
          Effect.tapError(markSlotConfigurationFailed),
        )
        const catalog = (yield* catalogMirror.get).state
        const failures = ModelCatalogLifecycle.match(catalog, {
          loading: () => [] as readonly ProviderCatalogFailure[],
          ready: () => [] as readonly ProviderCatalogFailure[],
          refreshing: (state) => state.failures,
          degraded: (state) => state.failures,
          unavailable: (state) => state.failures,
        })
        yield* publishSlots(list, failures)
        yield* publishAgentConfiguration(models, list.slots)
      }))

      const refreshInBackground = (force: boolean, providerId: Option.Option<ProviderId> = Option.none()) => Effect.forkIn(
        observeResourceDefects("rebuild-model-resources", rebuildSnapshots(force, providerId)),
        scope,
      ).pipe(Effect.asVoid)

      yield* refreshInBackground(false)
      yield* Effect.forkIn(localModels.changes.pipe(
        Stream.runForEach(() => observeResourceDefects("rebuild-local-model-resources", rebuildLocalSnapshots)),
      ), scope)
      yield* Effect.forkIn(modelConfiguration.changes.pipe(
        Stream.runForEach(() => observeResourceDefects("rebuild-model-slots", rebuildSlotSnapshot.pipe(
          Effect.catchAll((cause) => Effect.logWarning("Model slot snapshot rebuild failed").pipe(
            Effect.annotateLogs({ operation: "rebuild-model-slots", cause: String(cause) }),
          )),
        ))),
      ), scope)

      return {
        updateProviderAuth: (providerId, auth) =>
          Effect.gen(function* () {
            yield* storage.auth.set(providerId, auth).pipe(
              Effect.mapError(toAccountError("update provider auth")),
            )
            yield* providerClients.refreshAll
            yield* refreshInBackground(true, Option.some(providerId))
          }),

        getProviderAuth: (providerId) =>
          storage.auth.get(providerId).pipe(
            Effect.map((info) => info ?? null),
            Effect.mapError(toAccountError("get provider auth")),
          ),

        listProviderAuth: storage.auth.loadAll().pipe(
          Effect.mapError(toAccountError("list provider auth")),
        ),

        listPublicSlotProfiles: Effect.gen(function* () {
            const client = yield* HttpClient.HttpClient
            const useLocal = isEnvFlagOn(process.env.MAGNITUDE_USE_LOCAL)
            const baseUrl = process.env.MAGNITUDE_ENDPOINT
              ?? (useLocal ? "http://localhost:3000/api/v1" : "https://app.magnitude.dev/api/v1")
            const response = yield* client.execute(HttpClientRequest.get(`${baseUrl}/public/models`))
            if (response.status < 200 || response.status >= 300) return null
            const body = yield* response.json.pipe(
              Effect.flatMap(Schema.decodeUnknown(MagnitudeModelListResponseSchema)),
            )
            const models = body.data.map(toMagnitudeModelInfo)
            return compatibilityProfiles(slotStatesFromModels(models, null))
          }).pipe(
            Effect.provide(FetchHttpClient.layer),
            Effect.mapError(toAccountError("list public slot profiles")),
          ),

        modelCatalog: catalogMirror.get,
        refreshModelCatalog: (providerId) => refreshInBackground(true, providerId),
        modelSlots: slotMirror.get,
        agentModelConfigurations: agentConfiguration.changes.pipe(
          Stream.filter((state) => state.catalogLoaded),
        ),
        updateModelSlots: (slots) => slotMutationLock.withPermits(1)(Effect.gen(function* () {
          const models = yield* catalogModels
          for (const [slotId, slot] of Object.entries(slots)) {
            if (!slot?.providerId || !slot.providerModelId) continue
            const model = models.find((candidate) => candidate.providerId === slot.providerId && candidate.providerModelId === slot.providerModelId)
            if (!model || model.availability._tag !== "Available") {
              return yield* new SessionOperationFailed({
                operation: "update model slots",
                reason: `Model is not available for ${slotId}: ${slot.providerId}/${slot.providerModelId}`,
              })
            }
          }
          const normalized = Object.fromEntries(Object.entries(slots).map(([slotId, slot]) => {
            if (!slot?.providerId || !slot.providerModelId) return [slotId, slot]
            const model = models.find((candidate) => candidate.providerId === slot.providerId && candidate.providerModelId === slot.providerModelId)
            if (!model || !slot.reasoningEffort) return [slotId, slot]
            const reasoning = model.properties.reasoning
            const efforts = reasoning._tag === "Cached" || reasoning._tag === "Resolved" || reasoning._tag === "Refreshing"
              ? reasoning.value
              : []
            if (slot.reasoningEffort === model.defaultReasoningEffort || efforts.includes(slot.reasoningEffort)) return [slotId, slot]
            return [slotId, { ...slot, reasoningEffort: model.defaultReasoningEffort }]
          })) as typeof slots
          yield* modelConfiguration.updateSlots(normalized)
          yield* rebuildSlotSnapshot
          const currentSlots = (yield* slotMirror.get).state
          yield* Effect.forkIn(ModelSlotsLifecycle.match(currentSlots, {
            loading: () => Effect.void,
            ready: (state) => discoverSelectedProperties(models, state.slots, true),
            refreshing: (state) => discoverSelectedProperties(models, state.slots, true),
            degraded: (state) => discoverSelectedProperties(models, state.slots, true),
            unavailable: (state) => discoverSelectedProperties(models, state.slots, true),
          }), scope)
        })).pipe(
          Effect.mapError(toAccountError("update model slots")),
        ),

        applyReasoningEffortFallback: (input) => slotMutationLock.withPermits(1)(Effect.gen(function* () {
          const configured = yield* modelConfiguration.getModels
          const current = configured?.slots?.[input.slotId]
          if (!current
            || current.providerId !== input.providerId
            || current.providerModelId !== input.providerModelId
            || current.reasoningEffort !== input.requested) return
          yield* modelConfiguration.updateSlots({
            [input.slotId]: {
              ...current,
              reasoningEffort: input.fallback,
            },
          })
          yield* rebuildSlotSnapshot
        })).pipe(
          Effect.mapError(toAccountError("apply reasoning effort fallback")),
        ),

        getCloudUsage: (query) =>
          Effect.gen(function* () {
            const client: ProviderClientShape = yield* magnitudeAuthenticatedClient("get cloud usage")
            return yield* client.usage(query).pipe(
              Effect.provide(FetchHttpClient.layer),
              Effect.mapError(toAccountError("get cloud usage")),
            )
          }),
      }
    }),
  )
