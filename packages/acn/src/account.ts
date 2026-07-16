import { FetchHttpClient } from "@effect/platform"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import { Cause, Chunk, Context, Effect, Layer, Option, Ref, Schema, Scope, Stream } from "effect"
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
  type ProviderId,
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
  type ModelSummary,
  type ModelCatalog,
  type ModelSlots,
  type MirroredResourceInvalidation,
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
  ProviderCatalogUnavailable,
} from "@magnitudedev/protocol"
import { SessionStore } from "./session-store"
import { LocalModelProviderSource } from "./local-inference/provider-source"
import { LocalModelConfiguration } from "./local-inference/model-configuration"
import { foldProviderCatalogOutcomes, type FoldedProviderCatalogs } from "./model-catalog-snapshot"
import { makeMirroredResource } from "./mirrored-resource"

import {
  SLOT_IDS,
  DEFAULT_REASONING_EFFORT,
  resolveSlotModel,
} from "@magnitudedev/roles"

export interface AccountApi {
  readonly updateProviderAuth: (providerId: ProviderId, auth: ProviderAuth) => Effect.Effect<void, SessionError>
  readonly getProviderAuth: (providerId: ProviderId) => Effect.Effect<ProviderAuth | null, SessionError>
  readonly listProviderAuth: Effect.Effect<Readonly<Record<string, ProviderAuth>>, SessionError>
  readonly listPublicSlotProfiles: Effect.Effect<SlotProfiles | null, SessionError>
  readonly modelCatalog: Effect.Effect<ModelCatalog>
  readonly watchModelCatalog: Stream.Stream<MirroredResourceInvalidation>
  readonly refreshModelCatalog: (providerId: Option.Option<ProviderId>) => Effect.Effect<void>
  readonly modelSlots: Effect.Effect<ModelSlots>
  readonly watchModelSlots: Stream.Stream<MirroredResourceInvalidation>
  readonly updateModelSlots: (slots: Partial<Record<SlotId, SlotModelConfig>>) => Effect.Effect<void, SessionError>
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
    capabilities: { ...(model.capabilities.vision === undefined ? {} : { vision: model.capabilities.vision }) },
    availability: model.availability,
    reasoningEfforts: [...model.reasoningEfforts],
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
  readonly slotProfiles: SlotProfiles
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
      slotProfiles: slotProfilesFromModels(models, modelConfig),
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

/**
 * Build slot profiles from catalog models + user config.
 * Uses `resolveSlotModel`.
 */
function slotProfilesFromModels(
  models: readonly ProviderModel[],
  userConfig: {
    readonly slots?: Partial<Record<SlotId, { readonly providerId?: string; readonly providerModelId?: string; readonly reasoningEffort?: string }>>
    readonly localSlotIntent?: Partial<Record<SlotId, "local" | "cloud">>
  } | null,
): SlotProfiles {
  const routableModels = models.map((model) => ({
    ...model,
    slots: slotsForModel(model),
  }))
  let profiles: SlotProfiles = {}
  for (const slotId of SLOT_IDS) {
    const userSlotConfig = userConfig?.slots?.[slotId]
    const intent = userConfig?.localSlotIntent?.[slotId]
    const eligibleModels = intent === "local"
      ? routableModels.filter((model) => model.providerId === "llamacpp")
      : intent === "cloud"
        ? routableModels.filter((model) => model.providerId !== "llamacpp")
        : routableModels
    const resolved = resolveSlotModel(eligibleModels, userSlotConfig, slotId)
    if (!resolved) continue
    const model = routableModels.find((candidate) =>
      candidate.providerId === resolved.providerId
      && candidate.providerModelId === resolved.providerModelId)
    if (!model) continue

    const requestedEffort = userSlotConfig?.reasoningEffort ?? DEFAULT_REASONING_EFFORT[slotId]
    const reasoningEffort = model.reasoningEfforts.includes(requestedEffort)
      ? requestedEffort
      : model.reasoningEfforts[0] ?? "none"

    profiles = {
      ...profiles,
      [slotId]: {
        slotId,
        providerId: ProviderIdSchema.make(resolved.providerId),
        providerModelId: ProviderModelIdSchema.make(resolved.providerModelId),
        modelDisplayName: model.displayName,
        contextWindow: model.contextWindow,
        maxOutputTokens: model.maxOutputTokens,
        capabilities: { vision: model.capabilities.vision ?? false },
        reasoningEffort,
        isUserOverride: resolved.isUserOverride,
      },
    }
  }
  return profiles
}

// =============================================================================
// Layer
// =============================================================================

export const AccountLive: Layer.Layer<Account, never, SessionStore | ProviderClient | MagnitudeStorage | ProviderClientRegistry | LocalModelProviderSource | LocalModelConfiguration> =
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
      const catalogResource = yield* makeMirroredResource<ModelCatalogState>(new ModelCatalogLoading({}))
      const slotResource = yield* makeMirroredResource<ModelSlotsState>(new ModelSlotsLoading({}))
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
        yield* catalogResource.update((state) => ModelCatalogLifecycle.match(state, {
          loading: (current) => current,
          ready: (current) => ModelCatalogLifecycle.transition(current, "refreshing", { failures: [] }),
          refreshing: (current) => current,
          degraded: (current) => ModelCatalogLifecycle.transition(current, "refreshing", {}),
          unavailable: (current) => ModelCatalogLifecycle.transition(current, "refreshing", { models: [] }),
        }))
        yield* slotResource.update((state) => ModelSlotsLifecycle.match(state, {
          loading: (current) => current,
          ready: (current) => ModelSlotsLifecycle.transition(current, "refreshing", { failures: [] }),
          refreshing: (current) => current,
          degraded: (current) => ModelSlotsLifecycle.transition(current, "refreshing", {}),
          unavailable: (current) => ModelSlotsLifecycle.transition(current, "refreshing", { profiles: {} }),
        }))
      })

      const publishCatalog = (list: BuiltCatalogState, failures: readonly ProviderCatalogFailure[]) =>
        catalogResource.update((state) => {
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
        slotResource.update((state) => {
          if (failures.length === 0) {
            if (ModelSlotsLifecycle.is(state, "loading") || ModelSlotsLifecycle.is(state, "refreshing")) {
              return ModelSlotsLifecycle.transition(state, "ready", { profiles: list.slotProfiles, config: list.modelConfig })
            }
            return invalidSlotTransition(state, "publish ready")
          }
          if (Object.keys(list.slotProfiles).length > 0) {
            if (ModelSlotsLifecycle.is(state, "loading") || ModelSlotsLifecycle.is(state, "refreshing")) {
              return ModelSlotsLifecycle.transition(state, "degraded", { profiles: list.slotProfiles, config: list.modelConfig, failures })
            }
            return invalidSlotTransition(state, "publish degraded")
          }
          if (ModelSlotsLifecycle.is(state, "loading") || ModelSlotsLifecycle.is(state, "refreshing")) {
            return ModelSlotsLifecycle.transition(state, "unavailable", { config: list.modelConfig, failures })
          }
          return invalidSlotTransition(state, "publish unavailable")
        })

      const markSlotConfigurationFailed = (cause: unknown) => {
        const failure = new ModelSlotConfigurationUnavailable({
          message: cause instanceof Error ? cause.message : String(cause),
        })
        return slotResource.update((state) => {
          if (ModelSlotsLifecycle.is(state, "loading")) {
            return ModelSlotsLifecycle.transition(state, "unavailable", {
              config: { slots: {}, localSlotIntent: {} },
              failures: [failure],
            })
          }
          if (ModelSlotsLifecycle.is(state, "refreshing")) {
            const failures = [...state.failures, failure]
            return Object.keys(state.profiles).length > 0
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

      const rebuildSnapshots = (force: boolean, providerId: Option.Option<ProviderId> = Option.none()) => snapshotLock.withPermits(1)(
        Effect.gen(function* () {
          yield* markCatalogRefreshing
          const outcomes = yield* (force
            ? sharedClient.catalogs.refresh(Option.getOrUndefined(providerId))
            : sharedClient.catalogs.list).pipe(
            Effect.provide(FetchHttpClient.layer),
          )
          const folded = foldProviderCatalogOutcomes(yield* Ref.get(providerCatalogs), outcomes)
          yield* Ref.set(providerCatalogs, folded)
          const providers = yield* sharedClient.listProviders.pipe(
            Effect.provide(FetchHttpClient.layer),
          )
          yield* publishCatalog(buildCatalogState(folded.models, providers), folded.failures)
          const slots = yield* buildSlotState(folded.models, storage, "build model slots").pipe(
            Effect.tapError(markSlotConfigurationFailed),
          )
          yield* publishSlots(slots, folded.failures)
        }).pipe(Effect.catchAll((cause) => Effect.logWarning("Model resource refresh failed").pipe(
          Effect.annotateLogs({ operation: "rebuild-model-resources", cause: String(cause) }),
        ))),
      )

      const rebuildSlotSnapshot = snapshotLock.withPermits(1)(Effect.gen(function* () {
        yield* slotResource.update((state) => ModelSlotsLifecycle.match(state, {
          loading: (current) => current,
          ready: (current) => ModelSlotsLifecycle.transition(current, "refreshing", { failures: [] }),
          refreshing: (current) => current,
          degraded: (current) => ModelSlotsLifecycle.transition(current, "refreshing", {}),
          unavailable: (current) => ModelSlotsLifecycle.transition(current, "refreshing", { profiles: {} }),
        }))
        const models = yield* catalogModels
        const list = yield* buildSlotState(models, storage, "build model slots").pipe(
          Effect.tapError(markSlotConfigurationFailed),
        )
        const catalog = (yield* catalogResource.get).state
        const failures = ModelCatalogLifecycle.match(catalog, {
          loading: () => [] as readonly ProviderCatalogFailure[],
          ready: () => [] as readonly ProviderCatalogFailure[],
          refreshing: (state) => state.failures,
          degraded: (state) => state.failures,
          unavailable: (state) => state.failures,
        })
        yield* publishSlots(list, failures)
      }))

      const refreshInBackground = (force: boolean, providerId: Option.Option<ProviderId> = Option.none()) => Effect.forkIn(
        observeResourceDefects("rebuild-model-resources", rebuildSnapshots(force, providerId)),
        scope,
      ).pipe(Effect.asVoid)

      yield* refreshInBackground(false)
      yield* Effect.forkIn(localModels.changes.pipe(
        Stream.runForEach(() => observeResourceDefects("rebuild-model-resources", rebuildSnapshots(false))),
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
            return slotProfilesFromModels(models, null)
          }).pipe(
            Effect.provide(FetchHttpClient.layer),
            Effect.mapError(toAccountError("list public slot profiles")),
          ),

        modelCatalog: catalogResource.get,
        watchModelCatalog: catalogResource.changes,
        refreshModelCatalog: (providerId) => refreshInBackground(true, providerId),
        modelSlots: slotResource.get,
        watchModelSlots: slotResource.changes,
        updateModelSlots: (slots) => Effect.gen(function* () {
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
            if (!model || !slot.reasoningEffort || model.reasoningEfforts.includes(slot.reasoningEffort)) return [slotId, slot]
            return [slotId, { ...slot, reasoningEffort: model.reasoningEfforts[0] ?? "none" }]
          })) as typeof slots
          yield* modelConfiguration.updateSlots(normalized)
        }).pipe(
          Effect.mapError(toAccountError("update model slots")),
          Effect.andThen(rebuildSlotSnapshot),
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
