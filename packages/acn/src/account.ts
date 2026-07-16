import { isDeepStrictEqual } from "node:util"
import { FetchHttpClient } from "@effect/platform"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import { Cause, Context, Effect, Layer, Option, PubSub, Ref, Schema, Scope, Stream } from "effect"
import {
  ProviderClient,
  type BalanceQuery,
  type BalanceResponse,
  type ProviderModel,
  type ProviderClientShape,
  type ProviderRegistryInfo,
  MagnitudeModelListResponseSchema,
  toMagnitudeModelInfo,
  ProviderIdSchema,
  ProviderModelIdSchema,
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
  type ModelResourceInvalidation,
  type ModelConfigResponse,
  type SlotModelConfig,
  type ProviderInfo,
  type ProviderAuth,
  type SlotId,
} from "@magnitudedev/protocol"
import { SessionStore } from "./session-store"
import { LocalModelProviderSource } from "./local-inference/provider-source"
import { LocalModelConfiguration } from "./local-inference/model-configuration"

import {
  SLOT_IDS,
  DEFAULT_REASONING_EFFORT,
  resolveSlotModel,
} from "@magnitudedev/roles"

export interface AccountApi {
  readonly updateProviderAuth: (providerId: string, auth: ProviderAuth) => Effect.Effect<void, SessionError>
  readonly getProviderAuth: (providerId: string) => Effect.Effect<ProviderAuth | null, SessionError>
  readonly listProviderAuth: Effect.Effect<Readonly<Record<string, ProviderAuth>>, SessionError>
  readonly listPublicSlotProfiles: Effect.Effect<SlotProfiles | null, SessionError>
  readonly modelCatalog: Effect.Effect<ModelCatalog>
  readonly watchModelCatalog: Stream.Stream<ModelResourceInvalidation>
  readonly refreshModelCatalog: (providerId: Option.Option<string>) => Effect.Effect<void>
  readonly modelSlots: Effect.Effect<ModelSlots>
  readonly watchModelSlots: Stream.Stream<ModelResourceInvalidation>
  readonly updateModelSlots: (slots: Partial<Record<SlotId, SlotModelConfig>>) => Effect.Effect<void, SessionError>
  readonly getBalance: (query?: BalanceQuery) => Effect.Effect<BalanceResponse, SessionError>
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

/**
 * Build both independent projections from one consistent provider observation.
 */
interface BuiltModelState {
  readonly models: readonly ModelSummary[]
  readonly providers: readonly ProviderInfo[]
  readonly slotProfiles: SlotProfiles
  readonly modelConfig: ModelConfigResponse
}

function buildModelList(
  models: readonly ProviderModel[],
  providerInfos: readonly ProviderRegistryInfo[],
  storage: MagnitudeStorageShape,
  operation: string,
): Effect.Effect<BuiltModelState, SessionError> {
  return Effect.gen(function* () {
    const modelConfig = yield* storage.config.getModelConfig().pipe(
      Effect.mapError(toAccountError(operation)),
    )
    const providers: ProviderInfo[] = providerInfos.map((provider) => ({
      id: provider.id,
      displayName: provider.displayName,
      authStatus: provider.authStatus._tag,
      ...(provider.status ? { status: provider.status } : {}),
      ...(provider.message ? { message: provider.message } : {}),
      ...(provider.hint ? { hint: provider.hint } : {}),
    }))
    return {
      models: models.map(toModelSummary),
      providers,
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
      const catalogChanges = yield* PubSub.unbounded<ModelResourceInvalidation>()
      const slotChanges = yield* PubSub.unbounded<ModelResourceInvalidation>()
      const catalogState = yield* Ref.make<ModelCatalog>({ revision: 0, refreshing: true, models: [], providers: [] })
      const catalogModels = yield* Ref.make<readonly ProviderModel[]>([])
      const slotState = yield* Ref.make<ModelSlots>({ revision: 0, profiles: {}, config: { slots: {}, localSlotIntent: {} } })

      const publishSnapshots = (list: BuiltModelState, refreshing: boolean) => Effect.gen(function* () {
        const previousCatalog = yield* Ref.get(catalogState)
        const catalogChanged = previousCatalog.refreshing !== refreshing
          || !isDeepStrictEqual(previousCatalog.models, list.models)
          || !isDeepStrictEqual(previousCatalog.providers, list.providers)
        if (catalogChanged) {
          const catalog: ModelCatalog = {
            revision: previousCatalog.revision + 1,
            refreshing,
            models: list.models,
            providers: list.providers,
          }
          yield* Ref.set(catalogState, catalog)
          const catalogInvalidation: ModelResourceInvalidation = { _tag: "changed", revision: catalog.revision }
          yield* PubSub.publish(catalogChanges, catalogInvalidation)
        }
        const previousSlots = yield* Ref.get(slotState)
        const slotsChanged = !isDeepStrictEqual(previousSlots.profiles, list.slotProfiles)
          || !isDeepStrictEqual(previousSlots.config, list.modelConfig)
        if (slotsChanged) {
          const slots: ModelSlots = {
            revision: previousSlots.revision + 1,
            profiles: list.slotProfiles,
            config: list.modelConfig,
          }
          yield* Ref.set(slotState, slots)
          const slotInvalidation: ModelResourceInvalidation = { _tag: "changed", revision: slots.revision }
          yield* PubSub.publish(slotChanges, slotInvalidation)
        }
      })

      const markCatalogRefreshing = Effect.gen(function* () {
        const previous = yield* Ref.get(catalogState)
        if (previous.refreshing) return
        const next = { ...previous, revision: previous.revision + 1, refreshing: true }
        yield* Ref.set(catalogState, next)
        yield* PubSub.publish(catalogChanges, { _tag: "changed", revision: next.revision } satisfies ModelResourceInvalidation)
      })
      const markCatalogRefreshFailed = (cause: unknown) => Effect.gen(function* () {
        const previous = yield* Ref.get(catalogState)
        if (previous.refreshing) {
          const next = { ...previous, revision: previous.revision + 1, refreshing: false }
          yield* Ref.set(catalogState, next)
          yield* PubSub.publish(catalogChanges, { _tag: "changed", revision: next.revision } satisfies ModelResourceInvalidation)
        }
        yield* Effect.logWarning("Model catalog refresh failed").pipe(Effect.annotateLogs({ cause: String(cause) }))
      })

      const resolveApiKey = Effect.gen(function* () {
        const useLocal = isEnvFlagOn(process.env.MAGNITUDE_USE_LOCAL)
        if (useLocal) {
          const localKey = process.env.MAGNITUDE_LOCAL_API_KEY
          if (localKey?.trim()) return localKey
        }

        const stored = yield* storage.auth.get("magnitude").pipe(
          Effect.catchAll(() => Effect.void),
        )
        if (stored?.type === "api" && stored.key.trim()) return stored.key

        const envKey = process.env.MAGNITUDE_API_KEY
        if (envKey?.trim()) return envKey

        return null
      })

      const magnitudeAuthenticatedClient = (operation: string) =>
        Effect.gen(function* () {
          const apiKey = yield* resolveApiKey
          if (!apiKey) return yield* noApiKey(operation)
          return sharedClient
        })

      const rebuildSnapshots = (force: boolean, providerId: Option.Option<string> = Option.none()) => Effect.gen(function* () {
        if (force && Option.contains(providerId, "llamacpp")) {
          yield* localModels.catalog.refresh.pipe(
            Effect.provide(FetchHttpClient.layer),
            Effect.mapError(toAccountError("refresh local model catalog")),
          )
        }
        const refreshAllProviders = force && !Option.contains(providerId, "llamacpp")
        const models = yield* (refreshAllProviders ? sharedClient.catalog.refresh : sharedClient.catalog.list).pipe(
          Effect.provide(FetchHttpClient.layer),
          Effect.mapError(toAccountError(force ? "refresh model catalog" : "load model catalog")),
        )
        const providers = yield* sharedClient.listProviders.pipe(
          Effect.provide(FetchHttpClient.layer),
          Effect.mapError(toAccountError("inspect model providers")),
        )
        const list = yield* buildModelList(models, providers, storage, "build model state")
        yield* Ref.set(catalogModels, models)
        yield* publishSnapshots(list, false)
      })

      const refreshInBackground = (force: boolean, providerId: Option.Option<string> = Option.none()) => Effect.forkIn(
        rebuildSnapshots(force, providerId).pipe(
          Effect.catchAll(markCatalogRefreshFailed),
        ),
        scope,
      ).pipe(Effect.asVoid)

      yield* refreshInBackground(false)
      yield* Effect.forkIn(localModels.changes.pipe(
        Stream.runForEach(() => rebuildSnapshots(false).pipe(Effect.catchAll(() => Effect.void))),
      ), scope)

      return {
        updateProviderAuth: (providerId, auth) =>
          Effect.gen(function* () {
            yield* storage.auth.set(providerId, auth).pipe(
              Effect.mapError(toAccountError("update provider auth")),
            )
            yield* providerClients.refreshAll
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

        modelCatalog: Ref.get(catalogState),
        watchModelCatalog: Stream.fromPubSub(catalogChanges),
        refreshModelCatalog: (providerId) => markCatalogRefreshing.pipe(
          Effect.andThen(refreshInBackground(true, providerId)),
        ),
        modelSlots: Ref.get(slotState),
        watchModelSlots: Stream.fromPubSub(slotChanges),
        updateModelSlots: (slots) => Effect.gen(function* () {
          const models = yield* Ref.get(catalogModels)
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
          Effect.andThen(Effect.gen(function* () {
            const catalog = yield* Ref.get(catalogState)
            const models = yield* Ref.get(catalogModels)
            const providers = yield* sharedClient.listProviders.pipe(Effect.provide(FetchHttpClient.layer))
            const list = yield* buildModelList(models, providers, storage, "update model slots")
            yield* publishSnapshots(list, catalog.refreshing)
          }).pipe(Effect.mapError(toAccountError("update model slots")))),
        ),

        getBalance: (query) =>
          Effect.gen(function* () {
            const client: ProviderClientShape = yield* magnitudeAuthenticatedClient("get balance")
            return yield* client.balance(query).pipe(
              Effect.provide(FetchHttpClient.layer),
              Effect.mapError(toAccountError("get balance")),
            )
          }),
      }
    }),
  )
