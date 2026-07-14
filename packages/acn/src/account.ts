import { FetchHttpClient } from "@effect/platform"
import { Cause, Context, Effect, Layer, Ref } from "effect"
import {
  ProviderClient,
  type BalanceQuery,
  type BalanceResponse,
  type ProviderModel,
  type ProviderClientShape,
  type ProviderRegistryInfo,
} from "@magnitudedev/sdk"
import { isEnvFlagOn } from "@magnitudedev/utils"
import {
  SharedMagnitudeClientRef,
  makeClientWithAuth,
  resolveLlamaCppAuthFromStorage,
} from "./shared-client"
import {
  MagnitudeStorage,
  GlobalStorage,
  type MagnitudeStorageShape,
} from "@magnitudedev/storage"
import {
  SessionOperationFailed,
  type SessionError,
  type SlotProfiles,
  type SlotProfile,
  type SlotModelConfig,
  type ModelConfigResponse,
  type ModelSummary,
  type ModelList,
  type ProviderInfo,
  type ProviderAuth,
  type SlotId,
} from "@magnitudedev/protocol"
import { SessionStore } from "./session-store"

import {
  SLOT_IDS,
  DEFAULT_REASONING_EFFORT,
  resolveSlotModel,
  type UserSlotConfig,
} from "@magnitudedev/roles"

export interface AccountApi {
  readonly updateProviderAuth: (providerId: string, auth: ProviderAuth) => Effect.Effect<void, SessionError>
  readonly getProviderAuth: (providerId: string) => Effect.Effect<ProviderAuth | null, SessionError>
  readonly listProviderAuth: Effect.Effect<Readonly<Record<string, ProviderAuth>>, SessionError>
  readonly listPublicSlotProfiles: Effect.Effect<SlotProfiles | null, SessionError>
  readonly updateModelConfig: (slots: Partial<Record<SlotId, SlotModelConfig>>) => Effect.Effect<void, SessionError>
  readonly getCachedModelList: (providerId?: string) => Effect.Effect<ModelList, SessionError>
  readonly refreshCachedModelList: (providerId?: string) => Effect.Effect<ModelList, SessionError>
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

/**
 * Map a `ProviderModel` → `ModelSummary` for the protocol layer.
 */
function toModelSummary(model: ProviderModel): ModelSummary {
  const slots = (model as ProviderModel & { readonly slots?: readonly SlotId[] }).slots
  return {
    providerId: model.providerId,
    providerModelId: model.providerModelId,
    modelFamilyId: model.modelFamilyId,
    displayName: model.displayName,
    ...(slots && slots.length > 0 ? { slots: [...slots] } : {}),
    contextWindow: model.contextWindow,
    maxOutputTokens: model.maxOutputTokens,
    capabilities: { vision: model.capabilities.vision ?? false },
    reasoningEfforts: [...model.reasoningEfforts],
    pricing: model.pricing ? {
      input: model.pricing.input,
      output: model.pricing.output,
      ...(model.pricing.cached_input !== null ? { cachedInput: model.pricing.cached_input } : {}),
    } : undefined,
  }
}

/**
 * Build a unified `ModelList` response from catalog models + user config.
 */
function buildModelList(
  models: readonly ProviderModel[],
  providerInfos: readonly ProviderRegistryInfo[],
  storage: MagnitudeStorageShape,
  operation: string,
): Effect.Effect<ModelList, SessionError> {
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
      modelConfig: { slots: modelConfig?.slots ?? {} } as ModelConfigResponse,
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
  } | null,
): SlotProfiles {
  const out: Record<string, SlotProfile> = {}
  for (const slotId of SLOT_IDS) {
    const userSlotConfig = userConfig?.slots?.[slotId]

    const resolved = resolveSlotModel(models, userSlotConfig as UserSlotConfig | undefined, slotId)
    if (!resolved) continue
    const model = models.find((m) => m.providerId === resolved.providerId && m.providerModelId === resolved.providerModelId)
    if (!model) continue

    const reasoningEffort = userSlotConfig?.reasoningEffort ?? DEFAULT_REASONING_EFFORT[slotId]

    out[slotId] = {
      slotId,
      providerId: resolved.providerId,
      providerModelId: resolved.providerModelId,
      modelDisplayName: model.displayName,
      contextWindow: model.contextWindow,
      maxOutputTokens: model.maxOutputTokens,
      capabilities: { vision: model.capabilities.vision ?? false },
      reasoningEffort,
      isUserOverride: resolved.isUserOverride,
    }
  }
  return out as SlotProfiles
}

// =============================================================================
// Layer
// =============================================================================

export const AccountLive: Layer.Layer<Account, never, SessionStore | ProviderClient | MagnitudeStorage | SharedMagnitudeClientRef | GlobalStorage> =
  Layer.effect(
    Account,
    Effect.gen(function* () {
      const store = yield* SessionStore
      const sharedClient = yield* ProviderClient
      const storage = yield* MagnitudeStorage
      const clientRef = yield* SharedMagnitudeClientRef
      const globalStorage = yield* GlobalStorage

      // Semaphore to serialize config writes (§5.9)
      const configSemaphore = yield* Effect.makeSemaphore(1)

      const resolveApiKey = Effect.gen(function* () {
        const useLocal = isEnvFlagOn(process.env.MAGNITUDE_USE_LOCAL)
        if (useLocal) {
          const localKey = process.env.MAGNITUDE_LOCAL_API_KEY
          if (localKey?.trim()) return localKey
        }

        const stored = yield* storage.auth.get("magnitude").pipe(
          Effect.catchAll(() => Effect.succeed(undefined)),
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

      return {
        updateProviderAuth: (providerId, auth) =>
          Effect.gen(function* () {
            yield* storage.auth.set(providerId, auth).pipe(
              Effect.mapError(toAccountError("update provider auth")),
            )
            if (
              (providerId === "magnitude" && auth.type === "api")
              || (providerId === "llamacpp" && auth.type === "endpoint")
            ) {
              const apiKey = yield* resolveApiKey
              const llamacpp = yield* resolveLlamaCppAuthFromStorage(storage)
              const newClient = yield* makeClientWithAuth(apiKey, llamacpp).pipe(
                Effect.provideService(GlobalStorage, globalStorage),
              )
              yield* Ref.set(clientRef, newClient)
            }
          }),

        getProviderAuth: (providerId) =>
          storage.auth.get(providerId).pipe(
            Effect.map((info) => info ?? null),
            Effect.mapError(toAccountError("get provider auth")),
          ),

        listProviderAuth: storage.auth.loadAll().pipe(
          Effect.mapError(toAccountError("list provider auth")),
        ),

        listPublicSlotProfiles: Effect.tryPromise({
          try: async () => {
            // Fetch public model list without auth, then build slot profiles.
            const useLocal = isEnvFlagOn(process.env.MAGNITUDE_USE_LOCAL)
            const baseUrl = process.env.MAGNITUDE_ENDPOINT
              ?? (useLocal ? "http://localhost:3000/api/v1" : "https://app.magnitude.dev/api/v1")
            const response = await fetch(`${baseUrl}/public/models`)
            if (!response.ok) return null
            const body = await response.json() as { data: readonly ProviderModel[] }
            return slotProfilesFromModels(body.data, null)
          },
          catch: toAccountError("list public slot profiles"),
        }),

        updateModelConfig: (slots) =>
          configSemaphore.withPermits(1)(
            storage.config.updateModelConfig(slots).pipe(
              Effect.mapError(toAccountError("update model config")),
            ),
          ),

        getCachedModelList: (providerId?: string) =>
          Effect.gen(function* () {
            const models = yield* sharedClient.catalog.list.pipe(
              Effect.provide(FetchHttpClient.layer),
              Effect.mapError(toAccountError("get cached model list")),
            )
            const providers = yield* sharedClient.listProviders.pipe(
              Effect.provide(FetchHttpClient.layer),
            )
            return yield* buildModelList(models, providers, storage, "get cached model list")
          }),

        refreshCachedModelList: (providerId?: string) =>
          Effect.gen(function* () {
            const models = yield* sharedClient.catalog.refresh.pipe(
              Effect.provide(FetchHttpClient.layer),
              Effect.mapError(toAccountError("refresh cached model list")),
            )
            const providers = yield* sharedClient.listProviders.pipe(
              Effect.provide(FetchHttpClient.layer),
            )
            return yield* buildModelList(models, providers, storage, "refresh cached model list")
          }),

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
