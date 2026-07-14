import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import { Effect, Schema } from "effect"
import { ModelCatalogError } from "@magnitudedev/ai"

const ModelsDevReasoningOptionSchema = Schema.Struct({
  type: Schema.String,
  values: Schema.optional(Schema.Array(Schema.NullOr(Schema.String))),
})

const ModelsDevModelSchema = Schema.Struct({
  id: Schema.String,
  name: Schema.String,
  description: Schema.optional(Schema.String),
  family: Schema.optional(Schema.String),
  attachment: Schema.Boolean,
  reasoning: Schema.Boolean,
  reasoning_options: Schema.optional(Schema.Array(ModelsDevReasoningOptionSchema)),
  tool_call: Schema.Boolean,
  structured_output: Schema.optional(Schema.Boolean),
  temperature: Schema.optional(Schema.Boolean),
  open_weights: Schema.Boolean,
  modalities: Schema.Struct({
    input: Schema.Array(Schema.String),
    output: Schema.Array(Schema.String),
  }),
  limit: Schema.Struct({
    context: Schema.Number,
    output: Schema.Number,
  }),
  cost: Schema.optional(Schema.Struct({
    input: Schema.Number,
    output: Schema.Number,
    cache_read: Schema.optional(Schema.Number),
    cache_write: Schema.optional(Schema.Number),
  })),
})

const ModelsDevProviderSchema = Schema.Struct({
  id: Schema.String,
  name: Schema.String,
  api: Schema.optional(Schema.NullOr(Schema.String)),
  models: Schema.Record({ key: Schema.String, value: ModelsDevModelSchema }),
})

const ModelsDevSnapshotSchema = Schema.Record({
  key: Schema.String,
  value: ModelsDevProviderSchema,
})

export type ModelsDevModel = Schema.Schema.Type<typeof ModelsDevModelSchema>
export type ModelsDevProvider = Schema.Schema.Type<typeof ModelsDevProviderSchema>
export type ModelsDevSnapshot = Schema.Schema.Type<typeof ModelsDevSnapshotSchema>

export interface ModelsDevOverride {
  readonly providerId: string
  readonly modelId: string
  readonly patch: Partial<ModelsDevModel>
}

export interface ModelsDevClient {
  readonly getProvider: (
    providerId: string,
  ) => Effect.Effect<ModelsDevProvider | null, ModelCatalogError, HttpClient.HttpClient>
  readonly refresh: Effect.Effect<ModelsDevSnapshot, ModelCatalogError, HttpClient.HttpClient>
}

export interface ModelsDevClientConfig {
  readonly endpoint?: string
  readonly ttlMs?: number
  readonly overrides?: readonly ModelsDevOverride[]
}

function applyOverrides(
  snapshot: ModelsDevSnapshot,
  overrides: readonly ModelsDevOverride[],
): ModelsDevSnapshot {
  if (overrides.length === 0) return snapshot

  const providers: Record<string, ModelsDevProvider> = { ...snapshot }
  for (const override of overrides) {
    const provider = providers[override.providerId]
    const model = provider?.models[override.modelId]
    if (!provider || !model) continue
    providers[override.providerId] = {
      ...provider,
      models: {
        ...provider.models,
        [override.modelId]: { ...model, ...override.patch },
      },
    }
  }
  return providers
}

export function createModelsDevClient(
  config?: ModelsDevClientConfig,
): ModelsDevClient {
  const endpoint = config?.endpoint ?? "https://models.dev/api.json"
  const ttlMs = config?.ttlMs ?? 10 * 60 * 1000
  const overrides = config?.overrides ?? []
  let cached: ModelsDevSnapshot | null = null
  let fetchedAt = 0

  const fetchSnapshot: Effect.Effect<ModelsDevSnapshot, ModelCatalogError, HttpClient.HttpClient> =
    Effect.gen(function* () {
      const client = yield* HttpClient.HttpClient
      const response = yield* client.execute(HttpClientRequest.get(endpoint)).pipe(
        Effect.mapError((cause) => new ModelCatalogError({
          message: "Failed to fetch models.dev catalog",
          cause,
        })),
      )

      if (response.status < 200 || response.status >= 300) {
        const body = yield* response.text.pipe(Effect.orElseSucceed(() => ""))
        return yield* new ModelCatalogError({
          message: `Failed to fetch models.dev catalog: HTTP ${response.status} - ${body}`,
        })
      }

      const body = yield* response.json.pipe(
        Effect.mapError((cause) => new ModelCatalogError({
          message: "Failed to read models.dev catalog",
          cause,
        })),
      )
      const decoded = yield* Schema.decodeUnknown(ModelsDevSnapshotSchema)(body).pipe(
        Effect.mapError((cause) => new ModelCatalogError({
          message: "models.dev returned an invalid catalog",
          cause,
        })),
      )
      return applyOverrides(decoded, overrides)
    })

  const refresh: ModelsDevClient["refresh"] = fetchSnapshot.pipe(
    Effect.tap((snapshot) => Effect.sync(() => {
      cached = snapshot
      fetchedAt = Date.now()
    })),
    Effect.catchAll((error) => cached ? Effect.succeed(cached) : Effect.fail(error)),
  )

  const load = Effect.gen(function* () {
    if (cached && Date.now() - fetchedAt < ttlMs) return cached
    return yield* refresh
  })

  return {
    getProvider: (providerId) => load.pipe(
      Effect.map((snapshot) => snapshot[providerId] ?? null),
    ),
    refresh,
  }
}
