import { Effect } from "effect"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import type { MagnitudeModelInfo, ModelListResponse } from "./contract"
import type { RoleId } from "./contract"
import type { AuthApplicator } from "@magnitudedev/ai"

export interface ModelCatalog {
  /** Returns cached models if fresh, otherwise fetches. */
  readonly list: Effect.Effect<readonly MagnitudeModelInfo[], Error, HttpClient.HttpClient>
  /** Finds a model by ID. Fails if not found. */
  readonly get: (id: string) => Effect.Effect<MagnitudeModelInfo, Error, HttpClient.HttpClient>
  /** Finds the model backing a given role. Fails if no model has that role. */
  readonly getByRole: (role: RoleId) => Effect.Effect<MagnitudeModelInfo, Error, HttpClient.HttpClient>
  /** Forces a fresh fetch, replacing the cache. */
  readonly refresh: Effect.Effect<readonly MagnitudeModelInfo[], Error, HttpClient.HttpClient>
}

export interface ModelCatalogConfig {
  readonly endpoint: string
  readonly auth: AuthApplicator
  readonly ttlMs?: number
}

export function createModelCatalog(config: ModelCatalogConfig): ModelCatalog {
  const { endpoint, auth, ttlMs = 5 * 60 * 1000 } = config

  let cache: readonly MagnitudeModelInfo[] | null = null
  let fetchedAt = 0

  const fetchModels: Effect.Effect<readonly MagnitudeModelInfo[], Error, HttpClient.HttpClient> =
    Effect.gen(function* () {
      const client = yield* HttpClient.HttpClient

      const headers = new Headers()
      auth(headers)

      const headerRecord: Record<string, string> = {}
      headers.forEach((value, key) => {
        headerRecord[key] = value
      })

      const request = HttpClientRequest.get(`${endpoint}/models`).pipe(
        HttpClientRequest.setHeaders(headerRecord),
      )

      const response = yield* client.execute(request).pipe(
        Effect.mapError((err) => new Error(`Failed to fetch models: ${err.message}`)),
      )

      if (response.status < 200 || response.status >= 300) {
        const body = yield* response.text.pipe(Effect.orElseSucceed(() => ""))
        return yield* Effect.fail(
          new Error(`Failed to fetch models: HTTP ${response.status} — ${body}`),
        )
      }

      const text = yield* response.text.pipe(
        Effect.mapError((err) => new Error(`Failed to read models response: ${err}`)),
      )

      let parsed: ModelListResponse
      try {
        parsed = JSON.parse(text) as ModelListResponse
      } catch {
        return yield* Effect.fail(new Error(`Failed to parse models response: ${text.slice(0, 200)}`))
      }

      if (!parsed.data || !Array.isArray(parsed.data)) {
        return yield* Effect.fail(new Error(`Invalid models response: missing "data" array`))
      }

      return parsed.data
    })

  const list: ModelCatalog["list"] = Effect.gen(function* () {
    if (cache && Date.now() - fetchedAt < ttlMs) {
      return cache
    }
    const models = yield* fetchModels
    cache = models
    fetchedAt = Date.now()
    return models
  })

  const get: ModelCatalog["get"] = (id) =>
    Effect.gen(function* () {
      const models = yield* list
      const model = models.find((m) => m.id === id)
      if (!model) {
        return yield* Effect.fail(new Error(`Model not found: ${id}`))
      }
      return model
    })

  const getByRole: ModelCatalog["getByRole"] = (role) =>
    Effect.gen(function* () {
      const models = yield* list
      const model = models.find((m) => m.role === role)
      if (!model) {
        return yield* Effect.fail(new Error(`No model found for role: ${role}`))
      }
      return model
    })

  const refresh: ModelCatalog["refresh"] = Effect.gen(function* () {
    const models = yield* fetchModels
    cache = models
    fetchedAt = Date.now()
    return models
  })

  return { list, get, getByRole, refresh }
}
