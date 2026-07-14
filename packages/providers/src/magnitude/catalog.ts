import { Effect, Option } from "effect"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import type { MagnitudeModelInfo, MagnitudeRawModel, ModelListResponse } from "./contract"
import { ModelCatalogError, type ModelCatalog, type ModelCatalogConfig } from "@magnitudedev/ai"

type MagnitudeModelWithoutFamily = Omit<MagnitudeModelInfo, "modelFamilyId">

/**
 * Map a raw Magnitude API model to a MagnitudeModelInfo (without modelFamilyId).
 * The API returns `id`; ProviderModel uses `providerModelId`.
 */
function toMagnitudeModelInfo(raw: MagnitudeRawModel): MagnitudeModelWithoutFamily {
  return {
    providerModelId: raw.id,
    providerId: "magnitude",
    displayName: raw.displayName,
    contextWindow: raw.contextWindow,
    maxOutputTokens: raw.maxOutputTokens,
    capabilities: { vision: raw.capabilities?.vision ?? false },
    pricing: raw.pricing ?? { input: 0, output: 0, cached_input: null },
    reasoningEfforts: raw.reasoningEfforts ?? ["none"],
    object: raw.object,
    owned_by: raw.owned_by,
    roles: raw.roles,
    slots: raw.slots,
    ...(raw.type !== undefined ? { type: raw.type } : {}),
  }
}

export interface MagnitudeCatalogConfig extends ModelCatalogConfig {
  readonly classify: (model: MagnitudeModelWithoutFamily) => Option.Option<string>
}

/**
 * Magnitude-specific model catalog implementation.
 *
 * Fetches models from the Magnitude API, classifies each model into a
 * family using the shared classifier, and filters out any model that
 * cannot be classified (per §4.3 — unidentified models are excluded).
 */
export function createMagnitudeCatalog(config: MagnitudeCatalogConfig): ModelCatalog<MagnitudeModelInfo> {
  const { endpoint, auth, ttlMs = 5 * 60 * 1000, classify } = config

  let cache: readonly MagnitudeModelInfo[] | null = null
  let fetchedAt = 0

  const fetchModels: Effect.Effect<readonly MagnitudeModelInfo[], ModelCatalogError, HttpClient.HttpClient> =
    Effect.gen(function* () {
      const client = yield* HttpClient.HttpClient

      const headers = new Headers()
      yield* Effect.try({
        try: () => auth(headers),
        catch: (cause) => new ModelCatalogError({ message: "Magnitude authentication is not configured", cause }),
      })

      const headerRecord: Record<string, string> = {}
      headers.forEach((value, key) => {
        headerRecord[key] = value
      })

      const request = HttpClientRequest.get(`${endpoint}/models`).pipe(
        HttpClientRequest.setHeaders(headerRecord),
      )

      const response = yield* client.execute(request).pipe(
        Effect.mapError((cause) => new ModelCatalogError({ message: "Failed to fetch models", cause })),
      )

      if (response.status < 200 || response.status >= 300) {
        const body = yield* response.text.pipe(Effect.orElseSucceed(() => ""))
        return yield* new ModelCatalogError({ message: `Failed to fetch models: HTTP ${response.status} - ${body}` })
      }

      const body = yield* response.json.pipe(
        Effect.mapError((cause) => new ModelCatalogError({ message: "Failed to read models response", cause })),
      )

      const rawModels = (body as ModelListResponse).data

      const classified: MagnitudeModelInfo[] = []
      for (const raw of rawModels) {
        const model = toMagnitudeModelInfo(raw)
        const familyOption = classify(model)
        if (Option.isNone(familyOption)) continue
        classified.push({ ...model, modelFamilyId: familyOption.value })
      }

      return classified
    })

  const list: ModelCatalog<MagnitudeModelInfo>["list"] = Effect.gen(function* () {
    if (cache && Date.now() - fetchedAt < ttlMs) {
      return cache
    }
    const models = yield* fetchModels
    cache = models
    fetchedAt = Date.now()
    return models
  })

  const get: ModelCatalog<MagnitudeModelInfo>["get"] = (_providerId, providerModelId) =>
    Effect.gen(function* () {
      const models = yield* list
      const model = models.find((m) => m.providerModelId === providerModelId)
      if (!model) {
        return yield* new ModelCatalogError({ message: `Model not found: ${providerModelId}` })
      }
      return model
    })

  const refresh: ModelCatalog<MagnitudeModelInfo>["refresh"] = Effect.gen(function* () {
    const models = yield* fetchModels
    cache = models
    fetchedAt = Date.now()
    return models
  })

  return { list, get, refresh }
}
