import { Effect, Option, Schema } from "effect"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import { MagnitudeModelListResponseSchema, type MagnitudeModelInfo, type MagnitudeRawModel } from "./contract"
import { AVAILABLE_PROVIDER_MODEL, ModelCatalogError, ModelFamilyIdSchema, ProviderIdSchema, ReasoningEffortSchema, ReasoningProperty, VisionProperty, type AuthApplicator, type ModelCatalog, type ModelFamilyId } from "@magnitudedev/ai"

type MagnitudeModelWithoutFamily = Omit<MagnitudeModelInfo, "modelFamilyId">

const MAGNITUDE_REASONING_EFFORTS = ["none", "low", "medium", "high", "max"]
  .map((effort) => ReasoningEffortSchema.make(effort))
const MAGNITUDE_DEFAULT_REASONING_EFFORT = ReasoningEffortSchema.make("high")

/**
 * Map a raw Magnitude API model to a MagnitudeModelInfo (without modelFamilyId).
 * The API returns `id`; ProviderModel uses `providerModelId`.
 */
export function toMagnitudeModelInfo(raw: MagnitudeRawModel): MagnitudeModelWithoutFamily {
  return {
    providerModelId: raw.id,
    providerId: ProviderIdSchema.make("magnitude"),
    displayName: raw.displayName,
    contextWindow: raw.contextWindow,
    maxOutputTokens: raw.maxOutputTokens,
    defaultReasoningEffort: MAGNITUDE_DEFAULT_REASONING_EFFORT,
    properties: {
      vision: new VisionProperty.states.Resolved({ value: Option.match(raw.capabilities, {
        onNone: () => false,
        onSome: ({ vision }) => vision,
      }) }),
      reasoning: new ReasoningProperty.states.Resolved({ value: MAGNITUDE_REASONING_EFFORTS }),
    },
    servingCapabilities: {
      tools: true,
      structuredOutput: Option.match(raw.capabilities, {
        onNone: () => false,
        onSome: ({ structuredOutput }) => Option.getOrElse(structuredOutput, () => false),
      }),
    },
    availability: AVAILABLE_PROVIDER_MODEL,
    pricing: Option.getOrElse(raw.pricing, () => ({ input: 0, output: 0, cached_input: null })),
    object: raw.object,
    owned_by: raw.owned_by,
    roles: raw.roles,
    slots: raw.slots,
    ...Option.match(raw.type, { onNone: () => ({}), onSome: (type) => ({ type }) }),
  }
}

export type MagnitudeAuthentication =
  | {
      readonly _tag: "Configured"
      readonly apply: AuthApplicator
    }
  | {
      readonly _tag: "NotConfigured"
    }

export interface MagnitudeCatalogConfig {
  readonly endpoint: string
  readonly authentication: MagnitudeAuthentication
  readonly ttlMs?: number
  readonly classify: (model: MagnitudeModelWithoutFamily) => Option.Option<ModelFamilyId>
}

/**
 * Magnitude-specific model catalog implementation.
 *
 * Fetches models from the Magnitude API, classifies each model into a
 * family using the shared classifier, and filters out any model that
 * cannot be classified (per §4.3 — unidentified models are excluded).
 */
export function createMagnitudeCatalog(config: MagnitudeCatalogConfig): ModelCatalog<MagnitudeModelInfo> {
  const { endpoint, authentication, ttlMs = 5 * 60 * 1000, classify } = config

  let cache: readonly MagnitudeModelInfo[] | null = null
  let fetchedAt = 0

  const fetchModels: Effect.Effect<readonly MagnitudeModelInfo[], ModelCatalogError, HttpClient.HttpClient> =
    Effect.gen(function* () {
      if (authentication._tag === "NotConfigured") {
        return yield* new ModelCatalogError({ message: "Magnitude authentication is not configured" })
      }

      const client = yield* HttpClient.HttpClient
      const headers = new Headers()
      yield* Effect.sync(() => authentication.apply(headers))

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

      const rawModels = yield* Schema.decodeUnknown(MagnitudeModelListResponseSchema)(body).pipe(
        Effect.mapError((cause) => new ModelCatalogError({ message: "Invalid models response", cause })),
      )

      const classified: MagnitudeModelInfo[] = []
      for (const raw of rawModels.data) {
        const model = toMagnitudeModelInfo(raw)
        const familyOption = classify(model)
        if (Option.isNone(familyOption)) continue
        classified.push({ ...model, modelFamilyId: ModelFamilyIdSchema.make(familyOption.value) })
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
