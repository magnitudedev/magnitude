import { Effect } from "effect"
import type { ProviderModel } from "../../../lib/model/provider-model"
import { LocalDiscoveryError } from "./errors"
import type { LocalDiscoveryResult } from "./lmstudio"
import { discoverOpenAIModels } from "./openai-models"

interface LlamaCppPropsResponse {
  readonly default_generation_settings?: {
    readonly n_ctx?: number | string
  }
}

interface LlamaSlotsEntry {
  readonly n_ctx?: number | string
}

interface LlamaSlotsResponse {
  readonly slots?: readonly LlamaSlotsEntry[] | LlamaSlotsEntry
}

interface LlamaShowResponse {
  readonly model_info?: {
    readonly "llama.context_length"?: number | string
  }
}

interface LlamaV1MetaEntry {
  readonly id?: string
  readonly n_ctx_train?: number | string
  readonly meta?: {
    readonly n_ctx_train?: number | string
  }
}

interface LlamaV1MetaResponse {
  readonly data?: readonly LlamaV1MetaEntry[]
}

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, "")
}

function openAIModelsUrl(baseUrl: string): string {
  const normalized = normalizeBaseUrl(baseUrl)
  return normalized.endsWith("/v1") ? `${normalized}/models` : `${normalized}/v1/models`
}

function llamaCppPropsUrls(baseUrl: string, modelId: string): readonly string[] {
  const normalized = normalizeBaseUrl(baseUrl)
  return [`${normalized}/props?model=${encodeURIComponent(modelId)}`, `${normalized}/props`]
}

function llamaCppSlotsUrl(baseUrl: string): string {
  return `${normalizeBaseUrl(baseUrl)}/slots`
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} for ${url}`)
  }
  return (await response.json()) as T
}

function readNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value) && value > 0) {
    return value
  }
  if (typeof value === "string") {
    const parsed = Number(value)
    if (Number.isFinite(parsed) && parsed > 0) {
      return parsed
    }
  }
  return null
}

function parseLlamaCppContextLength(payload: unknown): number | null {
  if (!payload || typeof payload !== "object") {
    return null
  }
  return readNumber((payload as LlamaCppPropsResponse).default_generation_settings?.n_ctx)
}

function parseLlamaShowContextLength(payload: unknown): number | null {
  if (!payload || typeof payload !== "object") {
    return null
  }
  return readNumber((payload as LlamaShowResponse).model_info?.["llama.context_length"])
}

function parseLlamaSlotsContextLength(payload: unknown): number | null {
  const slots = Array.isArray(payload)
    ? payload
    : payload && typeof payload === "object"
      ? ((payload as LlamaSlotsResponse).slots ?? payload)
      : null

  if (Array.isArray(slots)) {
    for (const slot of slots) {
      const parsed = readNumber(slot?.n_ctx)
      if (parsed != null) {
        return parsed
      }
    }
    return null
  }

  if (slots && typeof slots === "object") {
    return readNumber((slots as LlamaSlotsEntry).n_ctx)
  }

  return null
}

function parseLlamaV1MetaContextMap(payload: unknown): ReadonlyMap<string, number> {
  const values = new Map<string, number>()
  if (!payload || typeof payload !== "object") {
    return values
  }

  for (const item of (payload as LlamaV1MetaResponse).data ?? []) {
    const id = typeof item.id === "string" && item.id.trim().length > 0 ? item.id.trim() : null
    if (!id) {
      continue
    }
    const value = readNumber(item.meta?.n_ctx_train) ?? readNumber(item.n_ctx_train)
    if (value != null) {
      values.set(id, value)
    }
  }

  return values
}

async function enrichLlamaCppContext(
  baseUrl: string,
  models: readonly ProviderModel[],
): Promise<readonly ProviderModel[]> {
  const root = normalizeBaseUrl(baseUrl)

  let slotsFetchAttempted = false
  let slotsFallback: number | null = null
  const getSlotsFallback = async (): Promise<number | null> => {
    if (slotsFetchAttempted) {
      return slotsFallback
    }
    slotsFetchAttempted = true
    try {
      const payload = await fetchJson<unknown>(llamaCppSlotsUrl(baseUrl))
      slotsFallback = parseLlamaSlotsContextLength(payload)
    } catch {
      slotsFallback = null
    }
    return slotsFallback
  }

  let v1MetaFetchAttempted = false
  let v1MetaMap: ReadonlyMap<string, number> = new Map()
  const getV1MetaMap = async (): Promise<ReadonlyMap<string, number>> => {
    if (v1MetaFetchAttempted) {
      return v1MetaMap
    }
    v1MetaFetchAttempted = true
    try {
      const payload = await fetchJson<unknown>(openAIModelsUrl(baseUrl))
      v1MetaMap = parseLlamaV1MetaContextMap(payload)
    } catch {
      v1MetaMap = new Map()
    }
    return v1MetaMap
  }

  const maxById = new Map<string, number>()

  for (const model of models) {
    const sources: ReadonlyArray<() => Promise<number | null>> = [
      async () => {
        for (const url of llamaCppPropsUrls(baseUrl, model.id)) {
          try {
            const payload = await fetchJson<unknown>(url)
            const parsed = parseLlamaCppContextLength(payload)
            if (parsed != null) {
              return parsed
            }
          } catch {
            // continue
          }
        }
        return null
      },
      async () => {
        try {
          const payload = await fetchJson<unknown>(`${root}/api/show`, {
            method: "POST",
            body: JSON.stringify({ name: model.id }),
          })
          return parseLlamaShowContextLength(payload)
        } catch {
          return null
        }
      },
      async () => {
        try {
          return await getSlotsFallback()
        } catch {
          return null
        }
      },
      async () => {
        try {
          return (await getV1MetaMap()).get(model.id) ?? null
        } catch {
          return null
        }
      },
    ]

    for (const source of sources) {
      const value = await source()
      if (value != null) {
        maxById.set(model.id, value)
        break
      }
    }
  }

  return models.map((model) => {
    const maxContextTokens = maxById.get(model.id) ?? model.maxContextTokens
    return {
      ...model,
      contextWindow: maxContextTokens ?? model.contextWindow,
      maxContextTokens,
    }
  })
}

export function discoverLlamaCppModels(
  providerId: string,
  providerName: string,
  baseUrl: string,
): Effect.Effect<LocalDiscoveryResult, never> {
  return discoverOpenAIModels(providerId, providerName, baseUrl).pipe(
    Effect.flatMap((models) =>
      Effect.tryPromise({
        try: async () => {
          const enriched = await enrichLlamaCppContext(baseUrl, models)
          return {
            models: enriched,
            error: null,
            source: "openai-v1-models",
            status: enriched.length > 0 ? "success_non_empty" : "success_empty",
            diagnostics: [],
          } satisfies LocalDiscoveryResult
        },
        catch: (error) =>
          new LocalDiscoveryError({
            message: "llama.cpp context enrichment failed",
            cause: error,
          }),
      }).pipe(
        Effect.orElseSucceed(() => ({
          models,
          error: null,
          source: "openai-v1-models",
          status: models.length > 0 ? "success_non_empty" : "success_empty",
          diagnostics: [],
        }) satisfies LocalDiscoveryResult),
      ),
    ),
    Effect.catchAll((error) =>
      Effect.succeed<LocalDiscoveryResult>({
        models: [],
        error: error.message || "discovery failed",
        source: null,
        status: "failure",
        diagnostics: [],
      }),
    ),
  )
}
