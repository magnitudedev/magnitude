import { Effect } from "effect"
import type { ProviderModel } from "../../../lib/model/provider-model"
import { LocalDiscoveryError } from "./errors"
import { discoverOpenAIModels } from "./openai-models"

interface LmStudioLoadedInstance {
  readonly config?: {
    readonly context_length?: number | string
  }
}

interface LmStudioModelV1 {
  readonly id?: string
  readonly model?: string
  readonly modelKey?: string
  readonly model_key?: string
  readonly name?: string
  readonly title?: string
  readonly modelName?: string
  readonly model_name?: string
  readonly max_context_length?: number | string
  readonly loaded_instances?: readonly LmStudioLoadedInstance[]
  readonly capabilities?: { readonly vision?: boolean }
}

interface LmStudioModelV0 {
  readonly id?: string
  readonly model?: string
  readonly modelKey?: string
  readonly model_key?: string
  readonly name?: string
  readonly title?: string
  readonly modelName?: string
  readonly model_name?: string
  readonly max_context_length?: number | string
  readonly loaded_context_length?: number | string
  readonly loadedContextLength?: number | string
  readonly context_length?: number | string
  readonly capabilities?: { readonly vision?: boolean }
}

type LmStudioModel = LmStudioModelV1 | LmStudioModelV0

interface LmStudioResponse {
  readonly data?: readonly LmStudioModel[]
  readonly models?: readonly LmStudioModel[]
  readonly items?: readonly LmStudioModel[]
}

export type LocalDiscoveryStatus = "success_non_empty" | "success_empty" | "failure"

export interface LocalDiscoveryResult {
  readonly models: readonly ProviderModel[]
  readonly error: string | null
  readonly source: string | null
  readonly status: LocalDiscoveryStatus
  readonly diagnostics: readonly string[]
}

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, "")
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} for ${url}`)
  }
  return (await response.json()) as T
}

function lmStudioNativeV1Url(baseUrl: string): string {
  const normalized = normalizeBaseUrl(baseUrl)
  const root = normalized.endsWith("/v1") ? normalized.slice(0, -3) : normalized
  return `${root}/api/v1/models`
}

function lmStudioNativeV0Url(baseUrl: string): string {
  const normalized = normalizeBaseUrl(baseUrl)
  const root = normalized.endsWith("/v1") ? normalized.slice(0, -3) : normalized
  return `${root}/api/v0/models`
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

function isLmStudioModel(value: unknown): value is LmStudioModel {
  return !!value && typeof value === "object"
}

function parseLmStudioModels(payload: unknown): readonly LmStudioModel[] {
  const list = Array.isArray(payload)
    ? payload
    : payload && typeof payload === "object"
      ? ((payload as LmStudioResponse).data ??
        (payload as LmStudioResponse).models ??
        (payload as LmStudioResponse).items ??
        [])
      : []

  return Array.isArray(list) ? list.filter(isLmStudioModel) : []
}

function readModelRecordId(item: LmStudioModel): string | null {
  return [item.id, item.model, item.modelKey, item.model_key].find(
    (value): value is string => typeof value === "string" && value.trim().length > 0,
  )?.trim() ?? null
}

function readModelRecordName(item: LmStudioModel): string | null {
  return [item.name, item.title, item.modelName, item.model_name].find(
    (value): value is string => typeof value === "string" && value.trim().length > 0,
  )?.trim() ?? null
}

function readLmStudioLoadedContext(item: LmStudioModelV1): number | null {
  for (const instance of item.loaded_instances ?? []) {
    const parsed = readNumber(instance.config?.context_length)
    if (parsed != null) {
      return parsed
    }
  }
  return null
}

function readLmStudioLoadedContextV0(item: LmStudioModelV0): number | null {
  return (
    readNumber(item.loaded_context_length) ??
    readNumber(item.loadedContextLength) ??
    readNumber(item.context_length)
  )
}

function updateModel(
  model: ProviderModel,
  maxContextTokens: number | null,
  supportsVision: boolean,
): ProviderModel {
  return {
    ...model,
    contextWindow: maxContextTokens ?? model.contextWindow,
    maxContextTokens,
    supportsVision,
  }
}

function fromLmStudioNativeModels(
  providerId: string,
  providerName: string,
  payload: unknown,
): readonly ProviderModel[] {
  const now = new Date().toISOString()

  return parseLmStudioModels(payload).flatMap((item): readonly ProviderModel[] => {
    const id = readModelRecordId(item)
    if (!id) {
      return []
    }

    const name = readModelRecordName(item) ?? id
    const loadedContext = readLmStudioLoadedContext(item) ?? readLmStudioLoadedContextV0(item)
    const staticMax = readNumber(item.max_context_length)
    const maxContextTokens = loadedContext ?? staticMax
    const supportsVision = item.capabilities?.vision ?? false

    return [
      {
        id,
        providerId,
        providerName,
        canonicalModelId: null,
        name,
        contextWindow: maxContextTokens ?? 200_000,
        maxContextTokens,
        maxOutputTokens: null,
        supportsToolCalls: true,
        supportsReasoning: false,
        supportsVision,
        costs: {
          inputPerM: 0,
          outputPerM: 0,
          cacheReadPerM: null,
          cacheWritePerM: null,
        },
        releaseDate: now,
        discovery: {
          primarySource: "local",
          fetchedAt: now,
        },
      },
    ]
  })
}

function normalizeLmStudioMatchKey(value: string): string {
  return value.trim().toLowerCase()
}

function findSingleLmStudioRecord(
  modelId: string,
  records: readonly LmStudioModel[],
): LmStudioModel | null {
  const exact = records.filter((item) => readModelRecordId(item) === modelId)
  if (exact.length === 1) {
    return exact[0] ?? null
  }
  if (exact.length > 1) {
    return null
  }

  const normalizedModelId = normalizeLmStudioMatchKey(modelId)
  const normalized = records.filter((item) => {
    const id = readModelRecordId(item)
    return id != null && normalizeLmStudioMatchKey(id) === normalizedModelId
  })

  return normalized.length === 1 ? (normalized[0] ?? null) : null
}

type ContextSourceOutcome =
  | { readonly outcome: "usable"; readonly value: number }
  | { readonly outcome: "missing" }
  | { readonly outcome: "failed" }

function toSourceOutcome(value: number | null): ContextSourceOutcome {
  return value == null ? { outcome: "missing" } : { outcome: "usable", value }
}

async function resolveFirstUsableContext(
  sources: ReadonlyArray<() => Promise<ContextSourceOutcome>>,
): Promise<number | null> {
  for (const source of sources) {
    const result = await source()
    if (result.outcome === "usable") {
      return result.value
    }
  }
  return null
}

async function enrichLmStudioContext(
  baseUrl: string,
  models: readonly ProviderModel[],
): Promise<readonly ProviderModel[]> {
  let v1Records: readonly LmStudioModel[] | null = null
  let v0Records: readonly LmStudioModel[] | null = null

  const getV1Records = async (): Promise<readonly LmStudioModel[]> => {
    if (v1Records != null) {
      return v1Records
    }
    const payload = await fetchJson<unknown>(lmStudioNativeV1Url(baseUrl))
    v1Records = parseLmStudioModels(payload)
    return v1Records
  }

  const getV0Records = async (): Promise<readonly LmStudioModel[]> => {
    if (v0Records != null) {
      return v0Records
    }
    const payload = await fetchJson<unknown>(lmStudioNativeV0Url(baseUrl))
    v0Records = parseLmStudioModels(payload)
    return v0Records
  }

  const maxById = new Map<string, number>()

  for (const model of models) {
    const value = await resolveFirstUsableContext([
      async () => {
        try {
          const record = findSingleLmStudioRecord(model.id, await getV1Records())
          return toSourceOutcome(record ? readLmStudioLoadedContext(record) : null)
        } catch {
          return { outcome: "failed" } as const
        }
      },
      async () => {
        try {
          const record = findSingleLmStudioRecord(model.id, await getV0Records())
          return toSourceOutcome(record ? readLmStudioLoadedContextV0(record) : null)
        } catch {
          return { outcome: "failed" } as const
        }
      },
      async () => {
        try {
          const record = findSingleLmStudioRecord(model.id, await getV1Records())
          return toSourceOutcome(record ? readNumber(record.max_context_length) : null)
        } catch {
          return { outcome: "failed" } as const
        }
      },
      async () => {
        try {
          const record = findSingleLmStudioRecord(model.id, await getV0Records())
          return toSourceOutcome(record ? readNumber(record.max_context_length) : null)
        } catch {
          return { outcome: "failed" } as const
        }
      },
    ])

    if (value != null) {
      maxById.set(model.id, value)
    }
  }

  return models.map((model) => {
    const maxContextTokens = maxById.get(model.id) ?? model.maxContextTokens
    return updateModel(model, maxContextTokens, model.supportsVision)
  })
}

export function discoverLmStudioModels(
  providerId: string,
  providerName: string,
  baseUrl: string,
): Effect.Effect<LocalDiscoveryResult, never> {
  return Effect.tryPromise({
    try: async (): Promise<LocalDiscoveryResult> => {
      const diagnostics: string[] = []
      let openAiWorked = false
      let nativeV1Worked = false
      let nativeV0Worked = false
      let openAiError: string | null = null
      let nativeV1Error: string | null = null
      let nativeV0Error: string | null = null

      try {
        const openAi = await Effect.runPromise(discoverOpenAIModels(providerId, providerName, baseUrl))
        openAiWorked = true
        if (openAi.length > 0) {
          const enriched = await enrichLmStudioContext(baseUrl, openAi)
          return {
            models: enriched,
            error: null,
            source: "openai-v1-models",
            status: "success_non_empty",
            diagnostics,
          }
        }
        diagnostics.push("openai-v1-models returned 0 models")
      } catch (error) {
        openAiError = error instanceof Error ? error.message : String(error)
        diagnostics.push(`openai-v1-models failed: ${openAiError}`)
      }

      try {
        const payload = await fetchJson<unknown>(lmStudioNativeV1Url(baseUrl))
        const native = fromLmStudioNativeModels(providerId, providerName, payload)
        nativeV1Worked = true
        if (native.length > 0) {
          const enriched = await enrichLmStudioContext(baseUrl, native)
          return {
            models: enriched,
            error: null,
            source: "lmstudio-native-v1-models",
            status: "success_non_empty",
            diagnostics,
          }
        }
        diagnostics.push("lmstudio-native-v1-models returned 0 models")
      } catch (error) {
        nativeV1Error = error instanceof Error ? error.message : String(error)
        diagnostics.push(`lmstudio-native-v1-models failed: ${nativeV1Error}`)
      }

      try {
        const payload = await fetchJson<unknown>(lmStudioNativeV0Url(baseUrl))
        const nativeLegacy = fromLmStudioNativeModels(providerId, providerName, payload)
        nativeV0Worked = true
        if (nativeLegacy.length > 0) {
          const enriched = await enrichLmStudioContext(baseUrl, nativeLegacy)
          return {
            models: enriched,
            error: null,
            source: "lmstudio-native-v0-models",
            status: "success_non_empty",
            diagnostics,
          }
        }
        diagnostics.push("lmstudio-native-v0-models returned 0 models")
      } catch (error) {
        nativeV0Error = error instanceof Error ? error.message : String(error)
        diagnostics.push(`lmstudio-native-v0-models failed: ${nativeV0Error}`)
      }

      if (!openAiWorked && !nativeV1Worked && !nativeV0Worked) {
        return {
          models: [],
          error: `LM Studio discovery failed: /v1/models (${openAiError ?? "unknown"}); /api/v1/models (${nativeV1Error ?? "unknown"}); /api/v0/models (${nativeV0Error ?? "unknown"})`,
          source: null,
          status: "failure",
          diagnostics,
        }
      }

      return {
        models: [],
        error: null,
        source: openAiWorked
          ? "openai-v1-models"
          : nativeV1Worked
            ? "lmstudio-native-v1-models"
            : "lmstudio-native-v0-models",
        status: "success_empty",
        diagnostics,
      }
    },
    catch: (error) => new LocalDiscoveryError({ message: "lmstudio discovery failed", cause: error }),
  }).pipe(
    Effect.catchAll((error) =>
      Effect.succeed<LocalDiscoveryResult>({
        models: [],
        error: error.message,
        source: null,
        status: "failure",
        diagnostics: ["lmstudio-discovery-unexpected-error"],
      }),
    ),
  )
}
