import { Effect } from "effect"
import type { ProviderModel } from "../../../lib/model/provider-model"
import { LocalDiscoveryError } from "./errors"
import type { LocalDiscoveryResult } from "./lmstudio"
import { discoverOpenAIModels } from "./openai-models"

interface OllamaTagsResponse {
  readonly models?: ReadonlyArray<{
    readonly name?: string
    readonly model?: string
  }>
}

interface OllamaShowResponse {
  readonly num_ctx?: number | string
  readonly parameters?: string
  readonly capabilities?: readonly string[]
  readonly model_info?: Record<string, unknown>
}

interface OllamaPsModel {
  readonly name?: string
  readonly model?: string
  readonly context?: number | string
  readonly CONTEXT?: number | string
  readonly context_length?: number | string
  readonly CONTEXT_LENGTH?: number | string
  readonly contextLength?: number | string
  readonly num_ctx?: number | string
  readonly n_ctx?: number | string
}

interface OllamaPsResponse {
  readonly models?: readonly OllamaPsModel[]
}

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, "")
}

function ollamaRoot(baseUrl: string): string {
  const normalized = normalizeBaseUrl(baseUrl)
  return normalized.endsWith("/v1") ? normalized.slice(0, -3) : normalized
}

function ollamaTagsUrl(baseUrl: string): string {
  const normalized = normalizeBaseUrl(baseUrl)
  return normalized.endsWith("/v1")
    ? `${normalized.slice(0, -3)}/api/tags`
    : `${normalized}/api/tags`
}

function ollamaPsUrl(baseUrl: string): string {
  return `${ollamaRoot(baseUrl)}/api/ps`
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

function trimModelKey(value: string): string {
  return value.trim()
}

function splitOllamaTag(value: string): { readonly base: string; readonly tag: string | null } {
  const trimmed = trimModelKey(value)
  const slash = trimmed.lastIndexOf("/")
  const colon = trimmed.lastIndexOf(":")
  if (colon > slash) {
    return { base: trimmed.slice(0, colon), tag: trimmed.slice(colon + 1) || null }
  }
  return { base: trimmed, tag: null }
}

function areOllamaIdsEquivalent(left: string, right: string): boolean {
  const leftTrimmed = trimModelKey(left)
  const rightTrimmed = trimModelKey(right)
  if (!leftTrimmed || !rightTrimmed) {
    return false
  }
  if (leftTrimmed === rightTrimmed) {
    return true
  }

  const leftSplit = splitOllamaTag(leftTrimmed)
  const rightSplit = splitOllamaTag(rightTrimmed)

  const leftIsLatestOrNoTag = leftSplit.tag == null || leftSplit.tag === "latest"
  const rightIsLatestOrNoTag = rightSplit.tag == null || rightSplit.tag === "latest"

  return leftSplit.base === rightSplit.base && leftIsLatestOrNoTag && rightIsLatestOrNoTag
}

interface OllamaPsEntry {
  readonly ids: readonly string[]
  readonly context: number
}

function parseOllamaPsEntries(payload: unknown): readonly OllamaPsEntry[] {
  if (!payload || typeof payload !== "object") {
    return []
  }

  const models = (payload as OllamaPsResponse).models
  if (!Array.isArray(models)) {
    return []
  }

  const entries: OllamaPsEntry[] = []
  for (const item of models) {
    const ids = [item.name, item.model]
      .filter((value): value is string => typeof value === "string" && value.trim().length > 0)
      .map((value) => value.trim())

    if (ids.length === 0) {
      continue
    }

    const context =
      readNumber(item.context) ??
      readNumber(item.CONTEXT) ??
      readNumber(item.context_length) ??
      readNumber(item.CONTEXT_LENGTH) ??
      readNumber(item.contextLength) ??
      readNumber(item.num_ctx) ??
      readNumber(item.n_ctx)

    if (context == null) {
      continue
    }

    entries.push({ ids: [...new Set(ids)], context })
  }

  return entries
}

function matchOllamaRuntimeContext(modelId: string, entries: readonly OllamaPsEntry[]): number | null {
  const matched = entries.filter((entry) => entry.ids.some((id) => areOllamaIdsEquivalent(modelId, id)))
  if (matched.length === 0) {
    return null
  }
  const distinct = [...new Set(matched.map((entry) => entry.context))]
  return distinct.length === 1 ? (distinct[0] ?? null) : null
}

function parseOllamaShowNumCtx(payload: unknown): number | null {
  if (!payload || typeof payload !== "object") {
    return null
  }

  const response = payload as OllamaShowResponse
  const direct = readNumber(response.num_ctx)
  if (direct != null) {
    return direct
  }

  if (typeof response.parameters === "string") {
    const match = response.parameters.match(/(?:^|\s)num_ctx(?:\s+|=)(\d+)(?:\s|$)/)
    if (match?.[1]) {
      return readNumber(match[1])
    }
  }

  return null
}

function parseOllamaShowModelInfoContextLength(payload: unknown): number | null {
  if (!payload || typeof payload !== "object") {
    return null
  }

  const modelInfo = (payload as OllamaShowResponse).model_info
  if (!modelInfo) {
    return null
  }

  for (const [key, value] of Object.entries(modelInfo)) {
    if (key === "context_length" || key.endsWith(".context_length")) {
      const parsed = readNumber(value)
      if (parsed != null) {
        return parsed
      }
    }
  }

  return null
}

function parseOllamaShowVision(payload: unknown): boolean | null {
  if (!payload || typeof payload !== "object") {
    return null
  }
  const capabilities = (payload as OllamaShowResponse).capabilities
  return Array.isArray(capabilities) ? capabilities.includes("vision") : null
}

async function enrichOllamaContext(
  baseUrl: string,
  models: readonly ProviderModel[],
): Promise<readonly ProviderModel[]> {
  const root = ollamaRoot(baseUrl)
  const maxById = new Map<string, number>()
  const visionById = new Map<string, boolean>()
  let runtimeEntries: readonly OllamaPsEntry[] | null = null

  const getRuntimeEntries = async (): Promise<readonly OllamaPsEntry[]> => {
    if (runtimeEntries != null) {
      return runtimeEntries
    }
    const psPayload = await fetchJson<unknown>(ollamaPsUrl(baseUrl))
    runtimeEntries = parseOllamaPsEntries(psPayload)
    return runtimeEntries
  }

  for (const model of models) {
    let showPayload: unknown | null = null

    const getShowPayload = async (): Promise<unknown> => {
      if (showPayload != null) {
        return showPayload
      }
      showPayload = await fetchJson<unknown>(`${root}/api/show`, {
        method: "POST",
        body: JSON.stringify({ name: model.id }),
      })
      return showPayload
    }

    const sources: ReadonlyArray<() => Promise<number | null>> = [
      async () => {
        try {
          return matchOllamaRuntimeContext(model.id, await getRuntimeEntries())
        } catch {
          return null
        }
      },
      async () => {
        try {
          return parseOllamaShowNumCtx(await getShowPayload())
        } catch {
          return null
        }
      },
      async () => {
        try {
          return parseOllamaShowModelInfoContextLength(await getShowPayload())
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

    try {
      const vision = parseOllamaShowVision(await getShowPayload())
      if (vision != null) {
        visionById.set(model.id, vision)
      }
    } catch {
      // ignore
    }
  }

  return models.map((model) => {
    const maxContextTokens = maxById.get(model.id) ?? model.maxContextTokens
    const supportsVision = visionById.get(model.id) ?? model.supportsVision
    return {
      ...model,
      contextWindow: maxContextTokens ?? model.contextWindow,
      maxContextTokens,
      supportsVision,
    }
  })
}

export function discoverOllamaModels(
  providerId: string,
  providerName: string,
  baseUrl: string,
): Effect.Effect<LocalDiscoveryResult, never> {
  return Effect.tryPromise({
    try: async () => {
      const payload = await fetchJson<OllamaTagsResponse>(ollamaTagsUrl(baseUrl))
      const discovered = (payload.models ?? [])
        .map((model) => (model.name ?? model.model ?? "").trim())
        .filter(Boolean)
        .map((id) => ({
          id,
          providerId,
          providerName,
          canonicalModelId: null,
          name: id,
          contextWindow: 200_000,
          maxContextTokens: null,
          maxOutputTokens: null,
          supportsToolCalls: true,
          supportsReasoning: false,
          supportsVision: false,
          costs: {
            inputPerM: 0,
            outputPerM: 0,
            cacheReadPerM: null,
            cacheWritePerM: null,
          },
          releaseDate: new Date().toISOString(),
          discovery: {
            primarySource: "local" as const,
            fetchedAt: new Date().toISOString(),
          },
        } satisfies ProviderModel))

      const enriched = await enrichOllamaContext(baseUrl, discovered)
      return {
        models: enriched,
        error: null,
        source: "ollama-api-tags",
        status: enriched.length > 0 ? "success_non_empty" : "success_empty",
        diagnostics: [],
      } satisfies LocalDiscoveryResult
    },
    catch: (error) => new LocalDiscoveryError({ message: "ollama discovery failed", cause: error }),
  }).pipe(
    Effect.catchAll(() =>
      discoverOpenAIModels(providerId, providerName, baseUrl).pipe(
        Effect.flatMap((models) =>
          Effect.tryPromise({
            try: () => enrichOllamaContext(baseUrl, models),
            catch: (error) =>
              new LocalDiscoveryError({
                message: "ollama context enrichment failed",
                cause: error,
              }),
          }).pipe(Effect.orElseSucceed(() => models)),
        ),
        Effect.map((models) => ({
          models,
          error: null,
          source: "openai-v1-models",
          status: models.length > 0 ? "success_non_empty" : "success_empty",
          diagnostics: [],
        }) satisfies LocalDiscoveryResult),
        Effect.catchAll((error) =>
          Effect.succeed<LocalDiscoveryResult>({
            models: [],
            error: error.message || "discovery failed",
            source: null,
            status: "failure",
            diagnostics: [],
          }),
        ),
      ),
    ),
  )
}
