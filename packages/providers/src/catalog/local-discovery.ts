import { Effect } from 'effect'
import type { ModelDefinition, ProviderDefinition } from '../types'

type OpenAIModelsResponse = {
  data?: Array<{ id?: string; name?: string }>
}

type OllamaTagsResponse = {
  models?: Array<{ name?: string; model?: string }>
}

export type LocalDiscoveryStatus = 'success_non_empty' | 'success_empty' | 'failure'

export type LocalDiscoveryResult = {
  models: ModelDefinition[]
  error: string | null
  source: string | null
  status: LocalDiscoveryStatus
  diagnostics: string[]
}

const STATIC_COST = { input: 0, output: 0 } as const

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, '')
}

function toModel(id: string, name?: string): ModelDefinition {
  const now = new Date().toISOString()
  return {
    id,
    name: name || id,
    contextWindow: 200_000,
    supportsToolCalls: true,
    supportsReasoning: false,
    cost: { ...STATIC_COST },
    family: 'local',
    releaseDate: now,
    discovery: { primarySource: 'static', fetchedAt: now },
  }
}

function fromOpenAIModels(payload: OpenAIModelsResponse): ModelDefinition[] {
  const out: ModelDefinition[] = []
  for (const model of payload.data ?? []) {
    const id = model.id?.trim()
    if (!id) continue
    out.push(toModel(id, model.name?.trim() || id))
  }
  return out
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`)
  return (await res.json()) as T
}

function openAIModelsUrl(baseUrl: string): string {
  const normalized = normalizeBaseUrl(baseUrl)
  return normalized.endsWith('/v1') ? `${normalized}/models` : `${normalized}/v1/models`
}

function ollamaTagsUrl(baseUrl: string): string {
  const normalized = normalizeBaseUrl(baseUrl)
  return normalized.endsWith('/v1')
    ? `${normalized.slice(0, -3)}/api/tags`
    : `${normalized}/api/tags`
}

function lmStudioNativeV1Url(baseUrl: string): string {
  const normalized = normalizeBaseUrl(baseUrl)
  const root = normalized.endsWith('/v1') ? normalized.slice(0, -3) : normalized
  return `${root}/api/v1/models`
}

function lmStudioNativeV0Url(baseUrl: string): string {
  const normalized = normalizeBaseUrl(baseUrl)
  const root = normalized.endsWith('/v1') ? normalized.slice(0, -3) : normalized
  return `${root}/api/v0/models`
}

export function discoverOpenAIModels(baseUrl: string): Effect.Effect<ModelDefinition[], Error> {
  return Effect.tryPromise({
    try: async () => {
      const payload = await fetchJson<OpenAIModelsResponse>(openAIModelsUrl(baseUrl))
      return fromOpenAIModels(payload)
    },
    catch: (error) => error instanceof Error ? error : new Error('discovery failed'),
  })
}

export function discoverOllamaHybrid(baseUrl: string): Effect.Effect<ModelDefinition[], Error> {
  return Effect.tryPromise({
    try: async () => {
      const payload = await fetchJson<OllamaTagsResponse>(ollamaTagsUrl(baseUrl))
      return (payload.models ?? [])
        .map((m) => (m.name ?? m.model ?? '').trim())
        .filter(Boolean)
        .map((id) => toModel(id))
    },
    catch: (error) => error instanceof Error ? error : new Error('ollama discovery failed'),
  }).pipe(
    Effect.catchAll(() => discoverOpenAIModels(baseUrl)),
  )
}

function fromLmStudioNativeModels(payload: unknown): ModelDefinition[] {
  const list = Array.isArray(payload)
    ? payload
    : (payload && typeof payload === 'object'
      ? ((payload as any).data ?? (payload as any).models ?? (payload as any).items ?? [])
      : [])
  if (!Array.isArray(list)) return []

  const out: ModelDefinition[] = []
  for (const raw of list) {
    if (!raw || typeof raw !== 'object') continue
    const item = raw as Record<string, unknown>
    const id = [
      item.id,
      item.model,
      item.modelKey,
      item.model_key,
    ].find((v): v is string => typeof v === 'string' && v.trim().length > 0)?.trim()
    if (!id) continue
    const name = [
      item.name,
      item.title,
      item.modelName,
      item.model_name,
    ].find((v): v is string => typeof v === 'string' && v.trim().length > 0)?.trim()
    out.push(toModel(id, name ?? id))
  }
  return out
}

function discoverLmStudioAware(baseUrl: string): Effect.Effect<LocalDiscoveryResult, never> {
  return Effect.tryPromise({
    try: async () => {
      const diagnostics: string[] = []
      let openAiWorked = false
      let nativeV1Worked = false
      let nativeV0Worked = false
      let openAiError: string | null = null
      let nativeV1Error: string | null = null
      let nativeV0Error: string | null = null

      try {
        const openAi = await Effect.runPromise(discoverOpenAIModels(baseUrl))
        openAiWorked = true
        if (openAi.length > 0) {
          return { models: openAi, error: null, source: 'openai-v1-models', status: 'success_non_empty', diagnostics }
        }
        diagnostics.push('openai-v1-models returned 0 models')
      } catch (error) {
        openAiError = error instanceof Error ? error.message : String(error)
        diagnostics.push(`openai-v1-models failed: ${openAiError}`)
      }

      try {
        const payload = await fetchJson<unknown>(lmStudioNativeV1Url(baseUrl))
        const native = fromLmStudioNativeModels(payload)
        nativeV1Worked = true
        if (native.length > 0) {
          return { models: native, error: null, source: 'lmstudio-native-v1-models', status: 'success_non_empty', diagnostics }
        }
        diagnostics.push('lmstudio-native-v1-models returned 0 models')
      } catch (error) {
        nativeV1Error = error instanceof Error ? error.message : String(error)
        diagnostics.push(`lmstudio-native-v1-models failed: ${nativeV1Error}`)
      }

      try {
        const payload = await fetchJson<unknown>(lmStudioNativeV0Url(baseUrl))
        const nativeLegacy = fromLmStudioNativeModels(payload)
        nativeV0Worked = true
        if (nativeLegacy.length > 0) {
          return { models: nativeLegacy, error: null, source: 'lmstudio-native-v0-models', status: 'success_non_empty', diagnostics }
        }
        diagnostics.push('lmstudio-native-v0-models returned 0 models')
      } catch (error) {
        nativeV0Error = error instanceof Error ? error.message : String(error)
        diagnostics.push(`lmstudio-native-v0-models failed: ${nativeV0Error}`)
      }

      const allFailed = !openAiWorked && !nativeV1Worked && !nativeV0Worked
      if (allFailed) {
        return {
          models: [],
          error: `LM Studio discovery failed: /v1/models (${openAiError ?? 'unknown'}); /api/v1/models (${nativeV1Error ?? 'unknown'}); /api/v0/models (${nativeV0Error ?? 'unknown'})`,
          source: null,
          status: 'failure',
          diagnostics,
        }
      }

      return {
        models: [],
        error: null,
        source: openAiWorked ? 'openai-v1-models' : (nativeV1Worked ? 'lmstudio-native-v1-models' : 'lmstudio-native-v0-models'),
        status: 'success_empty',
        diagnostics,
      }
    },
    catch: (error) => ({
      models: [],
      error: error instanceof Error ? error.message : 'discovery failed',
      source: null,
      status: 'failure',
      diagnostics: ['lmstudio-discovery-unexpected-error'],
    }),
  })
}

export function discoverLocalProviderModels(
  provider: ProviderDefinition,
  baseUrl: string | undefined,
): Effect.Effect<LocalDiscoveryResult, never> {
  if (!baseUrl || !provider.localDiscoveryStrategy) {
    return Effect.succeed({
      models: [],
      error: null,
      source: null,
      status: 'success_empty',
      diagnostics: [],
    })
  }

  if (provider.id === 'lmstudio') {
    return discoverLmStudioAware(baseUrl)
  }

  const run = (() => {
    switch (provider.localDiscoveryStrategy) {
      case 'openai-models':
        return discoverOpenAIModels(baseUrl).pipe(
          Effect.map((models) => ({
            models,
            error: null,
            source: 'openai-v1-models',
            status: models.length > 0 ? 'success_non_empty' : 'success_empty',
            diagnostics: [],
          })),
        )
      case 'ollama-hybrid':
        return discoverOllamaHybrid(baseUrl).pipe(
          Effect.map((models) => ({
            models,
            error: null,
            source: 'ollama-api-tags',
            status: models.length > 0 ? 'success_non_empty' : 'success_empty',
            diagnostics: [],
          })),
        )
      case 'openai-models-best-effort':
        return discoverOpenAIModels(baseUrl).pipe(
          Effect.map((models) => ({
            models,
            error: null,
            source: 'openai-v1-models',
            status: models.length > 0 ? 'success_non_empty' : 'success_empty',
            diagnostics: [],
          })),
        )
      default:
        return Effect.succeed({
          models: [],
          error: null,
          source: null,
          status: 'success_empty',
          diagnostics: [],
        })
    }
  })()

  return run.pipe(
    Effect.catchAll((error) =>
      Effect.succeed({
        models: [],
        error: error.message || 'discovery failed',
        source: null,
        status: 'failure',
        diagnostics: [],
      }),
    ),
  )
}

export function mergeDiscoveredAndRemembered(
  discoveredModels: readonly ModelDefinition[],
  rememberedModelIds: readonly string[] | undefined,
): ModelDefinition[] {
  const byId = new Map<string, ModelDefinition>()
  for (const model of discoveredModels) byId.set(model.id, model)
  for (const id of rememberedModelIds ?? []) {
    if (!id || byId.has(id)) continue
    byId.set(id, toModel(id))
  }
  return [...byId.values()]
}
