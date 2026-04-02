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

function toModel(id: string, name?: string, maxContextTokens: number | null = null): ModelDefinition {
  const now = new Date().toISOString()
  return {
    id,
    name: name || id,
    contextWindow: maxContextTokens ?? 200_000,
    maxContextTokens,
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

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
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
      const discovered = (payload.models ?? [])
        .map((m) => (m.name ?? m.model ?? '').trim())
        .filter(Boolean)
        .map((id) => toModel(id))
      return await enrichOllamaContext(baseUrl, discovered)
    },
    catch: (error) => error instanceof Error ? error : new Error('ollama discovery failed'),
  }).pipe(
    Effect.catchAll(() =>
      discoverOpenAIModels(baseUrl).pipe(
        Effect.flatMap((models) =>
          Effect.tryPromise({
            try: () => enrichOllamaContext(baseUrl, models),
            catch: () => models,
          }),
        ),
      )),
  )
}

function readNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value) && value > 0) return value
  if (typeof value === 'string') {
    const parsed = Number(value)
    if (Number.isFinite(parsed) && parsed > 0) return parsed
  }
  return null
}

function readLmStudioLoadedContext(item: Record<string, unknown>): number | null {
  const loaded = item.loaded_instances
  if (!Array.isArray(loaded)) return null
  for (const instance of loaded) {
    if (!instance || typeof instance !== 'object') continue
    const config = (instance as Record<string, unknown>).config
    if (!config || typeof config !== 'object') continue
    const parsed = readNumber((config as Record<string, unknown>).context_length)
    if (parsed != null) return parsed
  }
  return null
}

function readLmStudioLoadedContextV0(item: Record<string, unknown>): number | null {
  return (
    readNumber(item.loaded_context_length) ??
    readNumber(item.loadedContextLength) ??
    readNumber(item.context_length)
  )
}

function readModelLikeRecords(payload: unknown): Array<Record<string, unknown>> {
  const list = Array.isArray(payload)
    ? payload
    : (payload && typeof payload === 'object'
      ? ((payload as any).data ?? (payload as any).models ?? (payload as any).items ?? [])
      : [])
  if (!Array.isArray(list)) return []
  return list.filter((raw): raw is Record<string, unknown> => !!raw && typeof raw === 'object')
}

function readModelRecordId(item: Record<string, unknown>): string | null {
  return [
    item.id,
    item.model,
    item.modelKey,
    item.model_key,
  ].find((v): v is string => typeof v === 'string' && v.trim().length > 0)?.trim() ?? null
}

function readModelRecordName(item: Record<string, unknown>): string | null {
  return [
    item.name,
    item.title,
    item.modelName,
    item.model_name,
  ].find((v): v is string => typeof v === 'string' && v.trim().length > 0)?.trim() ?? null
}

function fromLmStudioNativeModels(payload: unknown): ModelDefinition[] {
  const out: ModelDefinition[] = []
  for (const item of readModelLikeRecords(payload)) {
    const id = readModelRecordId(item)
    if (!id) continue
    const name = readModelRecordName(item)
    const loadedContext = readLmStudioLoadedContext(item) ?? readLmStudioLoadedContextV0(item)
    const staticMax = readNumber(item.max_context_length)
    const maxContextTokens = loadedContext ?? staticMax
    out.push(toModel(id, name ?? id, maxContextTokens))
  }
  return out
}

function mergeContextMetadata(
  models: readonly ModelDefinition[],
  maxById: ReadonlyMap<string, number>,
): ModelDefinition[] {
  return models.map((model) => {
    const maxContextTokens = maxById.get(model.id) ?? null
    if (maxContextTokens == null) return model
    return {
      ...model,
      maxContextTokens,
      contextWindow: maxContextTokens,
    }
  })
}

type ContextSourceOutcome =
  | { outcome: 'usable'; value: number }
  | { outcome: 'missing' }
  | { outcome: 'failed' }

async function resolveFirstUsableContext(
  sources: ReadonlyArray<() => Promise<ContextSourceOutcome>>,
): Promise<number | null> {
  for (const source of sources) {
    const result = await source()
    if (result.outcome === 'usable') return result.value
  }
  return null
}

function toSourceOutcome(value: number | null): ContextSourceOutcome {
  return value == null ? { outcome: 'missing' } : { outcome: 'usable', value }
}

function normalizeLmStudioMatchKey(value: string): string {
  return value.trim().toLowerCase()
}

function findSingleLmStudioRecord(
  modelId: string,
  records: readonly Record<string, unknown>[],
): Record<string, unknown> | null {
  const exact = records.filter((item) => readModelRecordId(item) === modelId)
  if (exact.length === 1) return exact[0]!
  if (exact.length > 1) return null

  const normalizedModelId = normalizeLmStudioMatchKey(modelId)
  const normalized = records.filter((item) => {
    const id = readModelRecordId(item)
    return id != null && normalizeLmStudioMatchKey(id) === normalizedModelId
  })
  if (normalized.length === 1) return normalized[0]!
  return null
}

function maybeAttachContext(model: ModelDefinition, maxContextTokens: number | null): ModelDefinition {
  if (maxContextTokens == null) return model
  return {
    ...model,
    maxContextTokens,
    contextWindow: maxContextTokens,
  }
}

function lmStudioRoot(baseUrl: string): string {
  const normalized = normalizeBaseUrl(baseUrl)
  return normalized.endsWith('/v1') ? normalized.slice(0, -3) : normalized
}

function ollamaRoot(baseUrl: string): string {
  const normalized = normalizeBaseUrl(baseUrl)
  return normalized.endsWith('/v1') ? normalized.slice(0, -3) : normalized
}

function ollamaPsUrl(baseUrl: string): string {
  return `${ollamaRoot(baseUrl)}/api/ps`
}

function llamaCppPropsUrls(baseUrl: string, modelId: string): string[] {
  const normalized = normalizeBaseUrl(baseUrl)
  const withModel = `${normalized}/props?model=${encodeURIComponent(modelId)}`
  const plain = `${normalized}/props`
  return [withModel, plain]
}

function llamaCppSlotsUrl(baseUrl: string): string {
  return `${normalizeBaseUrl(baseUrl)}/slots`
}

function parseOllamaShowNumCtx(payload: unknown): number | null {
  if (!payload || typeof payload !== 'object') return null
  const obj = payload as Record<string, unknown>
  const direct = readNumber(obj.num_ctx)
  if (direct != null) return direct

  const parameters = obj.parameters
  if (typeof parameters === 'string') {
    const match = parameters.match(/(?:^|\s)num_ctx(?:\s+|=)(\d+)(?:\s|$)/)
    if (match?.[1]) {
      const parsed = readNumber(match[1])
      if (parsed != null) return parsed
    }
  }

  return null
}

function parseOllamaShowModelInfoContextLength(payload: unknown): number | null {
  if (!payload || typeof payload !== 'object') return null
  const obj = payload as Record<string, unknown>
  const modelInfo = obj.model_info
  if (!modelInfo || typeof modelInfo !== 'object') return null
  const info = modelInfo as Record<string, unknown>
  for (const [key, value] of Object.entries(info)) {
    if (key === 'context_length' || key.endsWith('.context_length')) {
      const parsed = readNumber(value)
      if (parsed != null) return parsed
    }
  }
  return null
}


function parseLlamaCppContextLength(payload: unknown): number | null {
  if (!payload || typeof payload !== 'object') return null
  const obj = payload as Record<string, unknown>
  const defaults = obj.default_generation_settings
  if (!defaults || typeof defaults !== 'object') return null
  return readNumber((defaults as Record<string, unknown>).n_ctx)
}

function parseLlamaShowContextLength(payload: unknown): number | null {
  if (!payload || typeof payload !== 'object') return null
  const obj = payload as Record<string, unknown>
  const modelInfo = obj.model_info
  if (!modelInfo || typeof modelInfo !== 'object') return null
  const info = modelInfo as Record<string, unknown>
  return readNumber(info['llama.context_length'])
}

function parseLlamaSlotsContextLength(payload: unknown): number | null {
  const slots = Array.isArray(payload)
    ? payload
    : (payload && typeof payload === 'object'
      ? ((payload as Record<string, unknown>).slots ?? payload)
      : null)

  if (Array.isArray(slots)) {
    for (const slot of slots) {
      if (!slot || typeof slot !== 'object') continue
      const parsed = readNumber((slot as Record<string, unknown>).n_ctx)
      if (parsed != null) return parsed
    }
    return null
  }

  if (slots && typeof slots === 'object') {
    return readNumber((slots as Record<string, unknown>).n_ctx)
  }

  return null
}

function trimModelKey(value: string): string {
  return value.trim()
}

function splitOllamaTag(value: string): { base: string; tag: string | null } {
  const trimmed = trimModelKey(value)
  const slash = trimmed.lastIndexOf('/')
  const colon = trimmed.lastIndexOf(':')
  if (colon > slash) {
    return { base: trimmed.slice(0, colon), tag: trimmed.slice(colon + 1) || null }
  }
  return { base: trimmed, tag: null }
}

function areOllamaIdsEquivalent(a: string, b: string): boolean {
  const aTrimmed = trimModelKey(a)
  const bTrimmed = trimModelKey(b)
  if (!aTrimmed || !bTrimmed) return false
  if (aTrimmed === bTrimmed) return true

  const aSplit = splitOllamaTag(aTrimmed)
  const bSplit = splitOllamaTag(bTrimmed)

  const aIsLatestOrNoTag = aSplit.tag == null || aSplit.tag === 'latest'
  const bIsLatestOrNoTag = bSplit.tag == null || bSplit.tag === 'latest'
  return aSplit.base === bSplit.base && aIsLatestOrNoTag && bIsLatestOrNoTag
}

async function enrichLmStudioContext(
  baseUrl: string,
  models: readonly ModelDefinition[],
): Promise<ModelDefinition[]> {
  let v1Records: Array<Record<string, unknown>> | null = null
  let v0Records: Array<Record<string, unknown>> | null = null

  const getV1Records = async (): Promise<Array<Record<string, unknown>>> => {
    if (v1Records != null) return v1Records
    const payload = await fetchJson<unknown>(lmStudioNativeV1Url(baseUrl))
    v1Records = readModelLikeRecords(payload)
    return v1Records
  }

  const getV0Records = async (): Promise<Array<Record<string, unknown>>> => {
    if (v0Records != null) return v0Records
    const payload = await fetchJson<unknown>(lmStudioNativeV0Url(baseUrl))
    v0Records = readModelLikeRecords(payload)
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
          return { outcome: 'failed' }
        }
      },
      async () => {
        try {
          const record = findSingleLmStudioRecord(model.id, await getV0Records())
          return toSourceOutcome(record ? readLmStudioLoadedContextV0(record) : null)
        } catch {
          return { outcome: 'failed' }
        }
      },
      async () => {
        try {
          const record = findSingleLmStudioRecord(model.id, await getV1Records())
          return toSourceOutcome(record ? readNumber(record.max_context_length) : null)
        } catch {
          return { outcome: 'failed' }
        }
      },
      async () => {
        try {
          const record = findSingleLmStudioRecord(model.id, await getV0Records())
          return toSourceOutcome(record ? readNumber(record.max_context_length) : null)
        } catch {
          return { outcome: 'failed' }
        }
      },
    ])

    if (value != null) maxById.set(model.id, value)
  }

  return mergeContextMetadata(models, maxById)
}

type OllamaPsEntry = { ids: string[]; context: number }

function parseOllamaPsEntries(payload: unknown): OllamaPsEntry[] {
  const out: OllamaPsEntry[] = []
  if (!payload || typeof payload !== 'object') return out
  const obj = payload as Record<string, unknown>
  const list = Array.isArray(obj.models) ? obj.models : []
  for (const raw of list) {
    if (!raw || typeof raw !== 'object') continue
    const item = raw as Record<string, unknown>
    const ids = [item.name, item.model]
      .filter((v): v is string => typeof v === 'string' && v.trim().length > 0)
      .map((v) => v.trim())
    if (ids.length === 0) continue
    const context =
      readNumber(item.context) ??
      readNumber(item.CONTEXT) ??
      readNumber(item.context_length) ??
      readNumber(item.CONTEXT_LENGTH) ??
      readNumber(item.contextLength) ??
      readNumber(item.num_ctx) ??
      readNumber(item.n_ctx)
    if (context == null) continue
    out.push({ ids: [...new Set(ids)], context })
  }
  return out
}

function matchOllamaRuntimeContext(modelId: string, entries: readonly OllamaPsEntry[]): number | null {
  const matched = entries.filter((entry) => entry.ids.some((id) => areOllamaIdsEquivalent(modelId, id)))
  if (matched.length === 0) return null
  const distinct = [...new Set(matched.map((m) => m.context))]
  return distinct.length === 1 ? distinct[0]! : null
}

async function enrichOllamaContext(
  baseUrl: string,
  models: readonly ModelDefinition[],
): Promise<ModelDefinition[]> {
  const root = ollamaRoot(baseUrl)
  const maxById = new Map<string, number>()
  let runtimeEntries: OllamaPsEntry[] | null = null

  const getRuntimeEntries = async (): Promise<OllamaPsEntry[]> => {
    if (runtimeEntries != null) return runtimeEntries
    const psPayload = await fetchJson<unknown>(ollamaPsUrl(baseUrl))
    runtimeEntries = parseOllamaPsEntries(psPayload)
    return runtimeEntries
  }

  for (const model of models) {
    let showPayload: unknown | null = null
    const getShowPayload = async (): Promise<unknown> => {
      if (showPayload != null) return showPayload
      showPayload = await fetchJson<unknown>(`${root}/api/show`, {
        method: 'POST',
        body: JSON.stringify({ name: model.id }),
      } as RequestInit)
      return showPayload
    }

    const value = await resolveFirstUsableContext([
      async () => {
        try {
          return toSourceOutcome(matchOllamaRuntimeContext(model.id, await getRuntimeEntries()))
        } catch {
          return { outcome: 'failed' }
        }
      },
      async () => {
        try {
          return toSourceOutcome(parseOllamaShowNumCtx(await getShowPayload()))
        } catch {
          return { outcome: 'failed' }
        }
      },
      async () => {
        try {
          return toSourceOutcome(parseOllamaShowModelInfoContextLength(await getShowPayload()))
        } catch {
          return { outcome: 'failed' }
        }
      },
    ])

    if (value != null) maxById.set(model.id, value)
  }

  return mergeContextMetadata(models, maxById)
}

function parseLlamaV1MetaContextMap(payload: unknown): Map<string, number> {
  const out = new Map<string, number>()
  if (!payload || typeof payload !== 'object') return out
  const list = Array.isArray((payload as Record<string, unknown>).data)
    ? (payload as Record<string, unknown>).data as unknown[]
    : []
  for (const raw of list) {
    if (!raw || typeof raw !== 'object') continue
    const item = raw as Record<string, unknown>
    const id = typeof item.id === 'string' && item.id.trim().length > 0 ? item.id.trim() : null
    if (!id) continue
    const meta = item.meta
    let value: number | null = null
    if (meta && typeof meta === 'object') {
      value = readNumber((meta as Record<string, unknown>).n_ctx_train)
    }
    value = value ?? readNumber(item.n_ctx_train)
    if (value != null) out.set(id, value)
  }
  return out
}

async function enrichLlamaCppContext(
  baseUrl: string,
  models: readonly ModelDefinition[],
): Promise<ModelDefinition[]> {
  const maxById = new Map<string, number>()
  const root = normalizeBaseUrl(baseUrl)

  let slotsFetchAttempted = false
  let slotsFallback: number | null = null
  const getSlotsFallback = async (): Promise<number | null> => {
    if (slotsFetchAttempted) return slotsFallback
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
  let v1MetaMap = new Map<string, number>()
  const getV1MetaMap = async (): Promise<Map<string, number>> => {
    if (v1MetaFetchAttempted) return v1MetaMap
    v1MetaFetchAttempted = true
    try {
      const payload = await fetchJson<unknown>(openAIModelsUrl(baseUrl))
      v1MetaMap = parseLlamaV1MetaContextMap(payload)
    } catch {
      v1MetaMap = new Map<string, number>()
    }
    return v1MetaMap
  }

  for (const model of models) {
    const value = await resolveFirstUsableContext([
      async () => {
        for (const url of llamaCppPropsUrls(baseUrl, model.id)) {
          try {
            const payload = await fetchJson<unknown>(url)
            const parsed = parseLlamaCppContextLength(payload)
            if (parsed != null) return { outcome: 'usable', value: parsed }
          } catch {
            // continue to next props fallback
          }
        }
        return { outcome: 'missing' }
      },
      async () => {
        try {
          const payload = await fetchJson<unknown>(`${root}/api/show`, {
            method: 'POST',
            body: JSON.stringify({ name: model.id }),
          } as RequestInit)
          return toSourceOutcome(parseLlamaShowContextLength(payload))
        } catch {
          return { outcome: 'failed' }
        }
      },
      async () => {
        try {
          return toSourceOutcome(await getSlotsFallback())
        } catch {
          return { outcome: 'failed' }
        }
      },
      async () => {
        try {
          const metaMap = await getV1MetaMap()
          return toSourceOutcome(metaMap.get(model.id) ?? null)
        } catch {
          return { outcome: 'failed' }
        }
      },
    ])

    if (value != null) maxById.set(model.id, value)
  }

  return mergeContextMetadata(models, maxById)
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
          const enriched = await enrichLmStudioContext(baseUrl, openAi)
          return { models: enriched, error: null, source: 'openai-v1-models', status: 'success_non_empty', diagnostics }
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
          const enriched = await enrichLmStudioContext(baseUrl, native)
          return { models: enriched, error: null, source: 'lmstudio-native-v1-models', status: 'success_non_empty', diagnostics }
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
          const enriched = await enrichLmStudioContext(baseUrl, nativeLegacy)
          return { models: enriched, error: null, source: 'lmstudio-native-v0-models', status: 'success_non_empty', diagnostics }
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

  if (provider.id === 'llama.cpp') {
    return discoverOpenAIModels(baseUrl).pipe(
      Effect.flatMap((models) =>
        Effect.tryPromise({
          try: async () => {
            const enriched = await enrichLlamaCppContext(baseUrl, models)
            return {
              models: enriched,
              error: null,
              source: 'openai-v1-models',
              status: enriched.length > 0 ? 'success_non_empty' : 'success_empty',
              diagnostics: [],
            } as LocalDiscoveryResult
          },
          catch: () => ({
            models,
            error: null,
            source: 'openai-v1-models',
            status: models.length > 0 ? 'success_non_empty' : 'success_empty',
            diagnostics: [],
          } as LocalDiscoveryResult),
        }),
      ),
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
