import { Effect } from 'effect'
import type { ModelDefinition, ProviderDefinition } from '../types'

type OpenAIModelsResponse = {
  data?: Array<{ id?: string; name?: string }>
}

type OllamaTagsResponse = {
  models?: Array<{ name?: string; model?: string }>
}

interface LmStudioLoadedInstance {
  config?: {
    context_length?: number | string
  }
}

interface LmStudioModelV1 {
  id?: string
  model?: string
  modelKey?: string
  model_key?: string
  name?: string
  title?: string
  modelName?: string
  model_name?: string
  max_context_length?: number | string
  loaded_instances?: LmStudioLoadedInstance[]
}

interface LmStudioModelV0 {
  id?: string
  model?: string
  modelKey?: string
  model_key?: string
  name?: string
  title?: string
  modelName?: string
  model_name?: string
  max_context_length?: number | string
  loaded_context_length?: number | string
  loadedContextLength?: number | string
  context_length?: number | string
}

type LmStudioModel = LmStudioModelV1 | LmStudioModelV0

interface LmStudioResponse {
  data?: LmStudioModel[]
  models?: LmStudioModel[]
  items?: LmStudioModel[]
}

interface OllamaShowResponse {
  num_ctx?: number | string
  parameters?: string
  model_info?: {
    context_length?: number | string
    [key: string]: unknown
  }
}

interface OllamaPsModel {
  name?: string
  model?: string
  context?: number | string
  CONTEXT?: number | string
  context_length?: number | string
  CONTEXT_LENGTH?: number | string
  contextLength?: number | string
  num_ctx?: number | string
  n_ctx?: number | string
}

interface OllamaPsResponse {
  models?: OllamaPsModel[]
}

interface LlamaCppPropsResponse {
  default_generation_settings?: {
    n_ctx?: number | string
  }
}

interface LlamaSlotsEntry {
  n_ctx?: number | string
}

interface LlamaSlotsResponse {
  slots?: LlamaSlotsEntry[] | LlamaSlotsEntry
}

interface LlamaShowResponse {
  model_info?: {
    'llama.context_length'?: number | string
  }
}

interface LlamaV1MetaEntry {
  id?: string
  n_ctx_train?: number | string
  meta?: {
    n_ctx_train?: number | string
  }
}

interface LlamaV1MetaResponse {
  data?: LlamaV1MetaEntry[]
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
            catch: (error) => error instanceof Error ? error : new Error('ollama context enrichment failed'),
          }).pipe(Effect.orElseSucceed(() => models)),
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

function isLmStudioModel(value: unknown): value is LmStudioModel {
  return !!value && typeof value === 'object'
}

function parseLmStudioModels(payload: unknown): LmStudioModel[] {
  const list = Array.isArray(payload)
    ? payload
    : payload && typeof payload === 'object'
      ? ((payload as LmStudioResponse).data ?? (payload as LmStudioResponse).models ?? (payload as LmStudioResponse).items ?? [])
      : []

  if (!Array.isArray(list)) return []
  return list.filter(isLmStudioModel)
}

function readModelRecordId(item: LmStudioModel): string | null {
  return [
    item.id,
    item.model,
    item.modelKey,
    item.model_key,
  ].find((v): v is string => typeof v === 'string' && v.trim().length > 0)?.trim() ?? null
}

function readModelRecordName(item: LmStudioModel): string | null {
  return [
    item.name,
    item.title,
    item.modelName,
    item.model_name,
  ].find((v): v is string => typeof v === 'string' && v.trim().length > 0)?.trim() ?? null
}

function readLmStudioLoadedContext(item: LmStudioModelV1): number | null {
  for (const instance of item.loaded_instances ?? []) {
    const parsed = readNumber(instance.config?.context_length)
    if (parsed != null) return parsed
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

function fromLmStudioNativeModels(payload: unknown): ModelDefinition[] {
  const out: ModelDefinition[] = []
  for (const item of parseLmStudioModels(payload)) {
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
  records: readonly LmStudioModel[],
): LmStudioModel | null {
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

function isOllamaShowResponse(payload: unknown): payload is OllamaShowResponse {
  return !!payload && typeof payload === 'object'
}

function parseOllamaShowNumCtx(payload: unknown): number | null {
  if (!isOllamaShowResponse(payload)) return null
  const direct = readNumber(payload.num_ctx)
  if (direct != null) return direct

  const parameters = payload.parameters
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
  if (!isOllamaShowResponse(payload) || !payload.model_info) return null
  for (const [key, value] of Object.entries(payload.model_info)) {
    if (key === 'context_length' || key.endsWith('.context_length')) {
      const parsed = readNumber(value)
      if (parsed != null) return parsed
    }
  }
  return null
}

function isLlamaCppPropsResponse(payload: unknown): payload is LlamaCppPropsResponse {
  return !!payload && typeof payload === 'object'
}

function parseLlamaCppContextLength(payload: unknown): number | null {
  if (!isLlamaCppPropsResponse(payload)) return null
  return readNumber(payload.default_generation_settings?.n_ctx)
}

function isLlamaShowResponse(payload: unknown): payload is LlamaShowResponse {
  return !!payload && typeof payload === 'object'
}

function parseLlamaShowContextLength(payload: unknown): number | null {
  if (!isLlamaShowResponse(payload)) return null
  return readNumber(payload.model_info?.['llama.context_length'])
}

function isLlamaSlotsEntry(payload: unknown): payload is LlamaSlotsEntry {
  return !!payload && typeof payload === 'object'
}

function isLlamaSlotsResponse(payload: unknown): payload is LlamaSlotsResponse {
  return !!payload && typeof payload === 'object'
}

function parseLlamaSlotsContextLength(payload: unknown): number | null {
  const slots = Array.isArray(payload)
    ? payload
    : isLlamaSlotsResponse(payload)
      ? (payload.slots ?? payload)
      : null

  if (Array.isArray(slots)) {
    for (const slot of slots) {
      if (!isLlamaSlotsEntry(slot)) continue
      const parsed = readNumber(slot.n_ctx)
      if (parsed != null) return parsed
    }
    return null
  }

  if (isLlamaSlotsEntry(slots)) {
    return readNumber(slots.n_ctx)
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
  let v1Records: LmStudioModel[] | null = null
  let v0Records: LmStudioModel[] | null = null

  const getV1Records = async (): Promise<LmStudioModel[]> => {
    if (v1Records != null) return v1Records
    const payload = await fetchJson<unknown>(lmStudioNativeV1Url(baseUrl))
    v1Records = parseLmStudioModels(payload)
    return v1Records
  }

  const getV0Records = async (): Promise<LmStudioModel[]> => {
    if (v0Records != null) return v0Records
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

function isOllamaPsResponse(payload: unknown): payload is OllamaPsResponse {
  return !!payload && typeof payload === 'object'
}

function parseOllamaPsEntries(payload: unknown): OllamaPsEntry[] {
  const out: OllamaPsEntry[] = []
  if (!isOllamaPsResponse(payload)) return out
  for (const item of payload.models ?? []) {
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

function isLlamaV1MetaResponse(payload: unknown): payload is LlamaV1MetaResponse {
  return !!payload && typeof payload === 'object'
}

function parseLlamaV1MetaContextMap(payload: unknown): Map<string, number> {
  const out = new Map<string, number>()
  if (!isLlamaV1MetaResponse(payload)) return out
  for (const item of payload.data ?? []) {
    const id = typeof item.id === 'string' && item.id.trim().length > 0 ? item.id.trim() : null
    if (!id) continue
    const value = readNumber(item.meta?.n_ctx_train) ?? readNumber(item.n_ctx_train)
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
    try: async (): Promise<LocalDiscoveryResult> => {
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
    catch: (error) => error instanceof Error ? error : new Error('lmstudio discovery failed'),
  }).pipe(
    Effect.catchAll((error) =>
      Effect.succeed<LocalDiscoveryResult>({
        models: [],
        error: error.message,
        source: null,
        status: 'failure',
        diagnostics: ['lmstudio-discovery-unexpected-error'],
      }),
    ),
  )
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
          catch: (error) => error instanceof Error ? error : new Error('llama.cpp context enrichment failed'),
        }).pipe(
          Effect.orElseSucceed((): LocalDiscoveryResult => ({
            models,
            error: null,
            source: 'openai-v1-models',
            status: models.length > 0 ? 'success_non_empty' : 'success_empty',
            diagnostics: [],
          })),
        ),
      ),
      Effect.catchAll((error): Effect.Effect<LocalDiscoveryResult, never> =>
        Effect.succeed({
          models: [],
          error: error.message || 'discovery failed',
          source: null,
          status: 'failure',
          diagnostics: [],
        } satisfies LocalDiscoveryResult),
      ),
    )
  }

  const run: Effect.Effect<LocalDiscoveryResult, Error> = (() => {
    switch (provider.localDiscoveryStrategy) {
      case 'openai-models':
        return discoverOpenAIModels(baseUrl).pipe(
          Effect.map((models): LocalDiscoveryResult => ({
            models,
            error: null,
            source: 'openai-v1-models',
            status: models.length > 0 ? 'success_non_empty' : 'success_empty',
            diagnostics: [],
          })),
        )
      case 'ollama-hybrid':
        return discoverOllamaHybrid(baseUrl).pipe(
          Effect.map((models): LocalDiscoveryResult => ({
            models,
            error: null,
            source: 'ollama-api-tags',
            status: models.length > 0 ? 'success_non_empty' : 'success_empty',
            diagnostics: [],
          })),
        )
      case 'openai-models-best-effort':
        return discoverOpenAIModels(baseUrl).pipe(
          Effect.map((models): LocalDiscoveryResult => ({
            models,
            error: null,
            source: 'openai-v1-models',
            status: models.length > 0 ? 'success_non_empty' : 'success_empty',
            diagnostics: [],
          })),
        )
      default:
        return Effect.succeed<LocalDiscoveryResult>({
          models: [],
          error: null,
          source: null,
          status: 'success_empty',
          diagnostics: [],
        })
    }
  })()

  return run.pipe(
    Effect.catchAll((error): Effect.Effect<LocalDiscoveryResult, never> =>
      Effect.succeed({
        models: [],
        error: error.message || 'discovery failed',
        source: null,
        status: 'failure',
        diagnostics: [],
      } satisfies LocalDiscoveryResult),
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
