import { Effect } from 'effect'
import type { ProviderModel } from '../model/model'
import type { ModelsDevResponse } from './types'
import { tryResolveCanonicalModelId } from '../registry'

const API_URL = 'https://models.dev/api.json'
const FETCH_TIMEOUT_MS = 10_000

function parseModelsDevStatus(_status?: string): undefined {
  return undefined
}

export const fetchModelsDevData: Effect.Effect<ModelsDevResponse, Error> = Effect.tryPromise({
  try: async () => {
    const response = await fetch(API_URL, {
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
      headers: { 'User-Agent': 'magnitude-agent' },
    })
    if (!response.ok) {
      throw new Error(`models.dev API returned ${response.status}`)
    }
    return await response.json() as ModelsDevResponse
  },
  catch: (error) => (error instanceof Error ? error : new Error(String(error))),
})

export function normalizeModelsDevProvider(providerId: string, data: ModelsDevResponse): ProviderModel[] {
  const providerData = data[providerId]
  if (!providerData?.models) return []

  const models = Object.values(providerData.models).flatMap((model): ProviderModel[] => {
    if (!model.limit?.context || !model.cost || !model.family) return []

    return [{
      id: model.id,
      providerId,
      providerName: providerData.name ?? providerId,
      modelId: tryResolveCanonicalModelId(model.id),
      name: model.name,
      contextWindow: model.limit.context,
      maxContextTokens: null,
      maxOutputTokens: model.limit.output ?? null,
      supportsToolCalls: model.tool_call,
      supportsReasoning: model.reasoning,
      supportsVision: model.modalities?.input?.includes('image') ?? false,
      costs: model.cost ? {
        inputPerM: model.cost.input,
        outputPerM: model.cost.output,
        cacheReadPerM: model.cost.cache_read ?? null,
        cacheWritePerM: model.cost.cache_write ?? null,
      } : null,
      releaseDate: model.release_date ?? '1970-01-01T00:00:00.000Z',
      discovery: {
        primarySource: 'models.dev',
      },
    }]
  })

  models.sort((a, b) => (b.releaseDate ?? '').localeCompare(a.releaseDate ?? ''))
  return models
}
