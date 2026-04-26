import { Effect } from 'effect'
import type { ModelDefinition } from '../types'
import type { ModelsDevResponse } from './types'

const API_URL = 'https://models.dev/api.json'
const FETCH_TIMEOUT_MS = 10_000

function parseModelsDevStatus(status?: string): ModelDefinition['status'] | undefined {
  if (status === 'alpha' || status === 'beta' || status === 'deprecated') return status
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

export function normalizeModelsDevProvider(providerId: string, data: ModelsDevResponse): ModelDefinition[] {
  const providerData = data[providerId]
  if (!providerData?.models) return []

  const models = Object.values(providerData.models).flatMap((model): ModelDefinition[] => {
    if (!model.limit?.context || !model.cost || !model.family) return []

    return [{
      id: model.id,
      name: model.name,
      contextWindow: model.limit.context,
      maxOutputTokens: model.limit.output,
      supportsToolCalls: model.tool_call,
      supportsReasoning: model.reasoning,
      supportsVision: model.modalities?.input?.includes('image') ?? false,
      cost: model.cost,
      family: model.family,
      releaseDate: model.release_date ?? '1970-01-01T00:00:00.000Z',
      status: parseModelsDevStatus(model.status),
      discovery: {
        primarySource: 'models.dev',
      },
    }]
  })

  models.sort((a, b) => b.releaseDate.localeCompare(a.releaseDate))
  return models
}