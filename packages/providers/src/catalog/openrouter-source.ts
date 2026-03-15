import { Effect } from 'effect'
import type { ModelDefinition } from '../types'
import type { OpenRouterResponse } from './types'

const API_URL = 'https://openrouter.ai/api/v1/models'
const FETCH_TIMEOUT_MS = 10_000

function parsePrice(value?: string): number {
  if (!value) return 0
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed * 1_000_000 : 0
}

function inferFamily(id: string): string {
  if (id.startsWith('anthropic/')) return 'claude'
  if (id.startsWith('openai/')) return 'gpt'
  if (id.startsWith('google/')) return 'gemini'
  if (id.startsWith('qwen/')) return 'qwen'
  if (id.startsWith('moonshotai/')) return 'kimi'
  if (id.startsWith('deepseek/')) return 'deepseek'
  if (id.startsWith('minimax/')) return 'minimax'
  if (id.startsWith('z-ai/') || id.startsWith('zai/')) return 'glm'
  if (id.startsWith('x-ai/') || id.startsWith('xai/')) return 'grok'
  if (id.startsWith('mistralai/') || id.startsWith('mistral/')) return 'mistral'
  return id.split('/')[0] ?? 'unknown'
}

function toIsoDate(created: number): string {
  return new Date(created * 1000).toISOString()
}

export const fetchOpenRouterModels: Effect.Effect<OpenRouterResponse, Error> = Effect.tryPromise({
  try: async () => {
    const response = await fetch(API_URL, {
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
      headers: { 'User-Agent': 'magnitude-agent' },
    })
    if (!response.ok) {
      throw new Error(`OpenRouter API returned ${response.status}`)
    }
    return await response.json() as OpenRouterResponse
  },
  catch: (error) => (error instanceof Error ? error : new Error(String(error))),
})

export function normalizeOpenRouterModels(data: OpenRouterResponse): ModelDefinition[] {
  const now = Date.now()

  return data.data.map((model): ModelDefinition => {
    const expiration = model.expiration_date ? Date.parse(model.expiration_date) : Number.NaN
    const inputModalities = model.architecture?.input_modalities ?? []
    const supportedParameters = model.supported_parameters ?? []

    return {
      id: model.id,
      name: model.name,
      contextWindow: model.context_length,
      maxOutputTokens: model.top_provider?.max_completion_tokens ?? undefined,
      supportsToolCalls: supportedParameters.includes('tools'),
      supportsReasoning: supportedParameters.includes('reasoning'),
      supportsVision: inputModalities.includes('image'),
      description: model.description,
      cost: {
        input: parsePrice(model.pricing.prompt),
        output: parsePrice(model.pricing.completion),
        cache_read: model.pricing.input_cache_read ? parsePrice(model.pricing.input_cache_read) : undefined,
        cache_write: model.pricing.input_cache_write ? parsePrice(model.pricing.input_cache_write) : undefined,
      },
      family: inferFamily(model.id),
      releaseDate: toIsoDate(model.created),
      status: (!Number.isNaN(expiration) && expiration < now) || model.deprecated ? 'deprecated' : undefined,
      discovery: {
        primarySource: 'openrouter-api',
      },
    }
  }).sort((a, b) => b.releaseDate.localeCompare(a.releaseDate))
}