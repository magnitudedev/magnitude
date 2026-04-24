import type { ProviderDefinition } from './types'
import { compareProviderOrder } from './registry'

export interface ModelRecommendationRule {
  provider?: string | RegExp
  model: string | RegExp
  classes: readonly string[]
}

export interface RecommendationMatch {
  classes: Set<string>
}

function rule(
  model: string | RegExp,
  classes: ModelRecommendationRule['classes'],
  provider?: string | RegExp,
): ModelRecommendationRule {
  return { provider, model, classes }
}

const ALL_SLOTS = ['lead', 'worker'] as const
const SUBAGENT_SLOTS = ['worker'] as const

export const MODEL_RECOMMENDATION_RULES: ModelRecommendationRule[] = [
  // OpenRouter
  rule(/^glm-5\.1$/, [...ALL_SLOTS], 'openrouter'),
  rule(/^kimi-k2\.6$/, [...ALL_SLOTS], 'openrouter'),
  rule(/^deepseek-v4-pro$/, [...SUBAGENT_SLOTS], 'openrouter'),

  // Vercel
  rule(/^glm-5\.1$/, [...ALL_SLOTS], 'vercel'),
  rule(/^kimi-k2\.6$/, [...ALL_SLOTS], 'vercel'),
  rule(/^deepseek-v4-pro$/, [...SUBAGENT_SLOTS], 'vercel'),

  // Anthropic (direct provider only)
  rule(/^claude-opus-4[.-]7(-v1:0)?$/, ['lead'], 'anthropic'),
  rule(/^claude-sonnet-4[.-]6(-v1:0)?$/, [...SUBAGENT_SLOTS], 'anthropic'),
  rule(/^claude-haiku-4[.-]5(-v1:0)?$/, [...SUBAGENT_SLOTS], 'anthropic'),

  // OpenAI (direct provider only)
  rule(/^gpt-5\.5$/, [...ALL_SLOTS], 'openai'),

  // ZAI standard
  rule(/^glm-5\.1$/, [...ALL_SLOTS], 'zai'),

  // ZAI Coding Plan
  rule(/^glm-5\.1$/, [...ALL_SLOTS], 'zai-coding-plan'),

  // MiniMax (direct provider only)
  rule(/^MiniMax-M2\.7$/, [...ALL_SLOTS], 'minimax'),

  // Moonshot
  rule(/^kimi-k2\.6$/, [...ALL_SLOTS], 'moonshotai'),

  // Kimi-for-coding
  rule(/^k2p6$/, [...ALL_SLOTS], 'kimi-for-coding'),

  // Magnitude
  rule(/^glm-5\.1$/, ['lead'], 'magnitude'),
  rule(/^kimi-k2\.6$/, [...SUBAGENT_SLOTS], 'magnitude'),

  // Fireworks
  rule(/^accounts\/fireworks\/models\/glm-5p1$/, [...ALL_SLOTS], 'fireworks-ai'),
  rule(/^accounts\/fireworks\/models\/kimi-k2p6$/, [...ALL_SLOTS], 'fireworks-ai'),

  // Local intentionally omitted: recommendations depend on user-local inventory
]

function matchesRule(value: string, match: string | RegExp): boolean {
  return typeof match === 'string' ? value === match : match.test(value)
}

function matchesProvider(providerId: string, provider?: string | RegExp): boolean {
  if (!provider) return true
  return matchesRule(providerId, provider)
}

export function normalizeModelId(providerId: string, modelId: string): string {
  if (providerId === 'openrouter' || providerId === 'vercel') {
    return modelId.replace(/^[^/]+\//, '')
  }

  return modelId
}

export function getModelRecommendation(providerId: string, modelId: string): RecommendationMatch | null {
  const normalizedModelId = normalizeModelId(providerId, modelId)

  for (const rule of MODEL_RECOMMENDATION_RULES) {
    if (!matchesProvider(providerId, rule.provider)) continue
    if (!matchesRule(normalizedModelId, rule.model)) continue
    return { classes: new Set(rule.classes) }
  }

  return null
}

export function resolveRecommendedModelForClass(
  targetClass: string,
  providers: ProviderDefinition[],
  connectedProviderIds: Set<string>,
  options?: {
    preferredProviderId?: string
    isAllowedModel?: (providerId: string, modelId: string) => boolean
  },
): { providerId: string; modelId: string } | null {
  const connectedProviders = providers.filter(provider =>
    connectedProviderIds.has(provider.id) && provider.models.length > 0,
  )

  const sortedProviders = [...connectedProviders].sort((a, b) => {
    const aPreferred = a.id === options?.preferredProviderId ? -1 : 0
    const bPreferred = b.id === options?.preferredProviderId ? -1 : 0
    if (aPreferred !== bPreferred) return aPreferred - bPreferred
    return compareProviderOrder(a.id, b.id)
  })

  for (const rule of MODEL_RECOMMENDATION_RULES) {
    if (!rule.classes.includes(targetClass)) continue

    for (const provider of sortedProviders) {
      if (!matchesProvider(provider.id, rule.provider)) continue

      for (const model of provider.models) {
        if (options?.isAllowedModel && !options.isAllowedModel(provider.id, model.id)) continue

        const normalizedModelId = normalizeModelId(provider.id, model.id)
        if (matchesRule(normalizedModelId, rule.model)) {
          return { providerId: provider.id, modelId: model.id }
        }
      }
    }
  }

  return null
}

export function resolveRecommendedModel(
  slot: string,
  providers: ProviderDefinition[],
  connectedProviderIds: Set<string>,
  options?: {
    preferredProviderId?: string
    isAllowedModel?: (providerId: string, modelId: string) => boolean
  },
): { providerId: string; modelId: string } | null {
  return resolveRecommendedModelForClass(slot, providers, connectedProviderIds, options)
}