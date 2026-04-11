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

const ALL_SLOTS = ['lead', 'explorer', 'planner', 'builder', 'reviewer', 'debugger', 'browser'] as const
const SUBAGENT_SLOTS = ['explorer', 'planner', 'builder', 'reviewer', 'debugger'] as const
const NON_BROWSER_SLOTS = ['lead', 'explorer', 'planner', 'builder', 'reviewer', 'debugger'] as const
const SUBAGENT_AND_BROWSER = ['explorer', 'planner', 'builder', 'reviewer', 'debugger', 'browser'] as const

export const MODEL_RECOMMENDATION_RULES: ModelRecommendationRule[] = [
  // Anthropic
  rule(/^claude-opus-4[.-]6(-v1:0)?$/, ['lead']),
  rule(/^claude-sonnet-4[.-]6(-v1:0)?$/, [...SUBAGENT_AND_BROWSER]),
  rule(/^claude-haiku-4[.-]5(-v1:0)?$/, ['browser']),

  // OpenAI
  rule(/^gpt-5\.4$/, ['lead']),
  rule(/^gpt-5\.3-codex$/, [...SUBAGENT_AND_BROWSER]),

  // Google
  rule(/^gemini-3\.1-pro-preview$/, ['lead']),
  rule(/^gemini-3-flash-preview$/, [...SUBAGENT_AND_BROWSER]),

  // Qwen
  rule(/^qwen3\.5-(397b-a17b|max-thinking|coder-next)$/, [...NON_BROWSER_SLOTS]),
  rule(/^qwen3\.5-27b$/, [...SUBAGENT_AND_BROWSER]),
  rule(/^qwen3\.5-9b$/, ['browser']),

  // ZAI standard
  rule(/^glm-4\.7$/, [...ALL_SLOTS], 'zai'),
  rule(/^glm-5$/, ['lead'], 'zai'),
  rule(/^glm-4\.7-flash$/, ['browser'], 'zai'),

  // ZAI Coding Plan
  rule(/^glm-4\.7$/, [...ALL_SLOTS], 'zai-coding-plan'),
  rule(/^glm-5\.1$/, ['lead'], 'zai-coding-plan'),
  rule(/^glm-4\.7-flash$/, ['browser'], 'zai-coding-plan'),

  // MiniMax
  rule(/^MiniMax-M2\.7$/, [...ALL_SLOTS]),

  // Moonshot
  rule(/^kimi-k2\.5$/, [...ALL_SLOTS]),
  rule(/^k2p5$/, [...ALL_SLOTS], 'kimi-for-coding'),

  // Fireworks
  rule(/^accounts\/fireworks\/models\/glm-5p1$/, [...ALL_SLOTS], 'fireworks'),
  rule(/^accounts\/fireworks\/routers\/kimi-k2p5-turbo$/, [...ALL_SLOTS], 'fireworks'),

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

  if (providerId === 'amazon-bedrock') {
    return modelId.replace(/^(?:(?:[a-z]{2}|global)\.)?anthropic\./i, '')
  }

  if (providerId === 'google-vertex-anthropic') {
    return modelId.replace(/@.+$/, '')
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