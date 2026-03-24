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

const PRIMARY = 'primary' as const
const SECONDARY = 'secondary' as const
const BROWSER = 'browser' as const

function rule(
  model: string | RegExp,
  classes: ModelRecommendationRule['classes'],
  provider?: string | RegExp,
): ModelRecommendationRule {
  return { provider, model, classes }
}

export const MODEL_RECOMMENDATION_RULES: ModelRecommendationRule[] = [
  // Primary — smartest/newest of each family
  rule(/^claude-opus-4[.-]6(-v1:0)?$/, [PRIMARY]),
  rule(/^gpt-5\.4$/, [PRIMARY, SECONDARY, BROWSER]),
  rule(/^gpt-5\.3-codex$/, [PRIMARY, SECONDARY, BROWSER]),
  rule(/^qwen3\.5-(397b-a17b|max-thinking|coder-next)$/, [PRIMARY, SECONDARY]),
  rule('gemini-3.1-pro-preview', [PRIMARY]),
  rule(/^MiniMax-M2\.5$/, [PRIMARY, SECONDARY]),
  rule(/^glm-5$/, [PRIMARY]),

  // Secondary — fast models
  rule(/^claude-sonnet-4[.-]6(-v1:0)?$/, [SECONDARY, BROWSER]),

  rule(/^gemini-3-flash-preview$/, [SECONDARY, BROWSER]),
  rule(/^gemini-3\.1-flash-lite-preview$/, [SECONDARY, BROWSER]),
  rule(/^qwen3\.5-27b$/, [SECONDARY, BROWSER]),
  rule(/^qwen3\.5-9b$/, [BROWSER]),
  rule(/^glm-5$/, [SECONDARY], 'zai'),

  // Browser
  rule(/^claude-haiku-4[.-]5(-v1:0)?$/, [BROWSER]),
  rule(/^gemini-3\.1-pro-preview$/, [BROWSER]),
  rule(/^kimi-k2\.5$/, [PRIMARY, SECONDARY, BROWSER]),
  rule(/^MiniMax-M2\.5$/, [BROWSER]),
  rule(/^glm-5$/, [BROWSER]),
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

export function resolveRecommendedModel<TSlot extends string>(
  slot: TSlot,
  providers: ProviderDefinition[],
  connectedProviderIds: Set<string>,
  options: {
    slotClassOf: (slot: TSlot) => string
    preferredProviderId?: string
    isAllowedModel?: (slot: TSlot, providerId: string, modelId: string) => boolean
  },
): { providerId: string; modelId: string } | null {
  const slotClass = options.slotClassOf(slot)
  return resolveRecommendedModelForClass(slotClass, providers, connectedProviderIds, {
    preferredProviderId: options.preferredProviderId,
    isAllowedModel: options.isAllowedModel
      ? (providerId, modelId) => options.isAllowedModel!(slot, providerId, modelId)
      : undefined,
  })
}