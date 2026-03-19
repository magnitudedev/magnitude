import type {
  ModelSelection,
  ProviderAuthMethodStatus,
} from '@magnitudedev/agent'
import { isBrowserCompatible } from '@magnitudedev/providers'
import type { ModelSlot, ProviderDefinition } from '@magnitudedev/providers'
import {
  compareProviderOrder,
  getModelRecommendation,
  resolveRecommendedModel,
} from '@magnitudedev/providers'
import { getDefaultModels } from './model-preferences'

export interface ModelPickerItem {
  type: 'model'
  providerId: string
  providerName: string
  connected: boolean
  modelId: string
  modelName: string
  recommended: boolean
  selectable: boolean
}

interface BuildModelPickerItemsArgs {
  allProviders: ProviderDefinition[]
  connectedProviderIds: Set<string>
  selectingModelFor: ModelSlot
  localProviderConfig?: { baseUrl?: string | null; modelId?: string | null } | null
  authStatusesByProviderId?: Map<string, ProviderAuthMethodStatus | null>
  detectedAuthTypeByProviderId?: Map<string, string | null>
}

interface FilterModelPickerItemsArgs {
  items: ModelPickerItem[]
  selectingModelFor: ModelSlot
  showAllProviders: boolean
  showRecommendedOnly?: boolean
  search: string
}

interface ResolveSlotDefaultSelectionArgs {
  allProviders: ProviderDefinition[]
  connectedProviderIds: Set<string>
  slot: ModelSlot
  preferredProviderId?: string | null
  detectedAuthTypeByProviderId?: Map<string, string | null>
}

function formatLocalModelName(localProviderConfig?: { baseUrl?: string | null; modelId?: string | null } | null): string {
  return localProviderConfig?.baseUrl?.trim() || 'Local model'
}

function normalizeForSearch(text: string): string {
  return text
    .toLowerCase()
    .replace(/(\d)[-_.](\d)/g, '$1.$2')  // keep version numbers together: 4-5 → 4.5
    .replace(/[-_]/g, ' ')                // word separators → space
}

function compareItemsForSlot(a: ModelPickerItem, b: ModelPickerItem, slot: ModelSlot): number {
  const aRecommended = getModelRecommendation(a.providerId, a.modelId)?.slots.has(slot) ?? false
  const bRecommended = getModelRecommendation(b.providerId, b.modelId)?.slots.has(slot) ?? false
  if (aRecommended !== bRecommended) return Number(bRecommended) - Number(aRecommended)
  return a.modelName.localeCompare(b.modelName) || a.modelId.localeCompare(b.modelId)
}

export function buildModelPickerItems({
  allProviders,
  connectedProviderIds,
  selectingModelFor,
  localProviderConfig,
  authStatusesByProviderId,
  detectedAuthTypeByProviderId,
}: BuildModelPickerItemsArgs): ModelPickerItem[] {
  const items: ModelPickerItem[] = []

  for (const provider of allProviders) {
    const connected = connectedProviderIds.has(provider.id)

    if (provider.id === 'local') {
      if (localProviderConfig?.modelId?.trim()) {
        const modelId = localProviderConfig.modelId.trim()
        items.push({
          type: 'model',
          providerId: provider.id,
          providerName: provider.name,
          connected,
          modelId,
          modelName: formatLocalModelName(localProviderConfig),
          recommended: false,
          selectable: connected,
        })
      }
      continue
    }

    const oauthOnlySet = provider.oauthOnlyModelIds ? new Set(provider.oauthOnlyModelIds) : null
    const detectedAuthType = detectedAuthTypeByProviderId?.get(provider.id)


    for (const model of provider.models) {
      if (selectingModelFor === 'browser' && !isBrowserCompatible(provider.id, model.id)) {
        continue
      }

      const recommendation = getModelRecommendation(provider.id, model.id)
      const recommended = recommendation?.slots.has(selectingModelFor) ?? false

      let selectable = connected

      if (connected) {
        if (oauthOnlySet?.has(model.id) && detectedAuthType !== 'oauth') {
          selectable = false
        } else if (model.status === 'deprecated') {
          selectable = false
        }
      }

      items.push({
        type: 'model',
        providerId: provider.id,
        providerName: provider.name,
        connected,
        modelId: model.id,
        modelName: model.name,
        recommended,
        selectable,
      })
    }
  }

  return items.sort((a, b) => {
    if (a.connected !== b.connected) return Number(b.connected) - Number(a.connected)
    if (a.providerId !== b.providerId) return compareProviderOrder(a.providerId, b.providerId)
    return compareItemsForSlot(a, b, selectingModelFor)
  })
}

export function filterModelPickerItems({
  items,
  selectingModelFor,
  showAllProviders,
  showRecommendedOnly,
  search,
}: FilterModelPickerItemsArgs): ModelPickerItem[] {
  const terms = normalizeForSearch(search).trim().split(/\s+/).filter(Boolean)

  return items.filter((item) => {
    if (!showAllProviders && !item.connected) return false
    if (showRecommendedOnly && !item.recommended) return false
    if (terms.length === 0) return true

    const haystack = normalizeForSearch(
      `${item.providerName} ${item.modelName} ${item.modelId}`
    )
    return terms.every(term => haystack.includes(term))
  })
}

export function resolveSlotDefaultSelection({
  allProviders,
  connectedProviderIds,
  slot,
  preferredProviderId,
  detectedAuthTypeByProviderId,
}: ResolveSlotDefaultSelectionArgs): ModelSelection | null {
  const connectedProviders = allProviders.filter(provider =>
    connectedProviderIds.has(provider.id) && provider.models.length > 0,
  )

  const preferredFirst = [...connectedProviders].sort((a, b) => {
    const aPreferred = a.id === preferredProviderId ? -1 : 0
    const bPreferred = b.id === preferredProviderId ? -1 : 0
    if (aPreferred !== bPreferred) return aPreferred - bPreferred
    return compareProviderOrder(a.id, b.id)
  })

  // 1. Try hardcoded per-provider defaults first
  for (const provider of preferredFirst) {
    const isOAuth = detectedAuthTypeByProviderId?.get(provider.id) === 'oauth'
    const defaults = getDefaultModels(provider.id, isOAuth)
    const defaultModelId = defaults[slot]
    if (defaultModelId && provider.models.some(model => model.id === defaultModelId)) {
      return { providerId: provider.id, modelId: defaultModelId }
    }
  }

  // 2. Fall back to recommendation rules
  const recommended = resolveRecommendedModel(slot, allProviders, connectedProviderIds, {
    preferredProviderId: preferredProviderId ?? undefined,
  })
  if (recommended) return recommended

  // 3. Fall back to browser-compatible model
  if (slot === 'browser') {
    for (const provider of preferredFirst) {
      const browserModel = provider.models.find(model => isBrowserCompatible(provider.id, model.id))
      if (browserModel) return { providerId: provider.id, modelId: browserModel.id }
    }
  }

  // 4. Fall back to first available model
  for (const provider of preferredFirst) {
    const firstModel = provider.models[0]
    if (firstModel) return { providerId: provider.id, modelId: firstModel.id }
  }

  return null
}