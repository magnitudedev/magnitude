import type {
  ModelSelection,
  ProviderAuthMethodStatus,
} from '@magnitudedev/agent'
import { isBrowserCompatible } from '@magnitudedev/providers'
import type { ProviderDefinition } from '@magnitudedev/providers'
import {
  compareProviderOrder,
  getModelRecommendation,
  resolveRecommendedModel,
} from '@magnitudedev/providers'
import { getDefaultModels } from './model-preferences'
import { pickLocalBrowserModel } from './local-browser-selection'

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
  selectingModelFor: string
  authStatusesByProviderId?: Map<string, ProviderAuthMethodStatus | null>
  detectedAuthTypeByProviderId?: Map<string, string | null>
}

interface FilterModelPickerItemsArgs {
  items: ModelPickerItem[]
  selectingModelFor: string
  showAllProviders: boolean
  showRecommendedOnly?: boolean
  search: string
}

interface ResolveSlotDefaultSelectionArgs {
  allProviders: ProviderDefinition[]
  connectedProviderIds: Set<string>
  slot: string
  preferredProviderId?: string | null
  detectedAuthTypeByProviderId?: Map<string, string | null>
}

interface ResolveWizardLocalDefaultModelIdArgs {
  providerId: string
  savedSlotModels: Record<string, ModelSelection | null>
  discoveredModels?: Array<{ id: string; name?: string }>
  rememberedModelIds?: string[]
}

const LOCAL_PROVIDER_IDS = new Set(['lmstudio', 'ollama', 'llama.cpp', 'openai-compatible-local'])

function normalizeModelId(value: string): string {
  return value.trim()
}

function normalizeForSearch(text: string): string {
  return text
    .toLowerCase()
    .replace(/(\d)[-_.](\d)/g, '$1.$2')  // keep version numbers together: 4-5 → 4.5
    .replace(/[-_]/g, ' ')                // word separators → space
}

function compareItemsForSlot(a: ModelPickerItem, b: ModelPickerItem, slot: string): number {
  const aRecommended = getModelRecommendation(a.providerId, a.modelId)?.classes.has(slot) ?? false
  const bRecommended = getModelRecommendation(b.providerId, b.modelId)?.classes.has(slot) ?? false
  if (aRecommended !== bRecommended) return Number(bRecommended) - Number(aRecommended)
  return a.modelName.localeCompare(b.modelName) || a.modelId.localeCompare(b.modelId)
}

export function buildModelPickerItems({
  allProviders,
  connectedProviderIds,
  selectingModelFor,
  authStatusesByProviderId,
  detectedAuthTypeByProviderId,
}: BuildModelPickerItemsArgs): ModelPickerItem[] {
  const items: ModelPickerItem[] = []

  for (const provider of allProviders) {
    const connected = connectedProviderIds.has(provider.id)

    const oauthOnlySet = provider.oauthOnlyModelIds ? new Set(provider.oauthOnlyModelIds) : null
    const detectedAuthType = detectedAuthTypeByProviderId?.get(provider.id)


    for (const model of provider.models) {
      if (selectingModelFor === 'browser' && !isBrowserCompatible(provider.id, model.id)) {
        continue
      }

      const recommendation = getModelRecommendation(provider.id, model.id)
      const recommended = recommendation?.classes.has(selectingModelFor) ?? false

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

export function resolveWizardLocalDefaultModelId({
  providerId,
  savedSlotModels,
  discoveredModels = [],
  rememberedModelIds = [],
}: ResolveWizardLocalDefaultModelIdArgs): string | null {
  for (const slot of Object.keys(savedSlotModels)) {
    const selection = savedSlotModels[slot]
    if (selection?.providerId === providerId && normalizeModelId(selection.modelId).length > 0) {
      return normalizeModelId(selection.modelId)
    }
  }

  const firstDiscovered = discoveredModels
    .map((m) => normalizeModelId(m.id))
    .find((id) => id.length > 0)
  if (firstDiscovered) return firstDiscovered

  const firstRemembered = rememberedModelIds
    .map(normalizeModelId)
    .find((id) => id.length > 0)
  if (firstRemembered) return firstRemembered

  return null
}

export function resolveSlotDefaultSelection({
  allProviders,
  connectedProviderIds,
  slot,
  preferredProviderId,
  detectedAuthTypeByProviderId,
}: ResolveSlotDefaultSelectionArgs): ModelSelection | null {
  const connectedProviders = allProviders.filter((provider) =>
    connectedProviderIds.has(provider.id),
  )

  // 1. If preferred provider is local-family, stay scoped to that provider inventory.
  const preferredProvider = preferredProviderId
    ? connectedProviders.find((provider) => provider.id === preferredProviderId)
    : null

  const connectedProvidersWithModels = connectedProviders.filter((provider) => provider.models.length > 0)

  const preferredFirst = [...connectedProvidersWithModels].sort((a, b) => {
    const aPreferred = a.id === preferredProviderId ? -1 : 0
    const bPreferred = b.id === preferredProviderId ? -1 : 0
    if (aPreferred !== bPreferred) return aPreferred - bPreferred
    return compareProviderOrder(a.id, b.id)
  })

  if (preferredProvider && LOCAL_PROVIDER_IDS.has(preferredProvider.id)) {
    const localFirstModel = slot === 'browser'
      ? pickLocalBrowserModel(preferredProvider.id, preferredProvider.models)
      : preferredProvider.models[0]

    if (localFirstModel) {
      return { providerId: preferredProvider.id, modelId: localFirstModel.id }
    }

    return null
  }

  // 2. Try hardcoded per-provider defaults first
  for (const provider of preferredFirst) {
    const isOAuth = detectedAuthTypeByProviderId?.get(provider.id) === 'oauth'
    const defaults = getDefaultModels(provider.id, isOAuth)
    if (!defaults) continue

    const defaultModelId = defaults[slot as keyof typeof defaults]
    if (defaultModelId && provider.models.some(model => model.id === defaultModelId)) {
      return { providerId: provider.id, modelId: defaultModelId }
    }
  }

  // 3. Fall back to recommendation rules
  const recommended = resolveRecommendedModel(slot, allProviders, connectedProviderIds, {
    preferredProviderId: preferredProviderId ?? undefined,
  })
  if (recommended) return recommended

  // 4. Fall back to browser-compatible model
  if (slot === 'browser') {
    for (const provider of preferredFirst) {
      const browserModel = provider.models.find(model => isBrowserCompatible(provider.id, model.id))
      if (browserModel) return { providerId: provider.id, modelId: browserModel.id }
    }
  }

  // 5. Fall back to first available model
  for (const provider of preferredFirst) {
    const firstModel = provider.models[0]
    if (firstModel) return { providerId: provider.id, modelId: firstModel.id }
  }

  return null
}