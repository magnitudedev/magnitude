import { useState, useCallback, useMemo } from 'react'
import type { KeyEvent } from '@opentui/core'
import type { ProviderDefinition } from '@magnitudedev/agent'
import { getLocalProviderConfig, getAuth } from '@magnitudedev/agent'

export interface ModelSelectItem {
  type: 'model'
  providerId: string
  providerName: string
  modelId: string
  modelName: string
}

interface ModelSelectNavigationState {
  selectedIndex: number
  setSelectedIndex: (index: number) => void
  items: ModelSelectItem[]
  handleKeyEvent: (key: KeyEvent) => boolean
}

/**
 * Manages keyboard navigation for the model select overlay.
 * Builds a flat list of selectable model items from all providers.
 */
export function useModelSelectNavigation(
  providers: ProviderDefinition[],
  onSelect: (providerId: string, modelId: string) => void,
  isActive: boolean,
): ModelSelectNavigationState {
  const [selectedIndex, setSelectedIndex] = useState(0)

  const items = useMemo(() => {
    const result: ModelSelectItem[] = []
    for (const provider of providers) {
      if (provider.id === 'local') {
        // Local provider has no static models — inject one from runtime config
        const localConfig = getLocalProviderConfig()
        if (localConfig.baseUrl && localConfig.modelId) {
          result.push({
            type: 'model',
            providerId: 'local',
            providerName: provider.name,
            modelId: localConfig.modelId,
            modelName: localConfig.baseUrl,
          })
        }
        continue
      }
      // Filter out OAuth-only models when user doesn't have OAuth auth
      const oauthOnly = provider.oauthOnlyModelIds
      const oauthOnlySet = oauthOnly ? new Set(oauthOnly) : null
      const isOAuth = oauthOnlySet ? getAuth(provider.id)?.type === 'oauth' : false

      for (const model of provider.models) {
        if (oauthOnlySet && !isOAuth && oauthOnlySet.has(model.id)) continue
        result.push({
          type: 'model',
          providerId: provider.id,
          providerName: provider.name,
          modelId: model.id,
          modelName: model.name,
        })
      }
    }
    return result
  }, [providers])

  const handleKeyEvent = useCallback((key: KeyEvent): boolean => {
    if (!isActive || items.length === 0) return false

    const isUp = key.name === 'up' && !key.ctrl && !key.meta && !key.option
    const isDown = key.name === 'down' && !key.ctrl && !key.meta && !key.option
    const isEnter = (key.name === 'return' || key.name === 'enter') &&
      !key.shift && !key.ctrl && !key.meta && !key.option

    if (isUp) {
      setSelectedIndex(prev => Math.max(0, prev - 1))
      return true
    }

    if (isDown) {
      setSelectedIndex(prev => Math.min(items.length - 1, prev + 1))
      return true
    }

    if (isEnter) {
      const item = items[selectedIndex]
      if (item) {
        onSelect(item.providerId, item.modelId)
      }
      return true
    }

    return false
  }, [isActive, items, selectedIndex, onSelect])

  return {
    selectedIndex,
    setSelectedIndex,
    items,
    handleKeyEvent,
  }
}
