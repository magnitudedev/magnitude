import { memo, useEffect, useMemo, useState } from 'react'
import { type KeyEvent, TextAttributes } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import { useTheme } from '../hooks/use-theme'
import { Button } from './button'
import { SingleLineInput } from './single-line-input'
import { BOX_CHARS } from '../utils/ui-constants'

interface LocalProviderPageProps {
  providerName: string
  endpoint: string
  endpointPlaceholder: string
  discoveredModels: Array<{ id: string; name?: string }>
  manualModelIds: string[]
  lastDiscoveryError?: string | null
  optionalApiKeyValue?: string
  showOptionalApiKey?: boolean
  onSaveEndpoint: (url: string) => void
  onRefreshModels: () => void
  onAddManualModel: (modelId: string) => void
  onRemoveManualModel: (modelId: string) => void
  onSaveOptionalApiKey?: (apiKey: string) => void
  endpointSaveLabel?: string
  showProviderTitle?: boolean
  showEndpointSaveButton?: boolean
  showApiKeySaveButton?: boolean
}

export const LocalProviderPage = memo(function LocalProviderPage({
  providerName,
  endpoint,
  endpointPlaceholder,
  discoveredModels,
  manualModelIds,
  lastDiscoveryError,
  optionalApiKeyValue = '',
  showOptionalApiKey = false,
  onSaveEndpoint,
  onRefreshModels,
  onAddManualModel,
  onRemoveManualModel,
  onSaveOptionalApiKey,
  endpointSaveLabel = 'Save endpoint',
  showProviderTitle = true,
  showEndpointSaveButton = true,
  showApiKeySaveButton = true,
}: LocalProviderPageProps) {
  const theme = useTheme()
  const [endpointDraft, setEndpointDraft] = useState(endpoint)
  const [manualDraft, setManualDraft] = useState('')
  const [apiKeyDraft, setApiKeyDraft] = useState(optionalApiKeyValue)
  const [activeField, setActiveField] = useState<'endpoint' | 'manual' | 'apiKey' | null>(null)
  const [saveEndpointHovered, setSaveEndpointHovered] = useState(false)
  const [addHovered, setAddHovered] = useState(false)
  const [refreshHovered, setRefreshHovered] = useState(false)
  const [saveApiKeyHovered, setSaveApiKeyHovered] = useState(false)
  const [hoveredRemoveModelId, setHoveredRemoveModelId] = useState<string | null>(null)
  const [endpointHovered, setEndpointHovered] = useState(false)
  const [manualHovered, setManualHovered] = useState(false)
  const [apiKeyHovered, setApiKeyHovered] = useState(false)

  useEffect(() => {
    setEndpointDraft(endpoint)
  }, [endpoint])

  useEffect(() => {
    setApiKeyDraft(optionalApiKeyValue)
  }, [optionalApiKeyValue])

  const discoveredCount = discoveredModels.length
  const hasEndpoint = endpoint.trim().length > 0

  const statusLine = useMemo(() => {
    if (!hasEndpoint) return 'Set an endpoint to discover models'
    if (discoveredCount > 0) return `Discovered ${discoveredCount} model${discoveredCount === 1 ? '' : 's'}`
    if (lastDiscoveryError) return "Couldn't refresh models right now"
    return 'No discovered models yet'
  }, [hasEndpoint, discoveredCount, lastDiscoveryError])

  const handleAddManual = () => {
    const trimmed = manualDraft.trim()
    if (!trimmed) return
    onAddManualModel(trimmed)
    setManualDraft('')
  }

  useKeyboard((key: KeyEvent) => {
    if (activeField === null) return

    if (key.name === 'up' && activeField === 'manual') {
      key.preventDefault?.()
      setActiveField('endpoint')
      return
    }

    if (key.name === 'up' && activeField === 'apiKey') {
      key.preventDefault?.()
      setActiveField('manual')
      return
    }

    if (key.name === 'down' && activeField === 'endpoint') {
      key.preventDefault?.()
      setActiveField('manual')
      return
    }

    if (key.name === 'down' && activeField === 'manual' && showOptionalApiKey) {
      key.preventDefault?.()
      setActiveField('apiKey')
      return
    }

    if (key.name === 'escape') {
      setActiveField(null)
      return
    }

    if (key.name !== 'return' && key.name !== 'enter') return
    key.preventDefault?.()

    if (activeField === 'endpoint') {
      onSaveEndpoint(endpointDraft)
      return
    }

    if (activeField === 'apiKey') {
      onSaveOptionalApiKey?.(apiKeyDraft)
      return
    }

    handleAddManual()
  })

  return (
    <box style={{ flexDirection: 'column' }}>
      {showProviderTitle && (
        <box style={{ paddingBottom: 1 }}>
          <text style={{ fg: theme.primary }}>
            <span attributes={TextAttributes.BOLD}>{providerName}</span>
          </text>
        </box>
      )}

      <box style={{ paddingBottom: 1, flexDirection: 'column' }}>
        <text style={{ fg: theme.muted }}>Endpoint</text>
        <box style={{ flexDirection: 'row', gap: 1, alignItems: 'center' }}>
          <box
            onMouseDown={() => setActiveField('endpoint')}
            onMouseOver={() => setEndpointHovered(true)}
            onMouseOut={() => setEndpointHovered(false)}
            style={{
              flexGrow: 1,
              borderStyle: 'single',
              borderColor: activeField === 'endpoint' || endpointHovered ? theme.primary : theme.border,
              customBorderChars: BOX_CHARS,
              paddingLeft: 1,
            }}
          >
            <SingleLineInput
              value={endpointDraft}
              onChange={setEndpointDraft}
              placeholder={endpointPlaceholder}
              focused={activeField === 'endpoint'}
            />
          </box>
          {showEndpointSaveButton && (
            <Button
              onClick={() => onSaveEndpoint(endpointDraft)}
              onMouseOver={() => setSaveEndpointHovered(true)}
              onMouseOut={() => setSaveEndpointHovered(false)}
            >
              <box
                style={{
                  borderStyle: 'single',
                  borderColor: saveEndpointHovered ? theme.foreground : theme.border,
                  customBorderChars: BOX_CHARS,
                  paddingLeft: 1,
                  paddingRight: 1,
                }}
              >
                <text style={{ fg: saveEndpointHovered ? theme.foreground : theme.primary }}>{endpointSaveLabel}</text>
              </box>
            </Button>
          )}
        </box>
      </box>

      <box style={{ paddingBottom: 1, flexDirection: 'row', gap: 1, alignItems: 'center' }}>
        <text style={{ fg: theme.muted }}>Models</text>
        <text style={{ fg: theme.muted }}>· {statusLine}</text>
        <Button onClick={onRefreshModels} onMouseOver={() => setRefreshHovered(true)} onMouseOut={() => setRefreshHovered(false)}>
          <text style={{ fg: refreshHovered ? theme.foreground : theme.muted }}>[Refresh]</text>
        </Button>
      </box>

      {discoveredCount > 0 && (
        <box style={{ paddingBottom: 1, flexDirection: 'column' }}>
          <text style={{ fg: theme.muted }}>
            <span attributes={TextAttributes.BOLD}>Discovered models</span>
          </text>
          {discoveredModels.slice(0, 20).map((model) => (
            <text key={model.id} style={{ fg: theme.foreground }}>
              {'  · '}{model.name ?? model.id}
              <span attributes={TextAttributes.DIM}>{' — '}{model.id}</span>
            </text>
          ))}
        </box>
      )}

      <box style={{ paddingBottom: 1, flexDirection: 'column' }}>
        <text style={{ fg: theme.muted }}>
          <span attributes={TextAttributes.BOLD}>Manual models</span>
        </text>
        <box style={{ flexDirection: 'row', gap: 1, paddingBottom: 1, alignItems: 'center' }}>
          <box
            onMouseDown={() => setActiveField('manual')}
            onMouseOver={() => setManualHovered(true)}
            onMouseOut={() => setManualHovered(false)}
            style={{
              flexGrow: 1,
              borderStyle: 'single',
              borderColor: activeField === 'manual' || manualHovered ? theme.primary : theme.border,
              customBorderChars: BOX_CHARS,
              paddingLeft: 1,
            }}
          >
            <SingleLineInput
              value={manualDraft}
              onChange={setManualDraft}
              placeholder="Add model ID"
              focused={activeField === 'manual'}
            />
          </box>
          <Button
            onClick={handleAddManual}
            onMouseOver={() => setAddHovered(true)}
            onMouseOut={() => setAddHovered(false)}
          >
            <box
              style={{
                borderStyle: 'single',
                borderColor: addHovered ? theme.foreground : theme.border,
                customBorderChars: BOX_CHARS,
                paddingLeft: 1,
                paddingRight: 1,
              }}
            >
              <text style={{ fg: addHovered ? theme.foreground : theme.primary }}>Add</text>
            </box>
          </Button>
        </box>
        {manualModelIds.length === 0 ? (
          <text style={{ fg: theme.muted }}>  None yet</text>
        ) : (
          manualModelIds.slice(0, 20).map((id) => (
            <box key={id} style={{ flexDirection: 'row' }}>
              <text style={{ fg: theme.foreground, flexGrow: 1 }}>{'  · '}{id}</text>
              <Button
                onClick={() => onRemoveManualModel(id)}
                onMouseOver={() => setHoveredRemoveModelId(id)}
                onMouseOut={() => setHoveredRemoveModelId((current) => (current === id ? null : current))}
              >
                <text style={{ fg: hoveredRemoveModelId === id ? theme.foreground : theme.muted }}>[Remove]</text>
              </Button>
            </box>
          ))
        )}
      </box>

      {showOptionalApiKey && (
        <box style={{ paddingBottom: 1, flexDirection: 'column' }}>
          <text style={{ fg: theme.muted }}>Optional API key</text>
          <box style={{ flexDirection: 'row', gap: 1, alignItems: 'center' }}>
            <box
              onMouseDown={() => setActiveField('apiKey')}
              onMouseOver={() => setApiKeyHovered(true)}
              onMouseOut={() => setApiKeyHovered(false)}
              style={{
                flexGrow: 1,
                borderStyle: 'single',
                borderColor: activeField === 'apiKey' || apiKeyHovered ? theme.primary : theme.border,
                customBorderChars: BOX_CHARS,
                paddingLeft: 1,
              }}
            >
              <SingleLineInput
                value={apiKeyDraft}
                onChange={setApiKeyDraft}
                placeholder="Leave empty if not required"
                focused={activeField === 'apiKey'}
                mask
              />
            </box>
            {showApiKeySaveButton && (
              <Button
                onClick={() => onSaveOptionalApiKey?.(apiKeyDraft)}
                onMouseOver={() => setSaveApiKeyHovered(true)}
                onMouseOut={() => setSaveApiKeyHovered(false)}
              >
                <box
                  style={{
                    borderStyle: 'single',
                    borderColor: saveApiKeyHovered ? theme.foreground : theme.border,
                    customBorderChars: BOX_CHARS,
                    paddingLeft: 1,
                    paddingRight: 1,
                  }}
                >
                  <text style={{ fg: saveApiKeyHovered ? theme.foreground : theme.primary }}>Save key</text>
                </box>
              </Button>
            )}
          </box>
        </box>
      )}

      {lastDiscoveryError && (
        <text style={{ fg: theme.muted }}>
          Note: {lastDiscoveryError}
        </text>
      )}
    </box>
  )
})
