import { memo, useState, useCallback } from 'react'
import { TextAttributes, type KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import { useTheme } from '../hooks/use-theme'
import { Button } from './button'
import { SingleLineInput } from './single-line-input'
import { WizardHeader, type WizardMode } from './wizard-header'
import { BOX_CHARS } from '../utils/ui-constants'
import type { ProviderDefinition } from '@magnitudedev/agent'

type FieldName = 'url' | 'model' | 'apiKey'
const FIELDS: FieldName[] = ['url', 'model', 'apiKey']

const LOCAL_PROVIDER_IDS = new Set(['lmstudio', 'ollama', 'llama.cpp', 'openai-compatible-local'])

interface ProviderEndpointOverlayProps {
  provider: ProviderDefinition
  initialOptions?: { baseUrl?: string | null; modelId?: string | null }
  onSubmit: (config: { providerId: string; url: string; modelId?: string; apiKey?: string }) => void
  onCancel: () => void
  wizardMode?: WizardMode
}

export const ProviderEndpointOverlay = memo(function ProviderEndpointOverlay({
  provider,
  initialOptions,
  onSubmit,
  onCancel,
  wizardMode,
}: ProviderEndpointOverlayProps) {
  const theme = useTheme()
  const [focusedField, setFocusedField] = useState<FieldName>('url')
  const [url, setUrl] = useState(initialOptions?.baseUrl ?? provider.defaultBaseUrl ?? '')
  const [model, setModel] = useState(initialOptions?.modelId ?? '')
  const [apiKey, setApiKey] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [backHovered, setBackHovered] = useState(false)

  const isLocalProvider = LOCAL_PROVIDER_IDS.has(provider.id)

  const handleSubmit = useCallback(() => {
    let trimmedUrl = url.trim().replace(/[\x00-\x1f\x7f]/g, '')
    if (!trimmedUrl) {
      setError('Base URL is required')
      setFocusedField('url')
      return
    }
    if (!trimmedUrl.startsWith('http://') && !trimmedUrl.startsWith('https://')) {
      trimmedUrl = `http://${trimmedUrl}`
    }

    setError(null)
    const trimmedModel = model.trim()
    const trimmedKey = apiKey.trim()
    onSubmit({
      providerId: provider.id,
      url: trimmedUrl,
      ...(trimmedModel ? { modelId: trimmedModel } : {}),
      ...(trimmedKey ? { apiKey: trimmedKey } : {}),
    })
  }, [url, model, apiKey, provider.id, onSubmit])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (key.name === 'escape') {
      key.preventDefault()
      wizardMode?.onSkip?.() ?? onCancel()
      return
    }

    if (key.name === 'tab' || key.name === 'up' || key.name === 'down') {
      key.preventDefault()
      const current = FIELDS.indexOf(focusedField)
      const prev = key.name === 'up' || (key.name === 'tab' && key.shift)
      const nextIndex = prev ? (current - 1 + FIELDS.length) % FIELDS.length : (current + 1) % FIELDS.length
      setFocusedField(FIELDS[nextIndex])
      setError(null)
      return
    }

    if ((key.name === 'return' || key.name === 'enter') && !key.shift) {
      key.preventDefault()
      handleSubmit()
      return
    }

    if (key.name === 'b' && !key.ctrl && !key.meta && !key.option && !key.shift && wizardMode?.onBack) {
      key.preventDefault()
      wizardMode.onBack()
      return
    }

    if (!key.defaultPrevented) key.preventDefault()
  }, [wizardMode, onCancel, focusedField, handleSubmit]))

  return (
    <box style={{ flexDirection: 'column', height: '100%' }}>
      {wizardMode ? (
        <WizardHeader stepLabel={wizardMode.stepLabel} subtitle={wizardMode.subtitle} onSkip={wizardMode.onSkip} theme={theme} />
      ) : (
        <>
          <box style={{ flexDirection: 'row', paddingLeft: 2, paddingRight: 2, paddingTop: 1, paddingBottom: 1, flexShrink: 0 }}>
            <text style={{ fg: theme.primary, flexGrow: 1 }}>
              <span attributes={TextAttributes.BOLD}>Connect {provider.name}</span>
            </text>
            <Button onClick={onCancel}>
              <text style={{ fg: theme.muted }} attributes={TextAttributes.UNDERLINE}>Cancel</text>
            </Button>
          </box>
          <box style={{ paddingLeft: 1, paddingRight: 1, flexShrink: 0 }}>
            <text style={{ fg: theme.border }}>{'─'.repeat(80)}</text>
          </box>
        </>
      )}

      <box style={{ paddingLeft: 2, paddingRight: 2, paddingTop: 1, flexGrow: 1, flexDirection: 'column' }}>
        {isLocalProvider && (
          <box style={{ paddingBottom: 1 }}>
            <text style={{ fg: theme.muted }}>
              Endpoint is optional here. Save to refresh available models; you can also add model IDs manually.
            </text>
          </box>
        )}
        <Field label="Base URL:" value={url} placeholder={provider.defaultBaseUrl ?? 'http://localhost:8000/v1'} focused={focusedField === 'url'} theme={theme} onFocus={() => setFocusedField('url')} onChange={(v) => { setUrl(v); setError(null) }} />
        <Field label={isLocalProvider ? 'Manual model ID (optional):' : 'Seed model (optional):'} value={model} placeholder="Model ID" focused={focusedField === 'model'} theme={theme} onFocus={() => setFocusedField('model')} onChange={(v) => { setModel(v); setError(null) }} />
        <Field label="API Key (optional):" value={apiKey} placeholder="Leave empty if not required" focused={focusedField === 'apiKey'} theme={theme} mask onFocus={() => setFocusedField('apiKey')} onChange={(v) => { setApiKey(v); setError(null) }} />
        {error && <text style={{ fg: theme.error }}>{error}</text>}
      </box>

      {wizardMode?.onBack && (
        <box style={{ paddingLeft: 2, paddingBottom: 1, flexShrink: 0 }}>
          <Button onClick={wizardMode.onBack} onMouseOver={() => setBackHovered(true)} onMouseOut={() => setBackHovered(false)}>
            <box style={{ borderStyle: 'single', borderColor: backHovered ? theme.primary : theme.border, customBorderChars: BOX_CHARS, paddingLeft: 1, paddingRight: 1 }}>
              <text style={{ fg: backHovered ? theme.primary : theme.muted }}>← Back (B)</text>
            </box>
          </Button>
        </box>
      )}
    </box>
  )
})

function Field({
  label,
  value,
  placeholder,
  focused,
  theme,
  mask,
  onFocus,
  onChange,
}: {
  label: string
  value: string
  placeholder: string
  focused: boolean
  theme: Record<string, any>
  mask?: boolean
  onFocus?: () => void
  onChange: (value: string) => void
}) {
  return (
    <box style={{ paddingBottom: 1, flexDirection: 'column' }}>
      <text style={{ fg: theme.muted }}>{label}</text>
      <Button onClick={onFocus}>
        <box style={{ borderStyle: 'single', borderColor: focused ? theme.primary : theme.border, paddingLeft: 1, paddingRight: 1 }}>
          <SingleLineInput value={value} onChange={onChange} placeholder={placeholder} focused={focused} masked={mask} />
        </box>
      </Button>
    </box>
  )
}
