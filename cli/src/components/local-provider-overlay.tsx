import { memo, useState, useCallback } from 'react'
import { TextAttributes, type KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import { useTheme } from '../hooks/use-theme'
import { Button } from './button'
import { SingleLineInput } from './single-line-input'
import { WizardHeader, type WizardMode } from './wizard-header'
import { BOX_CHARS } from '../utils/ui-constants'

type FieldName = 'url' | 'model' | 'apiKey'

const FIELDS: FieldName[] = ['url', 'model', 'apiKey']

interface LocalProviderOverlayProps {
  initialConfig?: { url?: string | null; modelId?: string | null }
  onSubmit: (config: { url: string; modelId: string; apiKey?: string }) => void
  onCancel: () => void
  wizardMode?: WizardMode
}

export const LocalProviderOverlay = memo(function LocalProviderOverlay({
  initialConfig,
  onSubmit,
  onCancel,
  wizardMode,
}: LocalProviderOverlayProps) {
  const theme = useTheme()
  const [backHovered, setBackHovered] = useState(false)
  const [cancelHover, setCancelHover] = useState(false)
  const [focusedField, setFocusedField] = useState<FieldName>('url')
  const [url, setUrl] = useState(initialConfig?.url ?? '')
  const [model, setModel] = useState(initialConfig?.modelId ?? '')
  const [apiKey, setApiKey] = useState('')
  const [error, setError] = useState<string | null>(null)


  const setFieldValue = (field: FieldName, value: string) => {
    switch (field) {
      case 'url': setUrl(value); break
      case 'model': setModel(value); break
      case 'apiKey': setApiKey(value); break
    }
  }

  const handleSubmit = useCallback(() => {
    // Strip control characters (ANSI escapes, etc.)
    let trimmedUrl = url.trim().replace(/[\x00-\x1f\x7f]/g, '')
    const trimmedModel = model.trim()

    if (!trimmedUrl) {
      setError('Base URL is required')
      setFocusedField('url')
      return
    }
    // Auto-prepend http:// for local URLs
    if (!trimmedUrl.startsWith('http://') && !trimmedUrl.startsWith('https://')) {
      trimmedUrl = 'http://' + trimmedUrl
    }
    if (!trimmedModel) {
      setError('Model ID is required')
      setFocusedField('model')
      return
    }

    setError(null)
    const trimmedKey = apiKey.trim()
    onSubmit({
      url: trimmedUrl,
      modelId: trimmedModel,
      ...(trimmedKey ? { apiKey: trimmedKey } : {}),
    })
  }, [url, model, apiKey, onSubmit])

  useKeyboard(
    useCallback((key: KeyEvent) => {
      if (key.name === 'escape') {
        key.preventDefault()
        wizardMode?.onBack?.() ?? onCancel()
        return
      }

      if (key.ctrl && key.name === 's' && !key.meta && !key.option && !key.shift && wizardMode?.onSkip) {
        key.preventDefault()
        wizardMode.onSkip()
        return
      }

      if (key.name === 'tab' || key.name === 'up' || key.name === 'down') {
        key.preventDefault()
        setError(null)
        const currentIndex = FIELDS.indexOf(focusedField)
        const goPrevious = key.name === 'up' || (key.name === 'tab' && key.shift)

        if (goPrevious) {
          const prev = (currentIndex - 1 + FIELDS.length) % FIELDS.length
          setFocusedField(FIELDS[prev])
        } else {
          const next = (currentIndex + 1) % FIELDS.length
          setFocusedField(FIELDS[next])
        }
        return
      }

      if ((key.name === 'return' || key.name === 'enter') && !key.shift) {
        key.preventDefault()
        handleSubmit()
        return
      }

      if (!key.defaultPrevented) {
        key.preventDefault()
      }
    }, [onCancel, wizardMode, focusedField, handleSubmit])
  )

  return (
    <box
      style={{ flexDirection: 'column', height: '100%' }}
    >
      {wizardMode ? (
        <WizardHeader
          stepLabel={wizardMode.stepLabel}
          subtitle={wizardMode.subtitle}
          onSkip={wizardMode.onSkip}
          theme={theme}
        />
      ) : (
        <>
          {/* Header */}
          <box style={{
            flexDirection: 'row',
            paddingLeft: 2,
            paddingRight: 2,
            paddingTop: 1,
            paddingBottom: 1,
            flexShrink: 0,
          }}>
            <text style={{ fg: theme.primary, flexGrow: 1 }}>
              <span attributes={TextAttributes.BOLD}>Connect Local</span>
            </text>
            <box style={{ flexDirection: 'row' }}>
              <Button onClick={onCancel} onMouseOver={() => setCancelHover(true)} onMouseOut={() => setCancelHover(false)}>
                <text style={{ fg: cancelHover ? theme.foreground : theme.muted }} attributes={TextAttributes.UNDERLINE}>Cancel</text>
              </Button>
              <text style={{ fg: theme.muted }}>
                <span attributes={TextAttributes.DIM}>{' '}(Esc)</span>
              </text>
            </box>
          </box>

          {/* Divider */}
          <box style={{ paddingLeft: 1, paddingRight: 1, flexShrink: 0 }}>
            <text style={{ fg: theme.border }}>
              {'─'.repeat(80)}
            </text>
          </box>
        </>
      )}

      {/* Content */}
      <box style={{ paddingLeft: 2, paddingRight: 2, paddingTop: 1, flexGrow: 1, flexDirection: 'column' }}>
        {/* Base URL */}
        <Field
          label="Base URL:"
          value={url}
          placeholder="localhost:1234/v1"
          focused={focusedField === 'url'}
          theme={theme}
          onFocus={() => setFocusedField('url')}
          onChange={(v) => {
            setFieldValue('url', v)
            setError(null)
          }}
        />

        {/* Model ID */}
        <Field
          label="Model ID:"
          value={model}
          placeholder="google/gemma-3-4b"
          focused={focusedField === 'model'}
          theme={theme}
          onFocus={() => setFocusedField('model')}
          onChange={(v) => {
            setFieldValue('model', v)
            setError(null)
          }}
        />

        {/* API Key */}
        <Field
          label="API Key (optional):"
          value={apiKey}
          placeholder="Leave empty if not required"
          focused={focusedField === 'apiKey'}
          theme={theme}
          mask
          onFocus={() => setFocusedField('apiKey')}
          onChange={(v) => {
            setFieldValue('apiKey', v)
            setError(null)
          }}
        />

        {/* Error */}
        {error && (
          <box style={{ paddingTop: 1 }}>
            <text style={{ fg: theme.error }}>{error}</text>
          </box>
        )}

        {/* Hint */}
        <box style={{ paddingTop: 1 }}>
          <text style={{ fg: theme.muted }}>
            <span attributes={TextAttributes.DIM}>Tab or ↑/↓ to move between fields  |  Enter to connect</span>
          </text>
        </box>
      </box>

      {wizardMode && (
        <box style={{ paddingLeft: 2, paddingTop: 1, paddingBottom: 1, flexShrink: 0 }}>
          <Button onClick={wizardMode.onBack} onMouseOver={() => setBackHovered(true)} onMouseOut={() => setBackHovered(false)}>
            <box style={{
              borderStyle: 'single',
              borderColor: backHovered ? theme.primary : theme.border,
              customBorderChars: BOX_CHARS,
              paddingLeft: 1,
              paddingRight: 1,
            }}>
              <text style={{ fg: backHovered ? theme.primary : theme.muted }}>← Back (Esc)</text>
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
      <box style={{ paddingBottom: 0 }}>
        <text style={{ fg: theme.muted }}>{label}</text>
      </box>
      <Button onClick={onFocus}>
        <box style={{
          borderStyle: 'single',
          borderColor: focused ? theme.primary : theme.border,
          paddingLeft: 1,
          paddingRight: 1,
          flexShrink: 0,
        }}>
          <SingleLineInput
            value={value}
            onChange={onChange}
            placeholder={placeholder}
            focused={focused}
            masked={mask}
          />
        </box>
      </Button>
    </box>
  )
}
