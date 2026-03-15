import { memo, useState, useCallback } from 'react'
import { TextAttributes, type KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import { useTheme } from '../hooks/use-theme'
import { Button } from './button'
import { SingleLineInput } from './single-line-input'
import { WizardHeader, type WizardMode } from './wizard-header'
import { BOX_CHARS } from '../utils/ui-constants'

interface ApiKeyOverlayProps {
  providerName: string
  envKeyHint: string
  initialKey?: string
  onSubmit: (key: string) => void
  onCancel: () => void
  wizardMode?: WizardMode
}

export const ApiKeyOverlay = memo(function ApiKeyOverlay({
  providerName,
  envKeyHint,
  initialKey,
  onSubmit,
  onCancel,
  wizardMode,
}: ApiKeyOverlayProps) {
  const theme = useTheme()
  const [apiKey, setApiKey] = useState(initialKey ?? '')
  const [error, setError] = useState<string | null>(null)
  const [cancelHover, setCancelHover] = useState(false)
  const [backHovered, setBackHovered] = useState(false)

  const handleSubmit = useCallback(() => {
    const trimmed = apiKey.trim()
    if (!trimmed) {
      setError('API key is required')
      return
    }
    onSubmit(trimmed)
  }, [apiKey, onSubmit])

  useKeyboard(
    useCallback((key: KeyEvent) => {
      if (key.name === 'escape') {
        key.preventDefault()
        wizardMode?.onSkip?.() ?? onCancel()
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

      if (!key.defaultPrevented) {
        key.preventDefault()
      }
    }, [onCancel, wizardMode, handleSubmit])
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
              <span attributes={TextAttributes.BOLD}>Connect {providerName}</span>
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
        {/* Label */}
        <box style={{ paddingBottom: 1 }}>
          <text style={{ fg: theme.muted }}>Enter your API key:</text>
        </box>

        {/* Input field */}
        <box style={{
          borderStyle: 'single',
          borderColor: error ? theme.error : theme.primary,
          paddingLeft: 1,
          paddingRight: 1,
          flexShrink: 0,
        }}>
          <SingleLineInput
            value={apiKey}
            onChange={(v) => {
              setApiKey(v)
              setError(null)
            }}
            placeholder="Paste or type API key"
            focused={true}
          />
        </box>

        {/* Error */}
        {error && (
          <box style={{ paddingTop: 1 }}>
            <text style={{ fg: theme.error }}>{error}</text>
          </box>
        )}

        {/* Env var hint */}
        {envKeyHint && (
          <box style={{ paddingTop: 1 }}>
            <text style={{ fg: theme.muted }}>
              <span attributes={TextAttributes.DIM}>Or set {envKeyHint} environment variable and restart Magnitude</span>
            </text>
          </box>
        )}

        {/* Hint */}
        <box style={{ paddingTop: 1 }}>
          <text style={{ fg: theme.muted }}>
            <span attributes={TextAttributes.DIM}>Enter to submit</span>
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
              <text style={{ fg: backHovered ? theme.primary : theme.muted }}>← Back (B)</text>
            </box>
          </Button>
        </box>
      )}
    </box>
  )
})
