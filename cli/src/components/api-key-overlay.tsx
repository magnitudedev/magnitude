import { memo, useState, useCallback } from 'react'
import { TextAttributes, type KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import { useTheme } from '../hooks/use-theme'
import { Button } from './button'
import { InputCursor } from './multiline-input'
import { readClipboardText } from '../utils/clipboard'
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
        onCancel()
        return
      }
      if ((key.name === 'return' || key.name === 'enter') && !key.shift) {
        handleSubmit()
        return
      }
      if (key.name === 'backspace' || key.name === 'delete') {
        setApiKey(prev => prev.slice(0, -1))
        setError(null)
        return
      }
      // Cmd+V paste
      if (key.meta && key.name === 'v') {
        const clip = readClipboardText()
        if (clip) {
          setApiKey(prev => prev + clip)
          setError(null)
        }
        return
      }
      // Type characters
      if (key.sequence && key.sequence.length === 1 && !key.ctrl && !key.meta) {
        setApiKey(prev => prev + key.sequence)
        setError(null)
      }
    }, [onCancel, handleSubmit])
  )

  return (
    <box
      focusable={true}
      focused={true}
      onPaste={(event: any) => {
        if (event.text) {
          setApiKey(prev => prev + event.text)
          setError(null)
        }
      }}
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
              <Button onClick={onCancel}>
                <text style={{ fg: theme.muted }} attributes={TextAttributes.UNDERLINE}>Cancel</text>
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
          <text style={{ fg: theme.foreground }}>
            {apiKey}<InputCursor visible={true} focused={true} />
            {!apiKey && <span style={{ fg: theme.muted }}>Paste or type API key</span>}
          </text>
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
              <text style={{ fg: backHovered ? theme.primary : theme.muted }}>← Back</text>
            </box>
          </Button>
        </box>
      )}
    </box>
  )
})
