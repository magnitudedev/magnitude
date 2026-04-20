import { memo, useState, useCallback, useEffect, useRef } from 'react'
import { TextAttributes, type KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import { useTheme } from '../hooks/use-theme'
import { Button } from './button'
import { SingleLineInput } from './single-line-input'
import { WizardHeader, type WizardMode } from './wizard-header'
import { BOX_CHARS } from '../utils/ui-constants'
import { writeTextToClipboard } from '../utils/clipboard'

const MAGNITUDE_URL = 'https://app.magnitude.dev'

function useCopyFeedback() {
  const [copied, setCopied] = useState(false)
  const timerRef = useRef<NodeJS.Timeout | null>(null)

  const showCopied = useCallback(() => {
    setCopied(true)
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => setCopied(false), 2000)
  }, [])

  useEffect(() => {
    return () => { if (timerRef.current) clearTimeout(timerRef.current) }
  }, [])

  return { copied, showCopied }
}

interface MagnitudeApiKeyOverlayProps {
  providerName?: string
  envKeyHint?: string
  initialKey?: string
  onSubmit: (key: string) => void
  onCancel: () => void
  wizardMode?: WizardMode
}

export const MagnitudeApiKeyOverlay = memo(function MagnitudeApiKeyOverlay({
  initialKey,
  onSubmit,
  onCancel,
  wizardMode,
}: MagnitudeApiKeyOverlayProps) {
  const theme = useTheme()
  const [apiKey, setApiKey] = useState(initialKey ?? '')
  const [error, setError] = useState<string | null>(null)
  const [backHovered, setBackHovered] = useState(false)
  const [continueHovered, setContinueHovered] = useState(false)
  const [cancelHover, setCancelHover] = useState(false)
  const [copyHovered, setCopyHovered] = useState(false)
  const urlCopy = useCopyFeedback()

  const handleBack = wizardMode?.onBack ?? onCancel

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
        ;(wizardMode?.onBack ?? onCancel)()
        return
      }

      if (key.ctrl && key.name === 's' && !key.meta && !key.option && !key.shift && wizardMode?.onSkip) {
        key.preventDefault()
        wizardMode.onSkip()
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
    }, [onCancel, wizardMode, handleSubmit])
  )

  return (
    <box style={{ flexDirection: 'column', height: '100%' }}>
      {wizardMode ? (
        <WizardHeader
          stepLabel={wizardMode.stepLabel}
          subtitle={wizardMode.subtitle}
          onSkip={wizardMode.onSkip}
          theme={theme}
        />
      ) : (
        <>
          <box style={{
            flexDirection: 'row',
            paddingLeft: 2,
            paddingRight: 2,
            paddingTop: 1,
            paddingBottom: 1,
            flexShrink: 0,
          }}>
            <text style={{ fg: theme.primary, flexGrow: 1 }}>
              <span attributes={TextAttributes.BOLD}>Connect Magnitude</span>
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
          <box style={{ paddingLeft: 1, paddingRight: 1, flexShrink: 0 }}>
            <text style={{ fg: theme.border }}>{'─'.repeat(80)}</text>
          </box>
        </>
      )}

      {/* Content */}
      <box style={{ paddingLeft: 2, paddingRight: 2, paddingTop: 1, flexGrow: 1, flexDirection: 'column' }}>

        {/* Step 1: Create account */}
        <box style={{ paddingBottom: 1 }}>
          <text style={{ fg: theme.muted }}>To get started, create a free account at:</text>
        </box>

        <box style={{ paddingBottom: 1, paddingLeft: 2, flexDirection: 'row' }}>
          <text style={{ fg: theme.primary }}>{MAGNITUDE_URL}</text>
          <text> </text>
          <Button
            onClick={async () => {
              try {
                await writeTextToClipboard(MAGNITUDE_URL)
                urlCopy.showCopied()
              } catch {}
            }}
            onMouseOver={() => setCopyHovered(true)}
            onMouseOut={() => setCopyHovered(false)}
          >
            <text style={{ fg: urlCopy.copied ? theme.success : (copyHovered ? theme.foreground : theme.muted) }}>
              {urlCopy.copied ? '[Copied ✓]' : '[Copy link]'}
            </text>
          </Button>
        </box>

        {/* Step 2: Get API key */}
        <box style={{ paddingBottom: 1 }}>
          <text style={{ fg: theme.muted }}>Then copy your API key from the home page and paste it below.</text>
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
            placeholder="Paste API key here"
            focused={true}
          />
        </box>

        {/* Error */}
        {error && (
          <box style={{ paddingTop: 1 }}>
            <text style={{ fg: theme.error }}>{error}</text>
          </box>
        )}

        {/* Trial info (wizard) or env var hint (non-wizard) */}
        {wizardMode ? (
          <box style={{ paddingTop: 1 }}>
            <text style={{ fg: theme.muted }}>
              <span attributes={TextAttributes.DIM}>This will start a free 3 day trial of the $20 Magnitude Pro subscription (no card required). If you would like to continue the subscription at the end of the 3 days, you can add a payment method.</span>
            </text>
          </box>
        ) : (
          <box style={{ paddingTop: 1 }}>
            <text style={{ fg: theme.muted }}>
              <span attributes={TextAttributes.DIM}>Or set MAGNITUDE_API_KEY environment variable and restart Magnitude</span>
            </text>
          </box>
        )}

        {/* Hint */}
        {!wizardMode && (
          <box style={{ paddingTop: 1 }}>
            <text style={{ fg: theme.muted }}>
              <span attributes={TextAttributes.DIM}>Enter to submit</span>
            </text>
          </box>
        )}
      </box>

      {/* Bottom buttons */}
      {wizardMode ? (
        <box style={{ flexDirection: 'row', paddingLeft: 2, paddingRight: 2, paddingTop: 1, paddingBottom: 1, flexShrink: 0 }}>
          <Button onClick={handleBack} onMouseOver={() => setBackHovered(true)} onMouseOut={() => setBackHovered(false)}>
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
          <box style={{ flexGrow: 1 }} />
          <Button onClick={handleSubmit} onMouseOver={() => setContinueHovered(true)} onMouseOut={() => setContinueHovered(false)}>
            <box style={{
              borderStyle: 'single',
              borderColor: continueHovered ? theme.primary : theme.border,
              customBorderChars: BOX_CHARS,
              paddingLeft: 1,
              paddingRight: 1,
            }}>
              <text style={{ fg: continueHovered ? theme.primary : theme.foreground }}>Continue (Enter)</text>
            </box>
          </Button>
        </box>
      ) : null}
    </box>
  )
})
