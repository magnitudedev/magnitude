import { memo, useState, useCallback, useRef } from 'react'
import { TextAttributes, type KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import { useTheme } from '../../hooks/use-theme'
import { Button } from '../../components/button'
import { SingleLineInput } from '../composer/single-line-input'
import { BOX_CHARS } from '../../utils/ui-constants'
import { writeTextToClipboard } from '../../utils/clipboard'

const MAGNITUDE_URL = 'https://app.magnitude.dev'

function useCopyFeedback() {
  const [copied, setCopied] = useState(false)
  const timerRef = useRef<NodeJS.Timeout | null>(null)

  const showCopied = useCallback(() => {
    setCopied(true)
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => setCopied(false), 2000)
  }, [])

  return { copied, showCopied }
}

interface CloudModelsConnectScreenProps {
  onSubmit: (key: string) => Promise<void> | void
  onExit: () => void
  busy?: boolean
  error?: string | null
}

export const CloudModelsConnectScreen = memo(function CloudModelsConnectScreen({
  onSubmit,
  onExit,
  busy = false,
  error: serverError = null,
}: CloudModelsConnectScreenProps) {
  const theme = useTheme()
  const [apiKey, setApiKey] = useState('')
  const [validationError, setValidationError] = useState<string | null>(null)
  const [continueHovered, setContinueHovered] = useState(false)
  const [copyHovered, setCopyHovered] = useState(false)
  const urlCopy = useCopyFeedback()

  const error = validationError ?? serverError

  const handleSubmit = useCallback(() => {
    if (busy) return
    const trimmed = apiKey.trim()
    if (!trimmed) {
      setValidationError('API key is required')
      return
    }
    setValidationError(null)
    try {
      void Promise.resolve(onSubmit(trimmed)).catch(() => {})
    } catch {}
  }, [apiKey, busy, onSubmit])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (key.name === 'escape') {
      key.preventDefault()
      onExit()
      return
    }
    if (key.ctrl && key.name === 'c') {
      return
    }
    if ((key.name === 'return' || key.name === 'enter') && !key.shift) {
      key.preventDefault()
      handleSubmit()
      return
    }
  }, [onExit, handleSubmit]))

  return (
    <box style={{ flexDirection: 'column', height: '100%' }}>
      {/* Compact header keeps setup usable in a standard 80x24 terminal. */}
      <box style={{
        flexDirection: 'column',
        paddingLeft: 2,
        paddingRight: 2,
        paddingTop: 1,
        paddingBottom: 1,
        flexShrink: 0,
      }}>
        <box style={{ flexDirection: 'column' }}>
          <text style={{ fg: theme.primary }}>
            <span attributes={TextAttributes.BOLD}>CLOUD MODELS</span>
          </text>
          <text style={{ fg: theme.foreground }}>
            <span attributes={TextAttributes.BOLD}>Connect hosted models with Magnitude Pro</span>
          </text>
          <text style={{ fg: theme.muted }}>
            Magnitude Pro lets you:
          </text>
          <box style={{ flexDirection: 'column', paddingTop: 1, paddingLeft: 2 }}>
            <text style={{ fg: theme.foreground }}>• Connect cloud models too large to run on this machine</text>
            <text style={{ fg: theme.foreground }}>• Use Exa web search for external research</text>
          </box>
          <box style={{ paddingTop: 1 }}>
            <text style={{ fg: theme.muted }}>
              Magnitude Pro is $10 for the first month, then $20/month.
            </text>
          </box>
        </box>
      </box>

      {/* Sign-up + API key input */}
      <box style={{
        paddingLeft: 2,
        paddingRight: 2,
        paddingTop: 1,
        flexGrow: 1,
        flexDirection: 'column',
      }}>
        <box style={{ paddingBottom: 1, flexDirection: 'row' }}>
          <text style={{ fg: theme.muted }}>Subscribe to Pro and copy your API key → </text>
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

        <box style={{ paddingBottom: 1 }}>
          <text style={{ fg: theme.foreground }}>Paste your API key:</text>
        </box>

        {/* Input field */}
        <box style={{
          borderStyle: 'single',
          borderColor: error ? theme.error : theme.primary,
          paddingLeft: 1,
          paddingRight: 1,
          flexShrink: 0,
          width: 80,
        }}>
          <SingleLineInput
            value={apiKey}
            onChange={(v) => {
              setApiKey(v)
              setValidationError(null)
            }}
            placeholder="Paste API key here"
            focused={true}
          />
        </box>

        {error && (
          <box style={{ paddingTop: 1 }}>
            <text style={{ fg: theme.error }}>{error}</text>
          </box>
        )}

        {/* Continue button */}
        <box style={{ paddingTop: 1, flexDirection: 'row', flexShrink: 0 }}>
          <Button
            onClick={handleSubmit}
            onMouseOver={() => setContinueHovered(true)}
            onMouseOut={() => setContinueHovered(false)}
          >
            <box style={{
              borderStyle: 'single',
              borderColor: continueHovered ? theme.primary : theme.border,
              customBorderChars: BOX_CHARS,
              paddingLeft: 1,
              paddingRight: 1,
            }}>
              <text style={{ fg: continueHovered ? theme.primary : theme.foreground }}>
                {busy ? 'Saving...' : 'Connect cloud models (Enter)'}
              </text>
            </box>
          </Button>
        </box>

        {/* Env-var hint */}
        <box style={{ paddingTop: 2 }}>
          <text style={{ fg: theme.muted }}>
            <span attributes={TextAttributes.DIM}>
              Prefer environment variables? Press Esc to exit, set MAGNITUDE_API_KEY, then relaunch.
            </span>
          </text>
        </box>
      </box>
    </box>
  )
})

export type { CloudModelsConnectScreenProps }
