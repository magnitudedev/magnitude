import { memo, useState, useCallback, useEffect, useRef } from 'react'
import { TextAttributes, type KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import { useTheme } from '../hooks/use-theme'
import { Button } from './button'
import { SingleLineInput } from './single-line-input'
import { WizardHeader, type WizardMode } from './wizard-header'
import { BOX_CHARS } from '../utils/ui-constants'

interface OAuthOverlayProps {
  providerName: string
  mode: 'paste' | 'auto'
  url: string
  onCancel: () => void
  onCopyUrl: () => void | Promise<void>
  // Paste mode
  onSubmitCode?: (code: string) => void
  codeError?: string | null
  isSubmitting?: boolean
  // Auto mode
  userCode?: string
  onCopyCode?: () => void | Promise<void>
  wizardMode?: WizardMode
  autoOpenBrowser?: boolean
}

export const OAuthOverlay = memo(function OAuthOverlay(props: OAuthOverlayProps) {
  const [cancelHover, setCancelHover] = useState(false)

  const autoOpenBrowser = props.autoOpenBrowser ?? process.env.NODE_ENV !== 'test'

  if (props.mode === 'paste') {
    return <PasteMode {...props} autoOpenBrowser={autoOpenBrowser} cancelHover={cancelHover} setCancelHover={setCancelHover} />
  }
  return <AutoMode {...props} autoOpenBrowser={autoOpenBrowser} cancelHover={cancelHover} setCancelHover={setCancelHover} />
})

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

function useOpenBrowser(url: string | undefined) {
  useEffect(() => {
    if (!url) return
    try {
      if (process.platform === 'darwin') {
        Bun.spawnSync(['open', url])
      } else if (process.platform === 'linux') {
        Bun.spawnSync(['xdg-open', url])
      } else if (process.platform === 'win32') {
        Bun.spawnSync(['cmd', '/c', 'start', url])
      }
    } catch {}
  }, [])
}

function PasteMode({
  providerName,
  url,
  onSubmitCode,
  codeError,
  isSubmitting,
  onCancel,
  onCopyUrl,
  wizardMode,
  autoOpenBrowser,
  cancelHover,
  setCancelHover,
}: OAuthOverlayProps & { autoOpenBrowser: boolean; cancelHover: boolean; setCancelHover: (hovered: boolean) => void }) {
  const theme = useTheme()
  const [code, setCode] = useState('')
  const urlCopy = useCopyFeedback()
  const [copyHovered, setCopyHovered] = useState(false)
  const [backHovered, setBackHovered] = useState(false)

  useOpenBrowser(autoOpenBrowser ? url : undefined)

  const handleSubmit = useCallback(() => {
    const trimmed = code.trim()
    if (trimmed && onSubmitCode && !isSubmitting) {
      onSubmitCode(trimmed)
    }
  }, [code, onSubmitCode, isSubmitting])

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
    <box
      focusable={true}
      focused={true}
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
        {/* Opening browser */}
        <box style={{ paddingBottom: 1 }}>
          <text style={{ fg: theme.muted }}>Opening browser...</text>
        </box>

        {/* URL */}
        <box style={{ paddingBottom: 0 }}>
          <text style={{ fg: theme.primary }}>{url}</text>
        </box>
        <box style={{ paddingBottom: 1 }}>
          <Button
            onClick={async () => {
              try {
                await onCopyUrl()
                urlCopy.showCopied()
              } catch {}
            }}
            onMouseOver={() => setCopyHovered(true)}
            onMouseOut={() => setCopyHovered(false)}
          >
            <text style={{ fg: urlCopy.copied ? theme.success : (copyHovered ? theme.foreground : theme.muted) }}>
              {urlCopy.copied ? '[Copied ✓]' : '[Copy]'}
            </text>
          </Button>
        </box>

        {/* Instructions */}
        <box style={{ paddingBottom: 1 }}>
          <text style={{ fg: theme.muted }}>
            Paste the authorization code below:
          </text>
        </box>

        {/* Code input */}
        <box style={{
          borderStyle: 'single',
          borderColor: codeError ? theme.error : theme.border,
          paddingLeft: 1,
          paddingRight: 1,
          flexShrink: 0,
        }}>
          <SingleLineInput
            value={code}
            onChange={setCode}
            placeholder="Authorization code"
            focused={true}
          />
        </box>

        {/* Warning banner */}
        <box style={{ paddingTop: 1 }}>
          <text>
            <span style={{ fg: theme.error }}>Warning:</span> Use this method at your own risk. There have been reports of users getting banned.
          </text>
        </box>

        {/* Error or submitting status */}
        {codeError && (
          <box style={{ paddingTop: 1 }}>
            <text style={{ fg: theme.error }}>{codeError}</text>
          </box>
        )}
        {isSubmitting && (
          <box style={{ paddingTop: 1 }}>
            <text style={{ fg: theme.muted }}>Verifying...</text>
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
              <text style={{ fg: backHovered ? theme.primary : theme.muted }}>← Back (Esc)</text>
            </box>
          </Button>
        </box>
      )}
    </box>
  )
}

function AutoMode({
  providerName,
  url,
  userCode,
  onCancel,
  onCopyUrl,
  onCopyCode,
  wizardMode,
  autoOpenBrowser,
  cancelHover,
  setCancelHover,
}: OAuthOverlayProps & { autoOpenBrowser: boolean; cancelHover: boolean; setCancelHover: (hovered: boolean) => void }) {
  const theme = useTheme()
  const urlCopy = useCopyFeedback()
  const codeCopy = useCopyFeedback()
  const [copyHovered, setCopyHovered] = useState(false)
  const [codeCopyHovered, setCodeCopyHovered] = useState(false)
  const [backHovered, setBackHovered] = useState(false)

  // Only auto-open browser when there's no device code to copy first
  useOpenBrowser(autoOpenBrowser && !userCode ? url : undefined)

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
      if (!key.defaultPrevented) {
        key.preventDefault()
      }
    }, [onCancel, wizardMode])
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
        {/* Opening browser */}
        <box style={{ paddingBottom: 1 }}>
          <text style={{ fg: theme.muted }}>Opening browser...</text>
        </box>

        {/* URL */}
        <box style={{ paddingBottom: 0 }}>
          <text style={{ fg: theme.primary }}>{url}</text>
        </box>
        <box style={{ paddingBottom: 1 }}>
          <Button
            onClick={async () => {
              try {
                await onCopyUrl()
                urlCopy.showCopied()
              } catch {}
            }}
            onMouseOver={() => setCopyHovered(true)}
            onMouseOut={() => setCopyHovered(false)}
          >
            <text style={{ fg: urlCopy.copied ? theme.success : (copyHovered ? theme.foreground : theme.muted) }}>
              {urlCopy.copied ? '[Copied ✓]' : '[Copy]'}
            </text>
          </Button>
        </box>

        {/* User code + copy button */}
        {userCode && (
          <>
            <box style={{ paddingBottom: 1 }}>
              <text style={{ fg: theme.muted }}>Enter this code:</text>
            </box>
            <box style={{ flexDirection: 'row', paddingBottom: 1 }}>
              <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>
                {userCode}
              </text>
              <text> </text>
              <Button
                onClick={async () => {
                  try {
                    await onCopyCode?.()
                    codeCopy.showCopied()
                  } catch {}
                }}
                onMouseOver={() => setCodeCopyHovered(true)}
                onMouseOut={() => setCodeCopyHovered(false)}
              >
                <text style={{ fg: codeCopy.copied ? theme.success : (codeCopyHovered ? theme.foreground : theme.muted) }}>
                  {codeCopy.copied ? '[Copied ✓]' : '[Copy]'}
                </text>
              </Button>
            </box>
          </>
        )}

        {/* Waiting message */}
        <box style={{ paddingTop: 1 }}>
          <text style={{ fg: theme.muted }}>Waiting for authorization...</text>
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
}
