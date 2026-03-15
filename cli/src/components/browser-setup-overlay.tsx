/**
 * Browser Setup Overlay
 *
 * Interactive overlay for installing the Chromium browser binary
 * required by the browser agent. Supports detection, guided install,
 * and streaming output. Can render with a wizard header when used
 * as a setup wizard step.
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { TextAttributes, type KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import { useTheme } from '../hooks/use-theme'
import { Button } from './button'
import { BOX_CHARS } from '../utils/ui-constants'
import { WizardHeader, type WizardMode } from './wizard-header'

type SetupState = 'checking' | 'required' | 'installing' | 'done' | 'failed' | 'already-installed'

interface BrowserSetupOverlayProps {
  onClose: () => void
  onResult: (installed: boolean) => void
  wizardMode?: WizardMode
}

export function BrowserSetupOverlay({ onClose, onResult, wizardMode }: BrowserSetupOverlayProps) {
  const theme = useTheme()
  const [state, setState] = useState<SetupState>('checking')
  const [installOutput, setInstallOutput] = useState<string[]>([])
  const [errorOutput, setErrorOutput] = useState<string>('')
  const [installPath, setInstallPath] = useState<string | null>(null)
  const [isInstallHovered, setIsInstallHovered] = useState(false)
  const [closeHover, setCloseHover] = useState(false)
  const [countdown, setCountdown] = useState(5)
  const outputRef = useRef<string[]>([])

  // Fire-once guard for onResult
  const resultFiredRef = useRef(false)
  const installSucceededRef = useRef(false)
  const fireResult = useCallback((installed: boolean) => {
    if (resultFiredRef.current) return
    resultFiredRef.current = true
    onResult(installed)
  }, [onResult])

  // On unmount, fire onResult based on install success if no result was sent yet
  useEffect(() => {
    return () => {
      if (!resultFiredRef.current) {
        onResult(installSucceededRef.current)
      }
    }
  }, [onResult])

  // Check browser installation on mount
  useEffect(() => {
    let mounted = true

    import('@magnitudedev/browser-harness').then(({ isBrowserInstalled, getBrowserExecutablePath }) => {
      if (!mounted) return
      setInstallPath(getBrowserExecutablePath())
      if (isBrowserInstalled()) {
        setState('already-installed')
        installSucceededRef.current = true
      } else {
        setState('required')
      }
    }).catch(() => {
      if (mounted) setState('required')
    })

    return () => { mounted = false }
  }, [fireResult])

  // Auto-close countdown after successful install or already-installed
  useEffect(() => {
    if (state !== 'done' && state !== 'already-installed') return
    setCountdown(5)
    const interval = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          clearInterval(interval)
          fireResult(true)
          onClose()
          return 0
        }
        return prev - 1
      })
    }, 1000)
    return () => clearInterval(interval)
  }, [state, onClose, fireResult])

  const handleInstall = useCallback(async () => {
    setState('installing')
    outputRef.current = []
    setInstallOutput([])

    try {
      const { installBrowser } = await import('@magnitudedev/browser-harness')
      const result = await installBrowser((chunk: string) => {
        outputRef.current = [...outputRef.current, chunk]
        setInstallOutput([...outputRef.current])
      })

      if (result.success) {
        installSucceededRef.current = true
        setState('done')
      } else {
        fireResult(false)
        setErrorOutput(result.output || 'Installation failed with no output.')
        setState('failed')
      }
    } catch (err: any) {
      fireResult(false)
      setErrorOutput(err.message || 'Unexpected error during installation.')
      setState('failed')
    }
  }, [fireResult])

  useKeyboard(useCallback((key: KeyEvent) => {
    const plain = !key.ctrl && !key.meta && !key.option
    const isEnter = (key.name === 'return' || key.name === 'enter') && plain && !key.shift

    if (key.name === 'escape') {
      key.preventDefault()
      onClose()
      return
    }

    if (wizardMode && key.name === 'b' && plain && !key.shift && state !== 'installing') {
      key.preventDefault()
      wizardMode.onBack?.()
      return
    }

    if (isEnter && state === 'required') {
      key.preventDefault()
      void handleInstall()
      return
    }

    if (!key.defaultPrevented) {
      key.preventDefault()
    }
  }, [onClose, wizardMode, state, handleInstall]))

  return (
    <box style={{ flexDirection: 'column', height: '100%' }}>
      {/* Header — wizard mode or standard */}
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
              <span attributes={TextAttributes.BOLD}>Browser Agent Setup</span>
            </text>
            <Button
              onClick={() => onClose()}
              onMouseOver={() => setCloseHover(true)}
              onMouseOut={() => setCloseHover(false)}
            >
              <text style={{ fg: closeHover ? theme.foreground : theme.muted }} attributes={TextAttributes.UNDERLINE}>Close</text>
            </Button>
            <text style={{ fg: theme.muted }}>
              <span attributes={TextAttributes.DIM}>{' '}(Esc)</span>
            </text>
          </box>

          <box style={{ paddingLeft: 1, paddingRight: 1, flexShrink: 0 }}>
            <text style={{ fg: theme.border }}>
              {'─'.repeat(80)}
            </text>
          </box>
        </>
      )}

      {/* Content */}
      <box style={{ flexDirection: 'column', paddingLeft: 2, paddingRight: 2, paddingTop: 1, flexGrow: 1 }}>
        {state === 'checking' && (
          <text style={{ fg: theme.muted }}>Checking browser installation...</text>
        )}

        {state === 'already-installed' && (
          <box style={{ flexDirection: 'column' }}>
            <text style={{ fg: theme.success }}>
              <span attributes={TextAttributes.BOLD}>Chromium is already installed and ready to use.</span>
            </text>
            <box style={{ paddingTop: 1 }}>
              <text style={{ fg: theme.muted }}>
                The browser agent can control web pages for tasks like testing, scraping, and interaction.
              </text>
            </box>
            <box style={{ paddingTop: 1 }}>
              <text style={{ fg: theme.muted }}>
                <span attributes={TextAttributes.DIM}>This overlay will close in {countdown}...</span>
              </text>
            </box>
          </box>
        )}

        {state === 'required' && (
          <box style={{ flexDirection: 'column' }}>
            <text style={{ fg: theme.foreground }}>
              The browser agent requires a Chromium browser binary to control web pages.
            </text>
            <box style={{ paddingTop: 1 }}>
              <text style={{ fg: theme.foreground }}>
                This is a <span attributes={TextAttributes.BOLD}>one-time download</span> (~200MB).
              </text>
            </box>

            <box style={{ paddingTop: 1 }}>
              <text>
                <span style={{ fg: theme.foreground }}>This will run: </span>
                <span style={{ fg: theme.muted }}>{' '}npx patchright install chromium</span>
              </text>
            </box>
            {installPath && (
              <box>
                <text>
                  <span style={{ fg: theme.foreground }}>Install path: </span>
                  <span style={{ fg: theme.muted }}>{' '}{installPath}</span>
                </text>
              </box>
            )}

            <box style={{ paddingTop: 1, alignSelf: 'flex-start' }}>
              <Button
                onClick={handleInstall}
                onMouseOver={() => setIsInstallHovered(true)}
                onMouseOut={() => setIsInstallHovered(false)}
              >
                <box style={{
                  borderStyle: 'single',
                  borderColor: isInstallHovered ? theme.primary : theme.border,
                  customBorderChars: BOX_CHARS,
                  paddingLeft: 2,
                  paddingRight: 2,
                }}>
                  <text style={{ fg: theme.primary }} attributes={TextAttributes.BOLD}>
                    ↓ Install Now (Enter)
                  </text>
                </box>
              </Button>
            </box>
          </box>
        )}

        {state === 'installing' && (
          <box style={{ flexDirection: 'column', flexGrow: 1 }}>
            <text style={{ fg: theme.warning }}>
              <span attributes={TextAttributes.BOLD}>Installing Chromium...</span>
            </text>
            <box style={{ paddingTop: 1, flexGrow: 1 }}>
              <scrollbox
                scrollX={false}
                scrollbarOptions={{ visible: false }}
                verticalScrollbarOptions={{
                  visible: true,
                  trackOptions: { width: 1 },
                }}
                style={{
                  flexGrow: 1,
                  rootOptions: {
                    flexGrow: 1,
                    backgroundColor: 'transparent',
                  },
                  wrapperOptions: {
                    border: false,
                    backgroundColor: 'transparent',
                  },
                  contentOptions: {
                    paddingLeft: 1,
                  },
                }}
              >
                <text style={{ fg: theme.muted }}>
                  {installOutput.join('') || 'Starting...'}
                </text>
              </scrollbox>
            </box>
          </box>
        )}

        {state === 'done' && (
          <box style={{ flexDirection: 'column' }}>
            <text style={{ fg: theme.success }}>
              <span attributes={TextAttributes.BOLD}>Chromium installed successfully.</span>
            </text>
            {installPath && (
              <box style={{ paddingTop: 1 }}>
                <text>
                  <span style={{ fg: theme.foreground }}>Installed at: </span>
                  <span style={{ fg: theme.muted }}>{installPath}</span>
                </text>
              </box>
            )}
            <box style={{ paddingTop: 1 }}>
              <text style={{ fg: theme.foreground }}>
                The browser agent is now ready to use.
              </text>
            </box>
            <box style={{ paddingTop: 1 }}>
              <text style={{ fg: theme.muted }}>
                <span attributes={TextAttributes.DIM}>This overlay will close in {countdown}...</span>
              </text>
            </box>
          </box>
        )}

        {state === 'failed' && (
          <box style={{ flexDirection: 'column' }}>
            <text style={{ fg: theme.error }}>
              <span attributes={TextAttributes.BOLD}>Installation failed</span>
            </text>
            <box style={{ paddingTop: 1 }}>
              <text style={{ fg: theme.foreground }}>
                Try running the command manually:
              </text>
            </box>
            <box style={{ paddingTop: 1 }}>
              <text style={{ fg: theme.warning }}>
                npx patchright install chromium
              </text>
            </box>
            {errorOutput && (
              <box style={{ paddingTop: 1 }}>
                <text style={{ fg: theme.muted }}>
                  <span attributes={TextAttributes.DIM}>{errorOutput.slice(0, 500)}</span>
                </text>
              </box>
            )}
          </box>
        )}
      </box>

      {/* Footer */}
      <box style={{ paddingLeft: 2, paddingTop: 1, paddingBottom: 1, flexShrink: 0, flexDirection: 'row', gap: 2 }}>
        {wizardMode && state !== 'installing' && (
          <Button onClick={wizardMode.onBack}>
            <box style={{
              borderStyle: 'single',
              borderColor: theme.border,
              customBorderChars: BOX_CHARS,
              paddingLeft: 1,
              paddingRight: 1,
            }}>
              <text style={{ fg: theme.muted }}>← Back (B)</text>
            </box>
          </Button>
        )}
        <text style={{ fg: theme.muted }}>
          <span attributes={TextAttributes.DIM}>
            {state === 'required' ? 'Press Enter to install, Esc to close' : 'Press Esc to close'}
          </span>
        </text>
      </box>
    </box>
  )
}
