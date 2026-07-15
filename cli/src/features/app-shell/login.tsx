import { memo, useState, useCallback, useMemo, useRef } from 'react'
import { TextAttributes, type KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import { useTheme } from '../../hooks/use-theme'
import { Button } from '../../components/button'
import { SingleLineInput } from '../composer/single-line-input'
import { BOX_CHARS } from '../../utils/ui-constants'
import { writeTextToClipboard } from '../../utils/clipboard'
import { green, orange, rose, useAgentClient } from '@magnitudedev/client-common'
import type { SlotProfile } from '@magnitudedev/sdk'
import { SLOT_IDS, SLOT_DISPLAY_NAMES } from '@magnitudedev/sdk'
import { Atom, Result, useAtomValue } from '@effect-atom/atom-react'

import type { BorderCharacters } from '@opentui/core'
import type { SlotId } from '@magnitudedev/sdk'

const MAGNITUDE_URL = 'https://app.magnitude.dev'

const SLOT_ICONS: Record<SlotId, string> = {
  primary: '★',
  secondary: '⚒',
}

const DOUBLE_BOX: BorderCharacters = {
  topLeft: '╔',
  topRight: '╗',
  bottomLeft: '╚',
  bottomRight: '╝',
  horizontal: '═',
  vertical: '║',
  leftT: '╠',
  rightT: '╣',
  topT: '╦',
  bottomT: '╩',
  cross: '╬',
}

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

function getModelColor(modelDisplayName: string, theme: ReturnType<typeof useTheme>): string {
  const name = modelDisplayName.toLowerCase()
  if (name.includes('glm')) return theme.primary
  if (name.includes('minimax')) return orange[400]
  if (name.includes('kimi')) return theme.warning
  if (name.includes('deepseek')) return rose[400]
  if (name.includes('gpt')) return green[400]
  return theme.foreground
}

function padEnd(s: string, length: number): string {
  return s + ' '.repeat(Math.max(0, length - s.length))
}

interface MagnitudeLoginScreenProps {
  onSubmit: (key: string) => Promise<void> | void
  onExit: () => void
  onBack?: () => void
  onSkip?: () => Promise<void> | void
  busy?: boolean
  error?: string | null
}

export const MagnitudeLoginScreen = memo(function MagnitudeLoginScreen({
  onSubmit,
  onExit,
  onBack,
  onSkip,
  busy = false,
  error: serverError = null,
}: MagnitudeLoginScreenProps) {
  const theme = useTheme()
  const client = useAgentClient()
  const [apiKey, setApiKey] = useState('')
  const [validationError, setValidationError] = useState<string | null>(null)
  const [continueHovered, setContinueHovered] = useState(false)
  const [copyHovered, setCopyHovered] = useState(false)
  const [skipHovered, setSkipHovered] = useState(false)
  const urlCopy = useCopyFeedback()

  const slotProfilesAtom = useMemo(
    () => client.query('ListPublicSlotProfiles', {}, { reactivityKeys: ['config'] }),
    [client],
  )
  const profilesResult = useAtomValue(slotProfilesAtom)
  const slotProfiles = Result.isSuccess(profilesResult) ? profilesResult.value : null

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

  const handleSkip = useCallback(() => {
    if (!onSkip || busy) return
    setValidationError(null)
    try {
      void Promise.resolve(onSkip()).catch(() => {})
    } catch {}
  }, [busy, onSkip])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (key.name === 'escape' && onSkip) {
      key.preventDefault()
      void handleSkip()
      return
    }
    if (key.name === 'left' && onBack) {
      key.preventDefault()
      onBack()
      return
    }
    if (key.ctrl && key.name === 'c') {
      key.preventDefault()
      onExit()
      return
    }
    if ((key.name === 'return' || key.name === 'enter') && !key.shift) {
      key.preventDefault()
      handleSubmit()
      return
    }
  }, [onBack, onExit, onSkip, handleSkip, handleSubmit]))

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
            <span attributes={TextAttributes.BOLD}>MAGNITUDE CLOUD FALLBACK</span>
          </text>
          <text style={{ fg: theme.foreground }}>
            <span attributes={TextAttributes.BOLD}>Use models that are too large for this machine</span>
          </text>
          <text style={{ fg: theme.muted }}>
            Optional cloud inference alongside, or instead of, local models
          </text>

          {/* Slot profiles list (conditional) — inside right column with gap */}
          {slotProfiles && (
            <box style={{
              borderStyle: 'single',
              customBorderChars: BOX_CHARS,
              borderColor: theme.border,
              marginTop: 1,
              paddingLeft: 1,
              paddingRight: 1,
              paddingTop: 1,
              paddingBottom: 1,
              flexShrink: 0,
              alignSelf: 'flex-start',
            }}>
              <box style={{ flexDirection: 'column' }}>
                {SLOT_IDS.map((slotId) => {
                  const profile = slotProfiles[slotId]
                  const label = SLOT_DISPLAY_NAMES[slotId]
                  const modelName = profile?.modelDisplayName ?? '?'
                  const modelColor = profile ? getModelColor(modelName, theme) : theme.muted
                  return (
                    <box key={slotId} style={{ flexDirection: 'row' }}>
                      <text style={{ fg: theme.foreground }}>
                        <span attributes={TextAttributes.BOLD}>
                          {SLOT_ICONS[slotId]}
                        </span>
                        {' '}
                        {padEnd(label, 12)}{' '}
                      </text>
                      <text style={{ fg: modelColor }}>
                        {padEnd(modelName, 18)}
                      </text>
                    </box>
                  )
                })}
              </box>
            </box>
          )}
        </box>
      </box>

      {/* Sign-up + API key input */}
      <box style={{
        paddingLeft: 2,
        paddingRight: 2,
        paddingTop: slotProfiles ? 1 : 2,
        flexGrow: 1,
        flexDirection: 'column',
      }}>
        <box style={{ paddingBottom: 1, flexDirection: 'row' }}>
          <text style={{ fg: theme.muted }}>Sign up for a free API key → </text>
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
                {busy ? 'Saving...' : 'Connect Cloud (Enter)'}
              </text>
            </box>
          </Button>
        </box>

        {/* Env-var hint */}
        <box style={{ paddingTop: 2 }}>
          <text style={{ fg: theme.muted }}>
            <span attributes={TextAttributes.DIM}>
              {onBack
                ? 'Press ← to return to local model setup.'
                : 'Prefer environment variables? Press Esc to exit, set MAGNITUDE_API_KEY, then relaunch.'}
            </span>
          </text>
        </box>
      </box>

      <box style={{
        flexDirection: 'row',
        justifyContent: 'space-between',
        paddingLeft: 2,
        paddingRight: 2,
        paddingBottom: 1,
        flexShrink: 0,
      }}>
        <text style={{ fg: theme.muted }}>← back to local models · Ctrl+C close</text>
        {onSkip && (
          <Button
            onClick={handleSkip}
            onMouseOver={() => setSkipHovered(true)}
            onMouseOut={() => setSkipHovered(false)}
          >
            <box style={{
              borderStyle: 'single',
              borderColor: skipHovered ? theme.primary : theme.border,
              customBorderChars: BOX_CHARS,
              paddingLeft: 1,
              paddingRight: 1,
            }}>
              <text style={{ fg: skipHovered ? theme.primary : theme.foreground }}>
                Skip for now (Esc)
              </text>
            </box>
          </Button>
        )}
      </box>
    </box>
  )
})

export type { MagnitudeLoginScreenProps }
