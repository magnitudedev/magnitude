import { memo, useState, useRef, useCallback } from 'react'
import { TextAttributes } from '@opentui/core'
import { useRenderer } from '@opentui/react'
import { Button } from '../../../components/button'
import { useTheme } from '../../../hooks/use-theme'
import { writeTextToClipboard } from '../../../utils/clipboard'
import { formatShortTimestamp } from '@magnitudedev/client-common'
import { formatChord } from '../../../utils/chord'
import { BOX_CHARS } from '../../../utils/ui-constants'
import type { ActionId, ErrorCta } from '../../../types/ui-actions'

const COPY_FEEDBACK_RESET_MS = 2000

interface ErrorMessageProps {
  tag?: string | null
  message: string
  timestamp: number
  cta?: ErrorCta
  onAction?: (actionId: ActionId) => void
}

export const ErrorMessage = memo(function ErrorMessage({ tag, message, timestamp, cta, onAction }: ErrorMessageProps) {
  const theme = useTheme()
  const renderer = useRenderer()
  const [isHovered, setIsHovered] = useState(false)
  const [isCopied, setIsCopied] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const mouseDownRef = useRef(false)

  // Copy-link states for the CTA copy button (URL CTAs only)
  const [linkCopied, setLinkCopied] = useState(false)
  const [linkHovered, setLinkHovered] = useState(false)
  const linkTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Action-button hover state
  const [actionHovered, setActionHovered] = useState(false)

  const showLinkCopied = useCallback(() => {
    setLinkCopied(true)
    if (linkTimerRef.current) clearTimeout(linkTimerRef.current)
    linkTimerRef.current = setTimeout(() => setLinkCopied(false), COPY_FEEDBACK_RESET_MS)
  }, [])

  const prefix = tag ? `[${tag}]` : '[Error]'
  const fullError = `${prefix} ${message}`

  const handleCopyLink = useCallback(async () => {
    if (!cta || cta.kind !== 'url') return
    try {
      await writeTextToClipboard(cta.url)
      showLinkCopied()
    } catch {
      // Error logged by writeTextToClipboard
    }
  }, [cta, showLinkCopied])

  const handleAction = useCallback(() => {
    if (!cta || cta.kind !== 'action' || !onAction) return
    onAction(cta.actionId as ActionId)
  }, [cta, onAction])

  const handleCopy = async () => {
    try {
      await writeTextToClipboard(fullError)
      setIsCopied(true)
      if (timerRef.current) clearTimeout(timerRef.current)
      timerRef.current = setTimeout(() => setIsCopied(false), COPY_FEEDBACK_RESET_MS)
    } catch {
      // Error logged by writeTextToClipboard
    }
  }

  const handleMouseDown = () => {
    mouseDownRef.current = true
  }

  const handleMouseUp = async () => {
    if (mouseDownRef.current) {
      await handleCopy()
    }
    mouseDownRef.current = false
  }

  const handleMouseOver = () => {
    setIsHovered(true)
    if (!cta) renderer.setMousePointer('pointer')
  }

  const handleMouseOut = () => {
    mouseDownRef.current = false
    setIsHovered(false)
    renderer.setMousePointer('default')
  }

  return (
    <box
      style={{ flexDirection: 'column', position: 'relative', marginBottom: 1 }}
      onMouseDown={cta ? undefined : handleMouseDown}
      onMouseUp={cta ? undefined : handleMouseUp}
      onMouseOver={handleMouseOver}
      onMouseOut={handleMouseOut}
    >
      <box style={{
        width: '100%',
        borderStyle: 'single',
        borderColor: theme.error,
        customBorderChars: BOX_CHARS,
        paddingLeft: 1,
        paddingRight: 1,
        flexDirection: 'column',
      }}>
        <text style={{ fg: theme.error }}>
          {fullError}
        </text>

        {cta && cta.kind === 'url' && (
          <box style={{ flexDirection: 'row' }}>
            <text style={{ fg: theme.muted }}>{`${cta.label} → `}</text>
            <text style={{ fg: theme.primary }} attributes={TextAttributes.UNDERLINE}>{cta.url}</text>
            <text style={{ fg: theme.muted }}>{' '}</text>
            <Button
              onClick={handleCopyLink}
              onMouseOver={() => setLinkHovered(true)}
              onMouseOut={() => setLinkHovered(false)}
            >
              <text style={{ fg: linkCopied ? theme.success : (linkHovered ? theme.foreground : theme.muted) }}>
                {linkCopied ? '[Copied ✓]' : '[Copy link]'}
              </text>
            </Button>
          </box>
        )}

        {cta && cta.kind === 'action' && (
          <Button
            style={{ alignSelf: 'flex-start' }}
            onClick={handleAction}
            onMouseOver={() => setActionHovered(true)}
            onMouseOut={() => setActionHovered(false)}
          >
            <text style={{ fg: actionHovered ? theme.foreground : theme.primary }}>
              {`[${cta.label} (${formatChord(cta.chord)})]`}
            </text>
          </Button>
        )}
      </box>

      {/* Only show copy overlay when there's no CTA */}
      {!cta && (isHovered || isCopied) && (
        <box style={{ position: 'absolute', bottom: 0, right: 0, flexDirection: 'row', backgroundColor: theme.terminalDetectedBg ?? 'transparent',  }}>
          <text style={{ fg: isCopied ? 'green' : theme.muted }} attributes={TextAttributes.DIM}>
            {isCopied ? '[Copied ✔] ' : '[Copy ⧉ ] '}
          </text>
          <text style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>
            {formatShortTimestamp(timestamp)}
          </text>
        </box>
      )}
    </box>
  )
})
