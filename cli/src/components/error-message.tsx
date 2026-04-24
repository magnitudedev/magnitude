import { memo, useState, useEffect, useRef, useCallback } from 'react'
import { TextAttributes } from '@opentui/core'
import { Button } from './button'
import { useTheme } from '../hooks/use-theme'
import { writeTextToClipboard } from '../utils/clipboard'
import { formatShortTimestamp } from '../utils/strings'
import { BOX_CHARS } from '../utils/ui-constants'

const COPY_FEEDBACK_RESET_MS = 2000
const MAGNITUDE_URL = 'app.magnitude.dev'

interface ErrorMessageProps {
  tag: string | null
  message: string
  timestamp: number
  errorCode?: string
}

export const ErrorMessage = memo(function ErrorMessage({ tag, message, timestamp, errorCode }: ErrorMessageProps) {
  const theme = useTheme()
  const [isHovered, setIsHovered] = useState(false)
  const [isCopied, setIsCopied] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const mouseDownRef = useRef(false)

  // Copy-link states for the CTA copy button
  const [linkCopied, setLinkCopied] = useState(false)
  const [linkHovered, setLinkHovered] = useState(false)
  const linkTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const showLinkCopied = useCallback(() => {
    setLinkCopied(true)
    if (linkTimerRef.current) clearTimeout(linkTimerRef.current)
    linkTimerRef.current = setTimeout(() => setLinkCopied(false), COPY_FEEDBACK_RESET_MS)
  }, [])

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
      if (linkTimerRef.current) clearTimeout(linkTimerRef.current)
    }
  }, [])

  const prefix = tag ? `[${tag}]` : '[Error]'
  const fullError = `${prefix} ${message}`

  // Detect Magnitude-specific errors using error code
  const isSubscriptionError = errorCode === 'subscription_required' || errorCode === 'trial_expired'
  const isUsageLimitError = errorCode?.startsWith('usage_limit_exceeded') ?? false
  const magnitudeCtaText = isSubscriptionError
    ? { label: 'Upgrade to Pro', url: MAGNITUDE_URL }
    : isUsageLimitError
      ? { label: 'Manage your subscription', url: MAGNITUDE_URL }
      : null

  const handleCopyLink = useCallback(async () => {
    try {
      await writeTextToClipboard(`https://${MAGNITUDE_URL}`)
      showLinkCopied()
    } catch {
      // Error logged by writeTextToClipboard
    }
  }, [showLinkCopied])

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
  }

  const handleMouseOut = () => {
    mouseDownRef.current = false
    setIsHovered(false)
  }

  return (
    <box
      style={{ flexDirection: 'column', position: 'relative', marginBottom: 1 }}
      onMouseDown={magnitudeCtaText ? undefined : handleMouseDown}
      onMouseUp={magnitudeCtaText ? undefined : handleMouseUp}
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
        {magnitudeCtaText && (
          <box style={{ flexDirection: 'row' }}>
            <text style={{ fg: theme.muted }}>{magnitudeCtaText.label}</text>
            <text style={{ fg: theme.muted }}>{' at '}</text>
            <text style={{ fg: theme.primary }} attributes={TextAttributes.UNDERLINE}>{magnitudeCtaText.url}</text>
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
      </box>

      {/* Only show copy overlay when there's no Magnitude CTA */}
      {magnitudeCtaText === null && (isHovered || isCopied) && (
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