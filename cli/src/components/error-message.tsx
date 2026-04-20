import { memo, useState, useEffect, useRef } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../hooks/use-theme'
import { writeTextToClipboard } from '../utils/clipboard'
import { formatShortTimestamp } from '../utils/strings'
import { BOX_CHARS } from '../utils/ui-constants'

const COPY_FEEDBACK_RESET_MS = 2000

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

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [])

  const prefix = tag ? `[${tag}]` : '[Error]'
  const fullError = `${prefix} ${message}`

  // Detect Magnitude-specific errors using error code
  const isSubscriptionError = errorCode === 'subscription_required' || errorCode === 'trial_expired'
  const isUsageLimitError = errorCode?.startsWith('usage_limit_exceeded') ?? false
  const magnitudeCtaText = isSubscriptionError
    ? { url: '→ app.magnitude.dev', suffix: ' — upgrade to Pro' }
    : isUsageLimitError
      ? { url: '→ app.magnitude.dev', suffix: ' — manage your subscription' }
      : null

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
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
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
          <text>
            <span style={{ fg: 'cyan' }} attributes={TextAttributes.UNDERLINE}>{magnitudeCtaText.url}</span>
            <span style={{ fg: theme.muted }}>{magnitudeCtaText.suffix}</span>
          </text>
        )}
      </box>

      {(isHovered || isCopied) && (
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
