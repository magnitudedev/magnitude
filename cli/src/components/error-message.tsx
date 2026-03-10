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
}

export const ErrorMessage = memo(function ErrorMessage({ tag, message, timestamp }: ErrorMessageProps) {
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
      }}>
        <text style={{ fg: theme.error }}>
          {fullError}
        </text>
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
