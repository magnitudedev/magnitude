import { memo, useState, useRef } from 'react'
import { TextAttributes } from '@opentui/core'
import { useRenderer } from '@opentui/react'
import stringWidth from 'string-width'
import { useTheme } from '../../../hooks/use-theme'
import { writeTextToClipboard } from '../../../utils/clipboard'
import { fitAttachments } from '@magnitudedev/client-common'
import { formatShortTimestamp } from '@magnitudedev/client-common'
import { BOX_CHARS } from '../../../utils/ui-constants'
import type { ImageAttachment } from '@magnitudedev/sdk'

const COPY_FEEDBACK_RESET_MS = 2000

const USER_MESSAGE_BOX_CHARS = { ...BOX_CHARS, vertical: '┃' }

interface UserMessageProps {
  content: string
  timestamp: number
  taskMode: boolean
  attachments?: readonly ImageAttachment[]
}

export const UserMessage = memo(function UserMessage({ content, timestamp, taskMode, attachments }: UserMessageProps) {
  const theme = useTheme()
  const renderer = useRenderer()
  const [isHovered, setIsHovered] = useState(false)
  const [isCopied, setIsCopied] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const mouseDownRef = useRef(false)

  const handleCopy = async () => {
    try {
      await writeTextToClipboard(content)
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
    renderer.setMousePointer('pointer')
  }

  const handleMouseOut = () => {
    mouseDownRef.current = false
    setIsHovered(false)
    renderer.setMousePointer('default')
  }

  const hasAttachments = Boolean(attachments && attachments.length > 0)
  const showMetadataRow = hasAttachments || isHovered || isCopied
  const metadataVisible = isHovered || isCopied
  const timestampText = formatShortTimestamp(timestamp)
  const copyText = isCopied ? '[Copied ✔] ' : '[Copy ⧉ ] '
  const terminalWidth = process.stdout.columns ?? 80
  const metadataRightText = metadataVisible ? `${copyText}${timestampText}` : ''
  const metadataRightWidth = metadataVisible ? stringWidth(metadataRightText) : 0
  const metadataGapWidth = hasAttachments && metadataVisible ? 1 : 0
  const metadataRowPadding = 3
  const metadataLeftInset = 2
  const metadataSafetyBuffer = 4
  const attachmentsMaxWidth = Math.max(
    0,
    terminalWidth - metadataRightWidth - metadataGapWidth - metadataRowPadding - metadataLeftInset - metadataSafetyBuffer,
  )
  const fittedAttachments = hasAttachments && attachments ? fitAttachments(attachments, attachmentsMaxWidth) : null
  const attachmentText = fittedAttachments
    ? fittedAttachments.visible.map((item, i) => `${i > 0 ? ' │ ' : ''}${item.label}`).join('')
    : ''
  const attachmentSuffix =
    fittedAttachments && fittedAttachments.hiddenCount > 0
      ? `${fittedAttachments.visible.length > 0 ? ' ' : ''}[and ${fittedAttachments.hiddenCount} more...]`
      : ''

  return (
    <box
      style={{ flexDirection: 'column', position: 'relative', marginBottom: 1 }}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      onMouseOver={handleMouseOver}
      onMouseOut={handleMouseOut}
    >
      {/* Outer box: border only (no background) */}
      <box
        style={{
          borderStyle: 'single',
          border: ['left'],
          borderColor: taskMode ? theme.modePlan : theme.primary,
          customBorderChars: USER_MESSAGE_BOX_CHARS,
        }}
      >
        {/* Inner box: background + padding (no border) */}
        <box
          style={{
            flexDirection: 'row',
            backgroundColor: isHovered ? theme.userMessageHoverBg : theme.userMessageBg,
            paddingTop: 1,
            paddingBottom: 1,
            paddingLeft: 1,
            paddingRight: 2,
            flexGrow: 1,
          }}
        >
          <text style={{ fg: theme.foreground, wrapMode: 'word', flexGrow: 1 }}>
{content}
          </text>
        </box>
      </box>

      {/* Metadata row - attachments left, copy/timestamp right */}
      {showMetadataRow && (
        <box style={{ position: 'absolute', bottom: -1, left: 1, right: 0, flexDirection: 'row' }}>
          <box style={{ flexDirection: 'row' }}>
            {attachmentText.length > 0 && (
              <text style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>
                {attachmentText}
              </text>
            )}
            {attachmentSuffix.length > 0 && (
              <text style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>
                {attachmentSuffix}
              </text>
            )}
          </box>

          <box style={{ flexGrow: 1 }} />

          <box style={{ flexDirection: 'row' }}>
            {metadataVisible && (
              <>
                <text style={{ fg: isCopied ? theme.success : theme.muted }} attributes={TextAttributes.DIM}>
                  {copyText}
                </text>
                <text style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>
                  {timestampText}
                </text>
              </>
            )}
          </box>
        </box>
      )}
    </box>
  )
})
