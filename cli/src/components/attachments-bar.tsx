import React, { memo, useState } from 'react'
import type { ImageAttachment } from '@magnitudedev/agent'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../hooks/use-theme'
import { fitAttachments, formatAttachmentLabel } from '../utils/attachment-overflow'

interface AttachmentsBarProps {
  attachments: ImageAttachment[]
  onRemove: (index: number) => void
  maxWidth?: number
}

export const AttachmentsBar = memo(function AttachmentsBar({ attachments, onRemove, maxWidth }: AttachmentsBarProps) {
  const theme = useTheme()
  const [hoveredRemoveIndex, setHoveredRemoveIndex] = useState<number | null>(null)

  if (attachments.length === 0) return null

  const fit = maxWidth !== undefined ? fitAttachments(attachments, maxWidth) : null
  const visible = fit ? fit.visible : attachments.map((attachment, index) => ({ index, label: formatAttachmentLabel(attachment) }))
  const hiddenCount = fit ? fit.hiddenCount : 0

  return (
    <box style={{ flexDirection: 'row', paddingBottom: 0 }}>
      {visible.map((item, visibleIndex) => (
        <box key={item.index} style={{ flexDirection: 'row' }}>
          <text style={{ fg: theme.muted }}>
            {`${visibleIndex > 0 ? ' │ ' : ''}${item.label} `}
          </text>
          <text
            style={{ fg: hoveredRemoveIndex === item.index ? theme.foreground : theme.muted }}
            onMouseOver={() => setHoveredRemoveIndex(item.index)}
            onMouseOut={() => setHoveredRemoveIndex((prev) => (prev === item.index ? null : prev))}
            onMouseDown={() => onRemove(item.index)}
          >
            {'[x]'}
          </text>
        </box>
      ))}
      {hiddenCount > 0 && (
        <text style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>
          {`${visible.length > 0 ? ' ' : ''}[and ${hiddenCount} more...]`}
        </text>
      )}
    </box>
  )
})
