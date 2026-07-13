import React, { memo, useState } from 'react'
import { TextAttributes } from '@opentui/core'
import { useRenderer } from '@opentui/react'
import stringWidth from 'string-width'
import { truncateFilename } from '@magnitudedev/client-common'
import { useTheme } from '../../hooks/use-theme'
import type { PendingImageAttachment } from './composer'

/** Format a pending image attachment as a display label. */
function formatAttachmentLabel(att: PendingImageAttachment): string {
  const filename = att.type === 'raw_image_file' ? att.filename : 'clipboard-image'
  return `■ ${truncateFilename(filename)} (${att.width}×${att.height})`
}

/** Fit attachments into a max width, hiding overflow. */
function fitAttachments(
  attachments: readonly PendingImageAttachment[],
  maxWidth: number,
): { visible: Array<{ index: number; label: string }>; hiddenCount: number } {
  if (attachments.length === 0 || maxWidth <= 0) {
    return { visible: [], hiddenCount: attachments.length }
  }

  const visible: Array<{ index: number; label: string }> = []
  let usedWidth = 0

  for (let index = 0; index < attachments.length; index++) {
    const label = formatAttachmentLabel(attachments[index]!)
    const piece = `${visible.length > 0 ? ' │ ' : ''}${label}`
    const pieceWidth = stringWidth(piece)
    if (usedWidth + pieceWidth > maxWidth) break
    visible.push({ index, label })
    usedWidth += pieceWidth
  }

  if (visible.length === attachments.length) {
    return { visible, hiddenCount: 0 }
  }

  let hiddenCount = attachments.length - visible.length
  const suffixFor = (count: number) => `[and ${count} more...]`

  while (visible.length > 0) {
    const joinedVisible = visible.map((v, i) => `${i > 0 ? ' │ ' : ''}${v.label}`).join('')
    const suffix = suffixFor(hiddenCount)
    const totalWidth = stringWidth(`${joinedVisible} ${suffix}`)
    if (totalWidth <= maxWidth) break
    visible.pop()
    hiddenCount += 1
  }

  if (visible.length === 0) {
    const suffixOnlyWidth = stringWidth(suffixFor(hiddenCount))
    if (suffixOnlyWidth > maxWidth) {
      return { visible: [], hiddenCount: attachments.length }
    }
  }

  return { visible, hiddenCount }
}

interface AttachmentsBarProps {
  attachments: PendingImageAttachment[]
  onRemove: (index: number) => void
  maxWidth?: number
}

export const AttachmentsBar = memo(function AttachmentsBar({ attachments, onRemove, maxWidth }: AttachmentsBarProps) {
  const theme = useTheme()
  const renderer = useRenderer()
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
            onMouseOver={() => { setHoveredRemoveIndex(item.index); renderer.setMousePointer('pointer') }}
            onMouseOut={() => { setHoveredRemoveIndex((prev) => (prev === item.index ? null : prev)); renderer.setMousePointer('default') }}
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
