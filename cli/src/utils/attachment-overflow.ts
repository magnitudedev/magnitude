import stringWidth from 'string-width'

type AttachmentLike = {
  filename: string
  width: number
  height: number
}

export function truncateFilename(name: string, maxLen = 30): string {
  if (maxLen <= 3) return '.'.repeat(Math.max(0, maxLen))
  if (name.length <= maxLen) return name
  return `${name.slice(0, maxLen - 3)}...`
}

export function formatAttachmentLabel(att: AttachmentLike): string {
  return `■ ${truncateFilename(att.filename)} (${att.width}×${att.height})`
}

export function fitAttachments<T extends AttachmentLike>(
  attachments: readonly T[],
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