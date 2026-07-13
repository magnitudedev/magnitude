import { useState } from 'react'
import { useCopyFeedback } from '@magnitudedev/client-common'
import { writeTextToClipboard } from '../../utils/clipboard'
import { Button } from '../../components/button'
import type { ChatTheme } from '../../types/theme-system'

export function CopyButton({ content, theme }: { content: string; theme: ChatTheme }) {
  const [hovered, setHovered] = useState(false)
  const { copied, triggerCopy } = useCopyFeedback(2000)

  const handleCopy = async () => {
    try {
      await writeTextToClipboard(content)
      triggerCopy()
    } catch {
      // Clipboard write failed — no feedback shown.
    }
  }

  const color = copied ? theme.success : hovered ? theme.foreground : theme.muted

  return (
    <Button
      onClick={handleCopy}
      onMouseOver={() => setHovered(true)}
      onMouseOut={() => setHovered(false)}
    >
      <text style={{ fg: color }}>
        {copied ? '[✓]' : '[Copy]'}
      </text>
    </Button>
  )
}

export function CloseButton({ theme, onClose }: { theme: ChatTheme; onClose: () => void }) {
  const [hovered, setHovered] = useState(false)
  const color = hovered ? theme.foreground : theme.muted

  return (
    <Button
      onClick={onClose}
      onMouseOver={() => setHovered(true)}
      onMouseOut={() => setHovered(false)}
    >
      <text style={{ fg: color }}>[✕]</text>
    </Button>
  )
}
