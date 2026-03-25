import { useCallback, useRef, useState } from 'react'
import { useSafeTimeout } from '../hooks/use-safe-timeout'
import { Button } from './button'

export function CopyButton({ content, theme }: { content: string; theme: any }) {
  const [copied, setCopied] = useState(false)
  const [hovered, setHovered] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const safeTimeout = useSafeTimeout()

  const handleCopy = useCallback(() => {
    const proc = require('child_process').spawn('pbcopy', [], { stdio: ['pipe', 'ignore', 'ignore'] })
    proc.stdin.write(content)
    proc.stdin.end()

    setCopied(true)
    safeTimeout.clear(timerRef.current)
    timerRef.current = safeTimeout.set(() => setCopied(false), 2000)
  }, [content, safeTimeout])

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

export function CloseButton({ theme, onClose }: { theme: any; onClose: () => void }) {
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
