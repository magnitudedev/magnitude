import React, { memo, useEffect, useRef, useState } from 'react'
import { useTheme } from '../hooks/use-theme'
import { buildMarkdownColorPalette } from '../utils/theme'
import { ChunksView, parseStreamingContent } from './markdown-content'

interface AssistantMessageProps {
  content: string
  isStreaming: boolean
  isInterrupted?: boolean
  onOpenArtifact?: (name: string, section?: string) => void
}

export const AssistantMessage = memo(function AssistantMessage({
  content,
  isStreaming,
  isInterrupted,
  onOpenArtifact,
}: AssistantMessageProps) {
  const theme = useTheme()
  const markdownPalette = buildMarkdownColorPalette(theme)
  const [displayedLength, setDisplayedLength] = useState(content.length)
  const isLinearDrainRef = useRef(!isStreaming)

  useEffect(() => {
    if (isStreaming) {
      isLinearDrainRef.current = false
    } else {
      isLinearDrainRef.current = true
    }
  }, [isStreaming])

  useEffect(() => {
    if (isInterrupted) setDisplayedLength(content.length)
  }, [isInterrupted, content.length])

  useEffect(() => {
    if (!isStreaming && displayedLength >= content.length) return

    const interval = setInterval(() => {
      setDisplayedLength((prev) => {
        const target = content.length
        if (prev >= target) return prev

        if (isLinearDrainRef.current) {
          return Math.min(target, prev + 8)
        }

        const remaining = target - prev
        const speed = Math.max(1, Math.floor(remaining * 0.15))
        return Math.min(target, prev + speed)
      })
    }, 33)

    return () => clearInterval(interval)
  }, [content.length, displayedLength, isStreaming])

  const displayContent = content.slice(0, displayedLength)
  const { chunks, pendingText } = parseStreamingContent(displayContent, { palette: markdownPalette })
  const showCursor = isStreaming || displayedLength < content.length

  return (
    <box style={{ flexDirection: 'column', marginBottom: 1 }}>
      <ChunksView
        chunks={chunks}
        codeBorderColor={markdownPalette.codeBorderColor}
        headerColor={markdownPalette.codeHeaderFg}
        textColor={markdownPalette.codeTextFg}
        foreground={theme.foreground}
        showCursor={showCursor && !pendingText}
        onCodeBlockHoverChange={() => {}}
        onOpenArtifact={onOpenArtifact}
      />
      {pendingText && (
        <text style={{ fg: theme.foreground, wrapMode: 'word' }}>
          {pendingText}
          {showCursor && <span style={{ fg: theme.muted }}>▍</span>}
        </text>
      )}
    </box>
  )
})
