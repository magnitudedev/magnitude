import React, { memo } from 'react'
import { useStreamingReveal } from '../hooks/use-streaming-reveal'
import { useTheme } from '../hooks/use-theme'
import { buildMarkdownColorPalette } from '../utils/theme'
import { useStreamingMarkdownCache } from '../markdown/streaming'
import { BlockRenderer } from '../markdown/block-renderer'
import { useBoxWidth } from '../hooks/use-chat-width'

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
  const { displayedContent, showCursor } = useStreamingReveal(content, isStreaming, isInterrupted)
  const box = useBoxWidth()
  const contentWidth = box.width ?? 79
  const codeBlockWidth = Math.max(20, contentWidth - 2)
  const { blocks, pendingText } = useStreamingMarkdownCache(displayedContent, {
    palette: markdownPalette,
    codeBlockWidth,
    streaming: isStreaming,
  })

  return (
    <box ref={box.ref} onSizeChange={box.onSizeChange} style={{ flexDirection: 'column', marginBottom: 1 }}>
      <BlockRenderer
        blocks={blocks}
        foreground={theme.foreground}
        palette={markdownPalette}
        contentWidth={contentWidth}
        showCursor={showCursor && !pendingText}
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