import React, { memo } from 'react'
import { useRenderer } from '@opentui/react'
import { useStreamingReveal } from '../hooks/use-streaming-reveal'
import { useTheme } from '../hooks/use-theme'
import { buildMarkdownColorPalette } from '../utils/theme'
import { useStreamingMarkdownCache } from '../hooks/use-streaming-markdown-cache'
import { BlockRenderer } from './block-renderer'

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
  const renderer = useRenderer()
  const markdownPalette = buildMarkdownColorPalette(theme)
  const { displayedContent, showCursor } = useStreamingReveal(content, isStreaming, isInterrupted)
  const codeBlockWidth = Math.max(20, ((renderer as any)?.terminal?.width ?? (renderer as any)?.screen?.width ?? 80) - 4)
  const { blocks, pendingText } = useStreamingMarkdownCache(displayedContent, {
    palette: markdownPalette,
    codeBlockWidth,
    streaming: isStreaming,
  })

  return (
    <box style={{ flexDirection: 'column', marginBottom: 1 }}>
      <BlockRenderer
        blocks={blocks}
        foreground={theme.foreground}
        palette={markdownPalette}
        contentWidth={codeBlockWidth}
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