import React, { memo } from 'react'
import { useRenderer } from '@opentui/react'
import { buildMarkdownColorPalette } from '../utils/theme'
import { parseMarkdownToMdast } from './parse'
import { renderDocumentToBlocks, type HighlightRange } from './blocks'
import { useStreamingMarkdownCache } from './streaming'
import { useTheme } from '../hooks/use-theme'
import { BlockRenderer } from './block-renderer'

export const MarkdownContent = memo(function MarkdownContent({
  content,
  onOpenArtifact,
  showCursor,
  highlightRanges,
  highlightAnchorId,
  codeBlockWidth,
}: {
  content: string
  onOpenArtifact?: (name: string, section?: string) => void
  showCursor?: boolean
  highlightRanges?: HighlightRange[]
  highlightAnchorId?: string
  codeBlockWidth?: number
}) {
  const theme = useTheme()
  const renderer = useRenderer()
  const palette = buildMarkdownColorPalette(theme)
  const effectiveCodeBlockWidth =
    codeBlockWidth ?? Math.max(20, ((renderer as any)?.terminal?.width ?? (renderer as any)?.screen?.width ?? 80) - 4)
  const blocks = renderDocumentToBlocks(parseMarkdownToMdast(content), {
    palette,
    codeBlockWidth: effectiveCodeBlockWidth,
    highlights: highlightRanges,
  })

  return (
    <box style={{ flexDirection: 'column' }}>
      <BlockRenderer
        blocks={blocks}
        foreground={theme.foreground}
        showCursor={showCursor}
        onOpenArtifact={onOpenArtifact}
        highlights={highlightRanges}
        highlightAnchorId={highlightAnchorId}
      />
      {showCursor && blocks.length === 0 && (
        <text style={{ fg: theme.foreground }}>▍</text>
      )}
    </box>
  )
})

export const StreamingMarkdownContent = memo(function StreamingMarkdownContent({
  content,
  showCursor,
  onOpenArtifact,
  highlightRanges,
  highlightAnchorId,
  streaming,
  codeBlockWidth,
}: {
  content: string
  showCursor?: boolean
  onOpenArtifact?: (name: string, section?: string) => void
  highlightRanges?: HighlightRange[]
  highlightAnchorId?: string
  streaming?: boolean
  codeBlockWidth?: number
}) {
  const theme = useTheme()
  const renderer = useRenderer()
  const palette = buildMarkdownColorPalette(theme)
  const effectiveCodeBlockWidth =
    codeBlockWidth ?? Math.max(20, ((renderer as any)?.terminal?.width ?? (renderer as any)?.screen?.width ?? 80) - 4)
  const { blocks, pendingText } = useStreamingMarkdownCache(content, {
    palette,
    codeBlockWidth: effectiveCodeBlockWidth,
    highlightRanges,
    streaming,
  })

  return (
    <box style={{ flexDirection: 'column' }}>
      <BlockRenderer
        blocks={blocks}
        foreground={theme.foreground}
        showCursor={showCursor && !pendingText}
        onOpenArtifact={onOpenArtifact}
        highlights={highlightRanges}
        highlightAnchorId={highlightAnchorId}
      />
      {pendingText && (
        <text style={{ fg: theme.foreground, wrapMode: 'word' }}>
          {pendingText}
          {showCursor && <span style={{ fg: theme.muted }}>▍</span>}
        </text>
      )}
      {showCursor && blocks.length === 0 && !pendingText && (
        <text style={{ fg: theme.foreground }}>▍</text>
      )}
    </box>
  )
})
