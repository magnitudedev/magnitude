import React, { memo } from 'react'
import { buildMarkdownColorPalette } from '../utils/theme'
import { parseMarkdownToMdast } from './parse'
import { renderDocumentToBlocks, type HighlightRange } from './blocks'
import { useStreamingMarkdownCache } from './streaming'
import { useTheme } from '../hooks/use-theme'
import { BlockRenderer } from './block-renderer'
import { useBoxWidth } from '../hooks/use-chat-width'

export const MarkdownContent = memo(function MarkdownContent({
  content,
  onOpenArtifact,
  onOpenFile,
  showCursor,
  highlightRanges,
  highlightAnchorId,
  codeBlockWidth,
  contentWidth: explicitContentWidth,
}: {
  content: string
  onOpenArtifact?: (name: string, section?: string) => void
  onOpenFile?: (path: string, section?: string) => void
  showCursor?: boolean
  highlightRanges?: HighlightRange[]
  highlightAnchorId?: string
  codeBlockWidth?: number
  contentWidth?: number
}) {
  const theme = useTheme()
  const palette = buildMarkdownColorPalette(theme)
  const box = useBoxWidth()
  const contentWidth = explicitContentWidth ?? box.width ?? 79
  const effectiveCodeBlockWidth = codeBlockWidth ?? Math.max(20, contentWidth - 2)
  const blocks = renderDocumentToBlocks(parseMarkdownToMdast(content), {
    palette,
    codeBlockWidth: effectiveCodeBlockWidth,
    highlights: highlightRanges,
  })

  return (
    <box ref={box.ref} onSizeChange={box.onSizeChange} style={{ flexDirection: 'column' }}>
      <BlockRenderer
        blocks={blocks}
        foreground={theme.foreground}
        contentWidth={contentWidth}
        showCursor={showCursor}
        onOpenArtifact={onOpenArtifact}
        onOpenFile={onOpenFile}
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
  onOpenFile,
  highlightRanges,
  highlightAnchorId,
  streaming,
  codeBlockWidth,
  contentWidth: explicitContentWidth,
}: {
  content: string
  showCursor?: boolean
  onOpenArtifact?: (name: string, section?: string) => void
  onOpenFile?: (path: string, section?: string) => void
  highlightRanges?: HighlightRange[]
  highlightAnchorId?: string
  streaming?: boolean
  codeBlockWidth?: number
  contentWidth?: number
}) {
  const theme = useTheme()
  const palette = buildMarkdownColorPalette(theme)
  const box = useBoxWidth()
  const contentWidth = explicitContentWidth ?? box.width ?? 79
  const effectiveCodeBlockWidth = codeBlockWidth ?? Math.max(20, contentWidth - 2)
  const { blocks, pendingText } = useStreamingMarkdownCache(content, {
    palette,
    codeBlockWidth: effectiveCodeBlockWidth,
    highlightRanges,
    streaming,
  })

  return (
    <box ref={box.ref} onSizeChange={box.onSizeChange} style={{ flexDirection: 'column' }}>
      <BlockRenderer
        blocks={blocks}
        foreground={theme.foreground}
        contentWidth={contentWidth}
        showCursor={showCursor && !pendingText}
        onOpenArtifact={onOpenArtifact}
        onOpenFile={onOpenFile}
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
