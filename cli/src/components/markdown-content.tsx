/**
 * Reusable markdown content renderer.
 *
 * Extracted from assistant-message.tsx so that both assistant messages
 * and the spec panel can render markdown with the same code.
 */

import React, { memo, useState } from 'react'
import { TextAttributes } from '@opentui/core'
import {
  parseMarkdownToChunks,
  hasOddFenceCount,
  type MarkdownChunk,
  type CodeChunk,
  type MermaidChunk,
} from '../utils/markdown-content-renderer'
import { buildMarkdownColorPalette } from '../utils/theme'
import { writeTextToClipboard } from '../utils/clipboard'
import { useTheme } from '../hooks/use-theme'



const COPY_FEEDBACK_RESET_MS = 2000

// Render a code block with proper box background
export const CodeBlockView = memo(function CodeBlockView({
  chunk,
  codeBorderColor,
  headerColor,
  foreground,
  onHoverChange,
}: {
  chunk: CodeChunk
  codeBorderColor: string
  headerColor: string
  foreground: string
  onHoverChange: (hovered: boolean) => void
}) {
  const [isHovered, setIsHovered] = useState(false)
  const [copied, setCopied] = useState(false)

  const handleCopy = (e: { stopPropagation?: () => void }) => {
    e.stopPropagation?.()
    writeTextToClipboard(chunk.rawCode)
    setCopied(true)
    setTimeout(() => setCopied(false), COPY_FEEDBACK_RESET_MS)
  }

  const handleMouseOver = () => {
    setIsHovered(true)
    onHoverChange(true)
  }

  const handleMouseOut = () => {
    setIsHovered(false)
    onHoverChange(false)
  }

  return (
    <box
      style={{ flexDirection: 'column', position: 'relative', marginBottom: 1 }}
      onMouseOver={handleMouseOver}
      onMouseOut={handleMouseOut}
      onMouseDown={handleCopy}
    >
      <text style={{ fg: codeBorderColor }}>┌ <span fg={headerColor} attributes={TextAttributes.DIM}>{chunk.lang || ''}</span></text>
      <box
        style={{
          flexDirection: 'row',
          borderStyle: 'single',
          border: ['left'],
          borderColor: codeBorderColor,
          paddingLeft: 1,
          paddingRight: 2,
        }}
      >
        <box style={{ flexGrow: 1, flexDirection: 'row' }}>
          <text style={{ wrapMode: 'word', flexGrow: 1 }}>
{chunk.lines.map((line, lineIdx) => (
<React.Fragment key={lineIdx}>
{line.map((seg, segIdx) => (
<span key={segIdx} fg={seg.fg} attributes={seg.attributes}>
{seg.text}
</span>
))}
{lineIdx < chunk.lines.length - 1 && '\n'}
</React.Fragment>
))}
          </text>
        </box>
      </box>
      <text style={{ fg: copied ? 'green' : foreground }}>
        <span fg={codeBorderColor}>└</span>{isHovered && (copied ? ' [Copied ✔]' : ' [Copy ⧉ ]')}
      </text>
    </box>
  )
})

// Render a mermaid diagram as plain text
export const MermaidBlockView = memo(function MermaidBlockView({
  chunk,
  foreground,
}: {
  chunk: MermaidChunk
  foreground: string
}) {
  return (
    <text style={{ fg: foreground }}>
{chunk.ascii}
    </text>
  )
})

// Render chunks - text inline, code blocks as boxes
export const ChunksView = memo(function ChunksView({
  chunks,
  codeBorderColor,
  headerColor,
  textColor,
  foreground,
  showCursor,
  onCodeBlockHoverChange,
  onOpenArtifact,
}: {
  chunks: MarkdownChunk[]
  codeBorderColor: string
  headerColor: string
  textColor: string
  foreground: string
  showCursor?: boolean
  onCodeBlockHoverChange: (hovered: boolean) => void
  onOpenArtifact?: (name: string, section?: string) => void
}) {
  return (
    <>
      {chunks.map((chunk, idx) => {
        const isLast = idx === chunks.length - 1
        if (chunk.type === 'text') {
          return (
            <text key={idx} style={{ fg: foreground, wrapMode: 'word' }}>
              {chunk.content}
              {showCursor && isLast && <span style={{ fg: foreground }}>▍</span>}
            </text>
          )
        } else if (chunk.type === 'code') {
          return (
            <React.Fragment key={idx}>
              <CodeBlockView
                chunk={chunk}
                codeBorderColor={codeBorderColor}
                headerColor={headerColor}
                foreground={foreground}
                onHoverChange={onCodeBlockHoverChange}
              />
              {showCursor && isLast && (
                <text style={{ fg: foreground }}>▍</text>
              )}
            </React.Fragment>
          )
        } else if (chunk.type === 'mermaid') {
          return (
            <React.Fragment key={idx}>
              <MermaidBlockView
                chunk={chunk}
                foreground={foreground}
              />
              {showCursor && isLast && (
                <text style={{ fg: foreground }}>▍</text>
              )}
            </React.Fragment>
          )
        }
        return null
      })}
    </>
  )
})

// Parse streaming content - handle incomplete code fences
export function parseStreamingContent(content: string, options: { palette: ReturnType<typeof buildMarkdownColorPalette> }): {
  chunks: MarkdownChunk[]
  pendingText: string
} {
  if (!hasOddFenceCount(content)) {
    return { chunks: parseMarkdownToChunks(content, options), pendingText: '' }
  }

  // Split at last incomplete fence
  const lastFenceIndex = content.lastIndexOf('```')
  if (lastFenceIndex === -1) {
    return { chunks: parseMarkdownToChunks(content, options), pendingText: '' }
  }

  const completeSection = content.slice(0, lastFenceIndex)
  const pendingSection = content.slice(lastFenceIndex)

  const chunks = completeSection ? parseMarkdownToChunks(completeSection, options) : []
  return { chunks, pendingText: pendingSection }
}

/**
 * Simple markdown content renderer — parses and renders markdown text.
 * No copy indicators, no streaming, no hover state.
 */
export const MarkdownContent = memo(function MarkdownContent({ content, onOpenArtifact }: { content: string; onOpenArtifact?: (name: string, section?: string) => void }) {
  const theme = useTheme()
  const palette = buildMarkdownColorPalette(theme)
  const chunks = parseMarkdownToChunks(content, { palette })

  return (
    <box style={{ flexDirection: 'column' }}>
      <ChunksView
        chunks={chunks}
        codeBorderColor={palette.codeBorderColor}
        headerColor={palette.codeHeaderFg}
        textColor={palette.codeTextFg}
        foreground={theme.foreground}
        onCodeBlockHoverChange={() => {}}
        onOpenArtifact={onOpenArtifact}
      />
    </box>
  )
})
