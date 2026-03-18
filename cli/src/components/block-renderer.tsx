import React, { memo, useRef, useState } from 'react'
import stringWidth from 'string-width'
import {
  TextAttributes,
  type LineInfo,
  type MouseEvent as OTMouseEvent,
  type TextBufferView,
  type TextRenderable,
} from '@opentui/core'
import { useRenderer } from '@opentui/react'
import type {
  Block,
  Span,
  CodeBlock,
  ListBlock,
  BlockquoteBlock,
  TableBlock,
  HighlightRange,
} from '../utils/render-blocks'
import { spansToText } from '../utils/render-blocks'
import { useMountedRef } from '../hooks/use-mounted-ref'
import { useSafeEvent } from '../hooks/use-safe-event'
import { useSafeTimeout } from '../hooks/use-safe-timeout'
import { useTheme } from '../hooks/use-theme'
import { safeRenderableAccess, safeRenderableCall } from '../utils/safe-renderable-access'
import { buildMarkdownColorPalette } from '../utils/theme'
import type { MarkdownPalette } from '../utils/markdown-content-renderer'
import { useArtifacts } from '../hooks/use-artifacts'
import { writeTextToClipboard } from '../utils/clipboard'
import { BOX_CHARS } from '../utils/ui-constants'

const COPY_FEEDBACK_RESET_MS = 2000

function spanAttributes(span: Span): number | undefined {
  let attrs = 0
  if (span.bold) attrs |= TextAttributes.BOLD
  if (span.italic) attrs |= TextAttributes.ITALIC
  if (span.dim) attrs |= TextAttributes.DIM
  return attrs || undefined
}

const SpanRenderer = memo(function SpanRenderer({
  spans,
  foreground,
  onOpenArtifact,
  showCursor,
  id,
}: {
  spans: Span[]
  foreground: string
  onOpenArtifact?: (name: string, section?: string) => void
  showCursor?: boolean
  id?: string
}) {
  const theme = useTheme()
  const renderer = useRenderer()
  const artifactState = useArtifacts()
  const mountedRef = useMountedRef()
  const textRef = useRef<TextRenderable | null>(null)
  const pressStartedRef = useRef<number | null>(null)
  const [hoveredZone, setHoveredZone] = useState<number | null>(null)

  const hitZones: Array<{ charStart: number; charEnd: number; name: string; section?: string }> = []
  let charOffset = 0
  const elements: React.ReactNode[] = []

  for (let i = 0; i < spans.length; i++) {
    const span = spans[i]
    const attrs = spanAttributes(span)

    if (span.ref) {
      const exists = !!artifactState?.artifacts.get(span.ref.name)
      const displayLabel = span.ref.label ?? (span.ref.section ? `${span.ref.name}#${span.ref.section}` : span.ref.name)
      const displayText = exists ? `[≡ ${displayLabel}]` : `[≡ ${displayLabel} (not found)]`

      if (exists) {
        const zoneIdx = hitZones.length
        hitZones.push({
          charStart: charOffset,
          charEnd: charOffset + displayText.length,
          name: span.ref.name,
          section: span.ref.section,
        })

        const isHovered = hoveredZone === zoneIdx
        elements.push(
          <span key={i} fg={isHovered ? theme.link : theme.primary} attributes={TextAttributes.BOLD}>
            {displayText}
          </span>,
        )
      } else {
        elements.push(
          <span key={i} fg={theme.muted} attributes={TextAttributes.DIM}>
            {displayText}
          </span>,
        )
      }
      charOffset += displayText.length
      continue
    }

    if (span.fg || span.bg || attrs) {
      elements.push(
        <span key={i} fg={span.fg ?? foreground} bg={span.bg} attributes={attrs}>
          {span.text}
        </span>,
      )
    } else {
      elements.push(span.text)
    }
    charOffset += span.text.length
  }

  if (showCursor) {
    elements.push(<span key="cursor" fg={foreground}>▍</span>)
  }

  const hitTest = (event: OTMouseEvent): number | null => {
    if (hitZones.length === 0) return null
    return safeRenderableAccess(
      textRef.current,
      (el) => {
        const localX = event.x - el.x
        const localY = event.y - el.y
        const view = (el as unknown as Record<string, unknown>).textBufferView as TextBufferView
        const info: LineInfo = view.lineInfo
        if (localY < 0 || localY >= info.lineStarts.length) return null
        const charIndex = info.lineStarts[localY] + localX
        for (let i = 0; i < hitZones.length; i++) {
          if (charIndex >= hitZones[i].charStart && charIndex < hitZones[i].charEnd) return i
        }
        return null
      },
      {
        mountedRef,
        fallback: null,
      },
    )
  }

  const handleMouseDown = useSafeEvent((event: OTMouseEvent) => {
    const hit = hitTest(event)
    if (hit !== null) pressStartedRef.current = hit
  })

  const handleMouseUp = useSafeEvent((event: OTMouseEvent) => {
    const hit = hitTest(event)
    if (hit !== null && hit === pressStartedRef.current) {
      safeRenderableCall(
        renderer,
        (r) => r.clearSelection(),
        { mountedRef },
      )
      onOpenArtifact?.(hitZones[hit].name, hitZones[hit].section)
    }
    pressStartedRef.current = null
  })

  const handleMouseMove = useSafeEvent((event: OTMouseEvent) => {
    setHoveredZone(hitTest(event))
  })

  const handleMouseOut = useSafeEvent(() => {
    setHoveredZone(null)
    pressStartedRef.current = null
  })

  if (hitZones.length === 0) {
    return (
      <text id={id} style={{ fg: foreground, wrapMode: 'word' }}>
        {elements}
      </text>
    )
  }

  return (
    <text
      id={id}
      ref={(el: TextRenderable | null) => {
        textRef.current = el
      }}
      style={{ fg: foreground, wrapMode: 'word' }}
      selectable={false}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      onMouseMove={handleMouseMove}
      onMouseOut={handleMouseOut}
    >
      {elements}
    </text>
  )
})

function CodeLine({ spans, fallbackFg }: { spans: Span[]; fallbackFg: string }) {
  if (spans.length === 0) return ' '
  return (
    <>
      {spans.map((span, idx) => {
        const attrs = spanAttributes(span)
        return span.fg || span.bg || attrs ? (
          <span key={idx} fg={span.fg ?? fallbackFg} bg={span.bg} attributes={attrs}>
            {span.text}
          </span>
        ) : (
          <React.Fragment key={idx}>{span.text}</React.Fragment>
        )
      })}
    </>
  )
}

function CodeBlockView({
  block,
  foreground,
  palette,
  id,
}: {
  block: CodeBlock
  foreground: string
  palette: MarkdownPalette
  id?: string
}) {
  const theme = useTheme()
  const [isHovered, setIsHovered] = useState(false)
  const [copied, setCopied] = useState(false)
  const copiedTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const safeTimeout = useSafeTimeout()

  const handleCopy = (e: { stopPropagation?: () => void }) => {
    e.stopPropagation?.()
    void writeTextToClipboard(block.rawCode)
    setCopied(true)
    safeTimeout.clear(copiedTimeoutRef.current)
    copiedTimeoutRef.current = safeTimeout.set(() => {
      setCopied(false)
      copiedTimeoutRef.current = null
    }, COPY_FEEDBACK_RESET_MS)
  }

  return (
    <box
      id={id}
      style={{ flexDirection: 'column', position: 'relative' }}
      onMouseOver={() => setIsHovered(true)}
      onMouseOut={() => setIsHovered(false)}
      onMouseDown={handleCopy}
    >
      <text style={{ fg: palette.codeBorderColor }}>
        ┌ <span fg={palette.codeHeaderFg} attributes={TextAttributes.DIM}>{block.language || ''}</span>
      </text>
      <box
        style={{
          flexDirection: 'row',
          borderStyle: 'single',
          border: ['left'],
          borderColor: palette.codeBorderColor,
          customBorderChars: BOX_CHARS,
          paddingLeft: 1,
          paddingRight: 2,
        }}
      >
        <box style={{ flexGrow: 1, flexDirection: 'row' }}>
          <text style={{ flexGrow: 1 }}>
            {block.lines.map((line, lineIdx) => (
              <React.Fragment key={lineIdx}>
                <CodeLine spans={line} fallbackFg={foreground} />
                {lineIdx < block.lines.length - 1 && '\n'}
              </React.Fragment>
            ))}
          </text>
        </box>
      </box>
      <text style={{ fg: copied ? theme.success : foreground }}>
        <span fg={palette.codeBorderColor}>└</span>
        {isHovered && (copied ? ' [Copied ✔]' : ' [Copy ⧉ ]')}
      </text>
    </box>
  )
}

function MermaidBlockView({
  ascii,
  foreground,
  palette,
  id,
}: {
  ascii: string
  foreground: string
  palette: MarkdownPalette
  id?: string
}) {
  return (
    <box id={id} style={{ flexDirection: 'column' }}>
      <text style={{ fg: palette.codeBorderColor }}>
        ┌ <span fg={palette.codeHeaderFg} attributes={TextAttributes.DIM}>mermaid</span>
      </text>
      <box
        style={{
          flexDirection: 'row',
          borderStyle: 'single',
          border: ['left'],
          borderColor: palette.codeBorderColor,
          customBorderChars: BOX_CHARS,
          paddingLeft: 1,
          paddingRight: 2,
        }}
      >
        <text style={{ fg: foreground }}>{ascii}</text>
      </box>
      <text style={{ fg: foreground }}>
        <span fg={palette.codeBorderColor}>└</span>
      </text>
    </box>
  )
}

function itemContentWithMarker(item: ListBlock['items'][number]): Block[] {
  const [first, ...rest] = item.content
  if (!first) {
    return [{ type: 'paragraph', content: [{ text: item.marker, fg: item.markerFg }], source: { start: 0, end: 0 } }]
  }
  if (first.type === 'paragraph' || first.type === 'heading') {
    return [
      {
        ...first,
        content: [{ text: item.marker, fg: item.markerFg }, ...first.content],
      },
      ...rest,
    ]
  }
  return [
    { type: 'paragraph', content: [{ text: item.marker, fg: item.markerFg }], source: { start: 0, end: 0 } },
    ...item.content,
  ]
}

function ListBlockView({
  block,
  foreground,
  palette,
  contentWidth,
  onOpenArtifact,
}: {
  block: ListBlock
  foreground: string
  palette: MarkdownPalette
  contentWidth: number
  onOpenArtifact?: (name: string, section?: string) => void
}) {
  return (
    <box style={{ flexDirection: 'column' }}>
      {block.items.map((item, idx) => {
        const markerWidth = stringWidth(item.marker)
        const [first, ...rest] = itemContentWithMarker(item)
        return (
          <box key={idx} style={{ flexDirection: 'column' }}>
            {first && (
              <BlockRenderer
                blocks={[first]}
                foreground={foreground}
                palette={palette}
                contentWidth={contentWidth}
                onOpenArtifact={onOpenArtifact}
              />
            )}
            {rest.length > 0 && (
              <box style={{ paddingLeft: markerWidth }}>
                <BlockRenderer
                  blocks={rest}
                  foreground={foreground}
                  palette={palette}
                  contentWidth={Math.max(10, contentWidth - markerWidth)}
                  onOpenArtifact={onOpenArtifact}
                />
              </box>
            )}
          </box>
        )
      })}
    </box>
  )
}

function BlockquoteView({
  block,
  foreground,
  palette,
  contentWidth,
  onOpenArtifact,
}: {
  block: BlockquoteBlock
  foreground: string
  palette: MarkdownPalette
  contentWidth: number
  onOpenArtifact?: (name: string, section?: string) => void
}) {
  return (
    <box style={{ flexDirection: 'row' }}>
      <text style={{ fg: palette.blockquoteBorderFg, flexShrink: 0 }}>{'> '}</text>
      <box style={{ flexDirection: 'column', flexGrow: 1 }}>
        <BlockRenderer
          blocks={block.content}
          foreground={foreground}
          palette={palette}
          contentWidth={Math.max(10, contentWidth - 2)}
          onOpenArtifact={onOpenArtifact}
        />
      </box>
    </box>
  )
}

function clipSpans(spans: Span[], maxWidth: number): Span[] {
  if (maxWidth <= 0) return []
  const totalWidth = stringWidth(spansToText(spans))
  if (totalWidth <= maxWidth) return spans

  const targetWidth = maxWidth - 1
  const result: Span[] = []
  let currentWidth = 0

  for (const span of spans) {
    const spanWidth = stringWidth(span.text)
    if (currentWidth + spanWidth <= targetWidth) {
      result.push(span)
      currentWidth += spanWidth
      continue
    }

    const remaining = targetWidth - currentWidth
    if (remaining > 0) {
      let clipped = ''
      let clippedWidth = 0
      for (const char of span.text) {
        const charWidth = stringWidth(char)
        if (clippedWidth + charWidth > remaining) break
        clipped += char
        clippedWidth += charWidth
      }
      if (clipped) {
        result.push({ ...span, text: clipped })
      }
    }

    result.push({ text: '…' })
    return result
  }

  return result
}

function padSpans(spans: Span[], targetWidth: number): Span[] {
  const currentWidth = stringWidth(spansToText(spans))
  if (currentWidth >= targetWidth) return spans
  return [...spans, { text: ' '.repeat(targetWidth - currentWidth) }]
}

function scaleTableWidths(widths: number[], contentWidth: number): number[] {
  if (widths.length === 0) return widths
  const maxTableWidth = Math.max(10, Math.min(contentWidth, 80))
  const naturalTotal = widths.reduce((sum, width) => sum + width, 0) + Math.max(0, widths.length - 1) * 3 + 2
  if (naturalTotal <= maxTableWidth) return widths

  const availableForContent = Math.max(widths.length * 3, maxTableWidth - Math.max(0, widths.length - 1) * 3 - 2)
  const totalNaturalContent = widths.reduce((sum, width) => sum + width, 0)
  const scaled = widths.map((width) => Math.max(3, Math.floor((width / totalNaturalContent) * availableForContent)))

  let used = scaled.reduce((sum, width) => sum + width, 0)
  let remaining = availableForContent - used
  while (remaining > 0) {
    let changed = false
    for (let i = 0; i < scaled.length && remaining > 0; i++) {
      if (scaled[i] < widths[i]) {
        scaled[i] += 1
        remaining -= 1
        changed = true
      }
    }
    if (!changed) break
    used = scaled.reduce((sum, width) => sum + width, 0)
    remaining = availableForContent - used
  }

  return scaled
}

function TableRow({
  cells,
  widths,
  foreground,
  borderColor,
  headerColor,
  header,
}: {
  cells: Span[][]
  widths: number[]
  foreground: string
  borderColor: string
  headerColor: string
  header?: boolean
}) {
  return (
    <text style={{ fg: foreground }}>
      <span fg={borderColor}>│</span>
      {cells.map((cell, idx) => {
        const clipped = clipSpans(cell, widths[idx] ?? 0)
        const padded = padSpans(clipped, widths[idx] ?? 0)
        return (
          <React.Fragment key={idx}>
            <span> </span>
            {padded.map((span, si) => {
              const attrs = header ? TextAttributes.BOLD : spanAttributes(span)
              return (
                <span
                  key={si}
                  fg={header ? paletteOr(headerColor, span.fg ?? foreground) : (span.fg ?? foreground)}
                  bg={span.bg}
                  attributes={attrs}
                >
                  {span.text}
                </span>
              )
            })}
            <span fg={borderColor}> │</span>
          </React.Fragment>
        )
      })}
    </text>
  )
}

function paletteOr(primary: string | undefined, fallback: string): string {
  return primary ?? fallback
}

function TableView({
  block,
  foreground,
  palette,
  contentWidth,
  id,
}: {
  block: TableBlock
  foreground: string
  palette: MarkdownPalette
  contentWidth: number
  id?: string
}) {
  const widths = scaleTableWidths(block.columnWidths, contentWidth)
  const sep = (left: string, mid: string, right: string) =>
    `${left}${widths.map((w) => '─'.repeat(w + 2)).join(mid)}${right}`

  return (
    <box id={id} style={{ flexDirection: 'column' }}>
      <text style={{ fg: palette.dividerFg }}>{sep('┌', '┬', '┐')}</text>
      <TableRow
        cells={block.headers}
        widths={widths}
        foreground={foreground}
        borderColor={palette.dividerFg}
        headerColor={palette.headingFg[3]}
        header
      />
      <text style={{ fg: palette.dividerFg }}>{sep('├', '┼', '┤')}</text>
      {block.rows.map((row, idx) => (
        <TableRow
          key={idx}
          cells={row}
          widths={widths}
          foreground={foreground}
          borderColor={palette.dividerFg}
          headerColor={palette.headingFg[3]}
        />
      ))}
      <text style={{ fg: palette.dividerFg }}>{sep('└', '┴', '┘')}</text>
    </box>
  )
}

function blockHasHighlight(block: Block, highlights: HighlightRange[]): boolean {
  if (!('source' in block)) return false
  return highlights.some((r) => r.start < block.source.end && r.end > block.source.start)
}

export const BlockRenderer = memo(function BlockRenderer({
  blocks,
  foreground,
  palette,
  contentWidth = 79,
  onOpenArtifact,
  showCursor,
  highlightAnchorId,
  highlights,
}: {
  blocks: Block[]
  foreground: string
  palette?: MarkdownPalette
  contentWidth?: number
  onOpenArtifact?: (name: string, section?: string) => void
  showCursor?: boolean
  highlightAnchorId?: string
  highlights?: HighlightRange[]
}) {
  const theme = useTheme()
  const resolvedPalette = palette ?? buildMarkdownColorPalette(theme)
  let didAssignHighlightAnchor = false

  const maybeAnchorId = (block: Block) => {
    if (!highlightAnchorId || !highlights?.length || didAssignHighlightAnchor || !blockHasHighlight(block, highlights)) {
      return undefined
    }
    didAssignHighlightAnchor = true
    return highlightAnchorId
  }

  return (
    <>
      {blocks.map((block, idx) => {
        const isLast = idx === blocks.length - 1
        const anchorId = maybeAnchorId(block)

        switch (block.type) {
          case 'paragraph':
            return (
              <SpanRenderer
                key={idx}
                spans={block.content}
                foreground={foreground}
                onOpenArtifact={onOpenArtifact}
                showCursor={showCursor && isLast}
                id={anchorId}
              />
            )
          case 'heading':
            return (
              <SpanRenderer
                key={idx}
                spans={block.content}
                foreground={foreground}
                onOpenArtifact={onOpenArtifact}
                id={block.slug ? `section-${block.slug}` : anchorId}
                showCursor={showCursor && isLast}
              />
            )
          case 'code':
            return (
              <CodeBlockView
                key={idx}
                block={block}
                foreground={foreground}
                palette={resolvedPalette}
                id={anchorId}
              />
            )
          case 'list':
            return (
              <ListBlockView
                key={idx}
                block={block}
                foreground={foreground}
                palette={resolvedPalette}
                contentWidth={contentWidth}
                onOpenArtifact={onOpenArtifact}
              />
            )
          case 'blockquote':
            return (
              <BlockquoteView
                key={idx}
                block={block}
                foreground={foreground}
                palette={resolvedPalette}
                contentWidth={contentWidth}
                onOpenArtifact={onOpenArtifact}
              />
            )
          case 'table':
            return (
              <TableView
                key={idx}
                block={block}
                foreground={foreground}
                palette={resolvedPalette}
                contentWidth={contentWidth}
                id={anchorId}
              />
            )
          case 'divider':
            return (
              <text key={idx} id={anchorId} style={{ fg: resolvedPalette.dividerFg }}>
                {'─'.repeat(Math.max(10, Math.min(contentWidth, 80)))}
              </text>
            )
          case 'mermaid':
            return (
              <MermaidBlockView
                key={idx}
                ascii={block.ascii}
                foreground={foreground}
                palette={resolvedPalette}
                id={anchorId}
              />
            )
          case 'spacer':
            if (block.lines <= 0) return null
            return <text key={idx}>{block.lines > 1 ? '\n'.repeat(block.lines - 1) : ''}</text>
        }
      })}
    </>
  )
})