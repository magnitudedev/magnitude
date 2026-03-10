import { TextAttributes } from '@opentui/core'
import React, { ReactNode } from 'react'
import stringWidth from 'string-width'
import { renderMermaidAscii } from 'beautiful-mermaid'
import { createLowlight, common } from 'lowlight'
import type { Element, Text, RootContent } from 'hast'
import { blue, slate, green, violet } from './palette'

// Create lowlight instance with common languages
const lowlight = createLowlight(common)

// Syntax highlighting color palette (GitHub dark inspired)
export interface SyntaxColors {
  keyword: string
  string: string
  number: string
  comment: string
  function: string
  variable: string
  type: string
  operator: string
  property: string
  punctuation: string
  literal: string
  default: string
}

export interface MarkdownPalette {
  inlineCodeFg: string
  codeBackground: string
  codeBorderColor: string
  codeHeaderFg: string
  headingFg: Record<number, string>
  listBulletFg: string
  blockquoteBorderFg: string
  blockquoteTextFg: string
  dividerFg: string
  codeTextFg: string
  codeMonochrome: boolean
  linkFg: string
  syntax: SyntaxColors
}

export interface MarkdownRenderOptions {
  palette?: Partial<MarkdownPalette>
  codeBlockWidth?: number
}

const defaultSyntaxColors: SyntaxColors = {
  keyword: violet[300],
  string: green[300],
  number: blue[300],
  comment: slate[500],
  function: blue[400],
  variable: slate[200],
  type: green[300],
  operator: slate[400],
  property: slate[200],
  punctuation: slate[500],
  literal: blue[300],
  default: slate[100],
}

const defaultPalette: MarkdownPalette = {
  inlineCodeFg: green[300],
  codeBackground: 'transparent',
  codeBorderColor: slate[400],
  codeHeaderFg: slate[500],
  headingFg: {
    1: blue[400],
    2: blue[400],
    3: blue[400],
    4: blue[400],
    5: blue[400],
    6: blue[400],
  },
  listBulletFg: slate[400],
  blockquoteBorderFg: slate[700],
  blockquoteTextFg: slate[200],
  dividerFg: slate[800],
  codeTextFg: slate[100],
  codeMonochrome: false,
  linkFg: blue[400],
  syntax: defaultSyntaxColors,
}

const buildMergedPalette = (overrides?: Partial<MarkdownPalette>): MarkdownPalette => {
  const palette: MarkdownPalette = {
    ...defaultPalette,
    headingFg: { ...defaultPalette.headingFg },
    syntax: { ...defaultPalette.syntax },
  }

  if (!overrides) {
    return palette
  }

  const { headingFg, syntax, ...rest } = overrides
  Object.assign(palette, rest)

  if (headingFg) {
    palette.headingFg = {
      ...palette.headingFg,
      ...headingFg,
    }
  }

  if (syntax) {
    palette.syntax = {
      ...palette.syntax,
      ...syntax,
    }
  }

  return palette
}

// Map hljs class names to our syntax color keys
const hljsClassToColor = (classNames: string[], syntax: SyntaxColors): string => {
  for (const cls of classNames) {
    switch (cls) {
      case 'hljs-keyword':
        return syntax.keyword
      case 'hljs-string':
      case 'hljs-template-string':
      case 'hljs-regexp':
        return syntax.string
      case 'hljs-number':
        return syntax.number
      case 'hljs-comment':
        return syntax.comment
      case 'hljs-title':
      case 'hljs-function':
        return syntax.function
      case 'hljs-variable':
      case 'hljs-attr':
      case 'hljs-params':
        return syntax.variable
      case 'hljs-type':
      case 'hljs-built_in':
      case 'hljs-class':
        return syntax.type
      case 'hljs-operator':
        return syntax.operator
      case 'hljs-property':
        return syntax.property
      case 'hljs-punctuation':
        return syntax.punctuation
      case 'hljs-literal':
        return syntax.literal
    }
  }
  return syntax.default
}

// A styled segment of text
interface StyledSegment {
  text: string
  fg?: string
  attributes?: number
  /** If set, this segment is an artifact reference */
  ref?: { artifactName: string; section?: string; label?: string }
}

type Line = StyledSegment[]

// Convert lowlight AST to styled lines
const highlightToLines = (nodes: RootContent[], syntax: SyntaxColors): Line[] => {
  const lines: Line[] = [[]]

  const processMarkdownNode = (node: RootContent, inheritedColor?: string): void => {
    if (node.type === 'text') {
      const textNode = node as Text
      const parts = textNode.value.split('\n')
      parts.forEach((part, idx) => {
        if (idx > 0) {
          lines.push([])
        }
        if (part) {
          lines[lines.length - 1].push({ text: part, fg: inheritedColor ?? syntax.default })
        }
      })
    } else if (node.type === 'element') {
      const el = node as Element
      const classNames = (el.properties?.className as string[]) ?? []
      const color = hljsClassToColor(classNames, syntax)
      for (const child of el.children) {
        processMarkdownNode(child as RootContent, color)
      }
    }
  }

  for (const node of nodes) {
    processMarkdownNode(node)
  }

  return lines
}

// Try to highlight code, returns null if language not supported
const tryHighlight = (code: string, lang: string, syntax: SyntaxColors): Line[] | null => {
  const langMap: Record<string, string> = {
    ts: 'typescript',
    js: 'javascript',
    py: 'python',
    rb: 'ruby',
    rs: 'rust',
    sh: 'bash',
    zsh: 'bash',
    yml: 'yaml',
    jsx: 'javascript',
    tsx: 'typescript',
  }

  const normalizedLang = langMap[lang] ?? lang

  if (!lowlight.registered(normalizedLang)) {
    return null
  }

  try {
    const result = lowlight.highlight(normalizedLang, code)
    return highlightToLines(result.children as RootContent[], syntax)
  } catch {
    return null
  }
}

// =============================================================================
// Chunk-based rendering - separates code blocks from inline content
// =============================================================================

export interface TextChunk {
  type: 'text'
  content: ReactNode
}

export interface CodeChunk {
  type: 'code'
  lang?: string
  lines: Line[] // syntax highlighted lines, or plain text lines
  rawCode: string
}

export interface MermaidChunk {
  type: 'mermaid'
  ascii: string
}

export type MarkdownChunk = TextChunk | CodeChunk | MermaidChunk

// Helper to create a simple text segment
const textSeg = (text: string): StyledSegment => ({ text })

// Helper to create a styled segment
const styledSeg = (text: string, fg?: string, attributes?: number): StyledSegment => ({
  text,
  fg,
  attributes,
})

// Merge a segment into a line
const lineWithPrefix = (prefix: Line, line: Line): Line => [...prefix, ...line]

// Artifact ref pattern
const ARTIFACT_REF_RE = /\[\[([^\]#|]+?)(?:#([^\]|]+?))?(?:\|([^\]]+?))?\]\]/g

/**
 * Process a line of styled segments, detecting [[artifact-ref]] patterns.
 * Merges adjacent segments with identical styling first (to reassemble
 * brackets split by Bun's markdown parser), then splits at ref boundaries.
 */
const processLineRefs = (line: Line): Line => {
  if (line.length === 0) return line

  // Step 1: merge adjacent segments with identical styling
  const merged: StyledSegment[] = []
  for (const seg of line) {
    const prev = merged.length > 0 ? merged[merged.length - 1] : null
    if (prev && !prev.ref && prev.fg === seg.fg && prev.attributes === seg.attributes) {
      prev.text += seg.text
    } else {
      merged.push({ ...seg })
    }
  }

  // Step 2: check if any merged segment contains a ref
  const hasRef = merged.some(seg => {
    ARTIFACT_REF_RE.lastIndex = 0
    return ARTIFACT_REF_RE.test(seg.text)
  })
  if (!hasRef) return merged

  // Step 3: split segments at ref boundaries
  const result: StyledSegment[] = []
  for (const seg of merged) {
    ARTIFACT_REF_RE.lastIndex = 0
    if (!ARTIFACT_REF_RE.test(seg.text)) {
      result.push(seg)
      continue
    }

    // Split this segment's text at ref boundaries
    const regex = new RegExp(ARTIFACT_REF_RE.source, 'g')
    let lastIndex = 0
    let match: RegExpExecArray | null
    while ((match = regex.exec(seg.text)) !== null) {
      // Text before the ref
      if (match.index > lastIndex) {
        result.push({ text: seg.text.slice(lastIndex, match.index), fg: seg.fg, attributes: seg.attributes })
      }
      // The ref segment
      result.push({
        text: match[0],
        fg: seg.fg,
        attributes: seg.attributes,
        ref: {
          artifactName: match[1],
          section: match[2] || undefined,
          label: match[3] || undefined,
        },
      })
      lastIndex = regex.lastIndex
    }
    // Text after last ref
    if (lastIndex < seg.text.length) {
      result.push({ text: seg.text.slice(lastIndex), fg: seg.fg, attributes: seg.attributes })
    }
  }

  return result
}

// Extract plain text from React children
const childrenToText = (children: ReactNode): string => {
  if (typeof children === 'string') return children
  if (typeof children === 'number') return String(children)
  if (children === null || children === undefined) return ''
  if (Array.isArray(children)) return children.map(childrenToText).join('')
  if (React.isValidElement(children)) {
    const el = children as React.ReactElement<{ children?: ReactNode }>
    return childrenToText(el.props.children)
  }
  return ''
}

// Truncate text to fit within specified width
const clipToWidth = (text: string, maxWidth: number): string => {
  if (maxWidth < 1) return ''
  const textWidth = stringWidth(text)
  if (textWidth <= maxWidth) return text
  if (maxWidth === 1) return '…'

  let truncated = ''
  let width = 0
  for (const char of text) {
    const charWidth = stringWidth(char)
    if (width + charWidth + 1 > maxWidth) break
    truncated += char
    width += charWidth
  }
  return truncated + '…'
}

// Pad text to exact width
const rightPadToWidth = (text: string, targetWidth: number): string => {
  const currentWidth = stringWidth(text)
  if (currentWidth >= targetWidth) return text
  return text + ' '.repeat(targetWidth - currentWidth)
}

interface RenderContext {
  palette: MarkdownPalette
  codeBlockWidth: number
  chunks: MarkdownChunk[]
  currentLines: Line[]
}

// Flush accumulated lines to a text chunk
const flushLines = (ctx: RenderContext): void => {
  if (ctx.currentLines.length === 0) return

  // Trim trailing empty lines
  while (ctx.currentLines.length > 0 && ctx.currentLines[ctx.currentLines.length - 1].length === 0) {
    ctx.currentLines.pop()
  }

  if (ctx.currentLines.length === 0) return

  const content = convertLinesToReactNodes(ctx.currentLines)
  ctx.chunks.push({ type: 'text', content })
  ctx.currentLines = []
}

// Convert lines to React nodes
const convertLinesToReactNodes = (lines: Line[]): ReactNode => {
  let keyCounter = 0
  const nextKey = () => `md-${++keyCounter}`

  const result: ReactNode[] = []

  // Process artifact refs in each line before rendering
  const processedLines = lines.map(processLineRefs)

  processedLines.forEach((line, lineIdx) => {
    if (line.length === 0) {
      result.push('\n')
    } else {
      line.forEach((seg) => {
        if (seg.ref) {
          result.push(
            <span
              key={nextKey()}
              fg={seg.fg}
              attributes={seg.attributes}
              data-artifact-ref={seg.ref.artifactName}
              data-artifact-section={seg.ref.section}
              data-artifact-label={seg.ref.label}
            >
              {seg.text}
            </span>,
          )
        } else if (seg.fg || seg.attributes) {
          result.push(
            <span key={nextKey()} fg={seg.fg} attributes={seg.attributes}>
              {seg.text}
            </span>,
          )
        } else {
          result.push(seg.text)
        }
      })
      if (lineIdx < lines.length - 1) {
        result.push('\n')
      }
    }
  })

  if (result.length === 0) return null
  if (result.length === 1) return result[0]

  return (
    <>
      {result.map((node, idx) => (
        <React.Fragment key={`out-${idx}`}>{node}</React.Fragment>
      ))}
    </>
  )
}

// Get element type name from a React element
const getTypeName = (el: React.ReactElement): string => {
  if (typeof el.type === 'string') return el.type
  if (typeof el.type === 'function') return (el.type as { name?: string }).name || ''
  if (typeof el.type === 'symbol') return ''
  return ''
}

// Process a React node tree, accumulating lines and emitting chunks for code blocks
const processMarkdownNode = (node: ReactNode, ctx: RenderContext): void => {
  if (node === null || node === undefined) return

  if (typeof node === 'string') {
    if (node === '\n') {
      ctx.currentLines.push([])
    } else if (node.trim() !== '') {
      ctx.currentLines.push([textSeg(node)])
    }
    return
  }

  if (typeof node === 'number') {
    ctx.currentLines.push([textSeg(String(node))])
    return
  }

  if (Array.isArray(node)) {
    node.forEach((n) => processMarkdownNode(n, ctx))
    return
  }

  if (!React.isValidElement(node)) return

  const el = node as React.ReactElement<Record<string, unknown>>
  const typeName = getTypeName(el)
  const children = el.props.children as ReactNode

  switch (typeName) {
    case 'h1':
    case 'h2':
    case 'h3':
    case 'h4':
    case 'h5':
    case 'h6': {
      const depth = parseInt(typeName[1], 10)
      const text = childrenToText(children)
      const color = ctx.palette.headingFg[depth] || ctx.palette.headingFg[6]
      ctx.currentLines.push([styledSeg(text, color, TextAttributes.BOLD)])
      ctx.currentLines.push([])
      return
    }

    case 'p': {
      const content = renderInlineToLine(children, ctx)
      ctx.currentLines.push(content)
      ctx.currentLines.push([])
      return
    }

    case 'blockquote': {
      const prefix: Line = [styledSeg('> ', ctx.palette.blockquoteBorderFg)]
      const innerLines = renderBlockToLines(children, ctx)
      for (const line of innerLines) {
        const styledLine = line.map((seg) => ({
          ...seg,
          fg: seg.fg || ctx.palette.blockquoteTextFg,
        }))
        ctx.currentLines.push(lineWithPrefix(prefix, styledLine))
      }
      return
    }

    case 'ul':
    case 'ol': {
      const isOrdered = typeName === 'ol'
      const start = typeof el.props.start === 'number' ? (el.props.start as number) : 1
      const items = React.Children.toArray(children)

      items.forEach((item, idx) => {
        if (React.isValidElement(item)) {
          const itemEl = item as React.ReactElement<Record<string, unknown>>
          const checked = itemEl.props.checked as boolean | undefined
          const itemChildren = itemEl.props.children as ReactNode

          let marker = isOrdered ? `${start + idx}. ` : '- '
          if (checked === true) marker += '[x] '
          else if (checked === false) marker += '[ ] '

          const markerSeg: Line = [styledSeg(marker, ctx.palette.listBulletFg)]
          const itemLines = renderListItemToLines(itemChildren, ctx)

          if (itemLines.length === 0) {
            ctx.currentLines.push(markerSeg)
          } else {
            const indent = ' '.repeat(stringWidth(marker))
            const indentLine: Line = [textSeg(indent)]

            itemLines.forEach((line, lineIdx) => {
              if (lineIdx === 0) {
                ctx.currentLines.push(lineWithPrefix(markerSeg, line))
              } else {
                ctx.currentLines.push(lineWithPrefix(indentLine, line))
              }
            })
          }
        }
      })
      ctx.currentLines.push([])
      return
    }

    case 'pre': {
      const lang = el.props.language as string | undefined
      const content = childrenToText(children)

      // Mermaid rendering
      if (lang === 'mermaid') {
        try {
          const ascii = renderMermaidAscii(content.trim(), {
            paddingX: 2,
            paddingY: 2,
            boxBorderPadding: 0,
          })
          // Flush any accumulated text first
          flushLines(ctx)
          ctx.chunks.push({ type: 'mermaid', ascii })
          return
        } catch {
          // Fall through to normal code block
        }
      }

      // Normal code block - flush text and emit code chunk
      flushLines(ctx)

      let codeContent = content
      if (codeContent.endsWith('\n')) {
        codeContent = codeContent.slice(0, -1)
      }

      // Try syntax highlighting
      const highlighted = lang ? tryHighlight(codeContent, lang, ctx.palette.syntax) : null

      let lines: Line[]
      if (highlighted) {
        lines = highlighted.map((line) =>
          line.length === 0 ? [{ text: ' ', fg: ctx.palette.syntax.default }] : line,
        )
      } else {
        // Plain text fallback
        lines = codeContent.split('\n').map((line) => [{ text: line || ' ', fg: ctx.palette.codeTextFg }])
      }

      ctx.chunks.push({ type: 'code', lang, lines, rawCode: codeContent })
      return
    }

    case 'hr': {
      const width = Math.max(10, Math.min(ctx.codeBlockWidth, 80))
      const divider = '─'.repeat(width)
      ctx.currentLines.push([styledSeg(divider, ctx.palette.dividerFg)])
      ctx.currentLines.push([])
      return
    }

    case 'table': {
      const tableLines = buildTableLines(children, ctx)
      ctx.currentLines.push(...tableLines)
      ctx.currentLines.push([])
      return
    }

    case 'html': {
      const text = childrenToText(children).trim()
      if (text) {
        ctx.currentLines.push([textSeg(text)])
      }
      return
    }

    default: {
      // For inline elements or unknown, render inline
      const line = renderInlineToLine(node, ctx)
      if (line.length > 0) {
        ctx.currentLines.push(line)
      }
      return
    }
  }
}

// Render block children to lines (for blockquote, etc. - doesn't handle code blocks specially)
const renderBlockToLines = (children: ReactNode, ctx: RenderContext): Line[] => {
  if (children === null || children === undefined) return []

  if (React.isValidElement(children)) {
    const el = children as React.ReactElement<{ children?: ReactNode }>
    if (el.type === React.Fragment) {
      return renderBlockToLines(el.props.children, ctx)
    }
  }

  const lines: Line[] = []
  const childArray = React.Children.toArray(children)

  for (const child of childArray) {
    if (!React.isValidElement(child)) {
      if (typeof child === 'string' && child.trim()) {
        lines.push([textSeg(child)])
      }
      continue
    }

    const el = child as React.ReactElement<Record<string, unknown>>
    const typeName = getTypeName(el)
    const elChildren = el.props.children as ReactNode

    switch (typeName) {
      case 'p':
        lines.push(renderInlineToLine(elChildren, ctx))
        lines.push([])
        break
      default:
        lines.push(renderInlineToLine(child, ctx))
        break
    }
  }

  return lines
}

// Render list item children to lines
const renderListItemToLines = (children: ReactNode, ctx: RenderContext): Line[] => {
  if (children === null || children === undefined) return []

  const childArray = React.Children.toArray(children)
  const inlineContent: ReactNode[] = []
  const nestedLines: Line[] = []

  for (const child of childArray) {
    if (React.isValidElement(child)) {
      const typeName = getTypeName(child)
      if (typeName === 'ul' || typeName === 'ol') {
        // Render nested list
        const tempCtx: RenderContext = {
          ...ctx,
          chunks: [],
          currentLines: [],
        }
        processMarkdownNode(child, tempCtx)
        nestedLines.push(...tempCtx.currentLines)
      } else {
        inlineContent.push(child)
      }
    } else {
      inlineContent.push(child)
    }
  }

  const lines: Line[] = []
  if (inlineContent.length > 0) {
    const text = childrenToText(inlineContent).trim()
    if (text) {
      lines.push([textSeg(text)])
    }
  }
  lines.push(...nestedLines)

  return lines
}

// Render inline content to a single line
const renderInlineToLine = (node: ReactNode, ctx: RenderContext): Line => {
  if (node === null || node === undefined) return []
  if (typeof node === 'string') return [textSeg(node)]
  if (typeof node === 'number') return [textSeg(String(node))]
  if (Array.isArray(node)) {
    return node.flatMap((n) => renderInlineToLine(n, ctx))
  }

  if (!React.isValidElement(node)) return []

  const el = node as React.ReactElement<Record<string, unknown>>
  const typeName = getTypeName(el)
  const children = el.props.children as ReactNode

  switch (typeName) {
    case 'strong':
      return renderInlineToLine(children, ctx).map((seg) => ({
        ...seg,
        attributes: (seg.attributes || 0) | TextAttributes.BOLD,
      }))

    case 'em':
      return renderInlineToLine(children, ctx).map((seg) => ({
        ...seg,
        attributes: (seg.attributes || 0) | TextAttributes.ITALIC,
      }))

    case 'del':
      return renderInlineToLine(children, ctx).map((seg) => ({
        ...seg,
        attributes: (seg.attributes || 0) | TextAttributes.DIM,
      }))

    case 'code': {
      const text = childrenToText(children)
      return [styledSeg(` ${text} `, ctx.palette.inlineCodeFg, TextAttributes.BOLD)]
    }

    case 'a': {
      const childLine = renderInlineToLine(children, ctx)
      return childLine.map((seg) => ({
        ...seg,
        fg: seg.fg || ctx.palette.linkFg,
      }))
    }

    case 'img': {
      const alt = el.props.alt as string | undefined
      const displayAlt = alt || 'image'
      return [styledSeg(`[${displayAlt}]`, ctx.palette.linkFg)]
    }

    case 'br':
      return [textSeg('\n')]

    case 'u':
      return renderInlineToLine(children, ctx).map((seg) => ({
        ...seg,
        attributes: (seg.attributes || 0) | TextAttributes.UNDERLINE,
      }))

    case 'math':
      return [styledSeg(childrenToText(children), ctx.palette.inlineCodeFg)]

    default:
      return renderInlineToLine(children, ctx)
  }
}

// Render table to lines
const buildTableLines = (children: ReactNode, ctx: RenderContext): Line[] => {
  const rows: string[][] = []

  const extractRows = (node: ReactNode): void => {
    if (!node) return
    if (Array.isArray(node)) {
      node.forEach(extractRows)
      return
    }
    if (React.isValidElement(node)) {
      const el = node as React.ReactElement<{ children?: ReactNode }>
      const typeName = getTypeName(el)

      if (typeName === 'tr') {
        const cells: string[] = []
        const extractCells = (cellNode: ReactNode): void => {
          if (!cellNode) return
          if (Array.isArray(cellNode)) {
            cellNode.forEach(extractCells)
            return
          }
          if (React.isValidElement(cellNode)) {
            const cellEl = cellNode as React.ReactElement<{ children?: ReactNode }>
            const cellTypeName = getTypeName(cellEl)
            if (cellTypeName === 'td' || cellTypeName === 'th') {
              cells.push(childrenToText(cellEl.props.children).trim())
            }
          }
        }
        extractCells(el.props.children)
        if (cells.length > 0) {
          rows.push(cells)
        }
      } else if (typeName === 'thead' || typeName === 'tbody') {
        extractRows(el.props.children)
      }
    }
  }
  extractRows(children)

  if (rows.length === 0) return []

  const numCols = Math.max(...rows.map((r) => r.length))
  if (numCols === 0) return []

  const naturalWidths: number[] = Array(numCols).fill(3)
  for (const row of rows) {
    for (let i = 0; i < row.length; i++) {
      const cellWidth = stringWidth(row[i] || '')
      naturalWidths[i] = Math.max(naturalWidths[i], cellWidth)
    }
  }

  const separatorWidth = 3
  const numSeparators = numCols - 1
  const totalNaturalWidth = naturalWidths.reduce((a, b) => a + b, 0) + numSeparators * separatorWidth
  const availableWidth = Math.max(20, ctx.codeBlockWidth - 2)

  let columnWidths: number[]
  if (totalNaturalWidth <= availableWidth) {
    columnWidths = naturalWidths
  } else {
    const availableForContent = availableWidth - numSeparators * separatorWidth
    const totalNaturalContent = naturalWidths.reduce((a, b) => a + b, 0)
    const scale = availableForContent / totalNaturalContent

    columnWidths = naturalWidths.map((w) => Math.max(3, Math.floor(w * scale)))

    let usedWidth = columnWidths.reduce((a, b) => a + b, 0)
    let remaining = availableForContent - usedWidth
    for (let i = 0; i < columnWidths.length && remaining > 0; i++) {
      if (columnWidths[i] < naturalWidths[i]) {
        const add = Math.min(remaining, naturalWidths[i] - columnWidths[i])
        columnWidths[i] += add
        remaining -= add
      }
    }
  }

  const lines: Line[] = []

  const renderSeparator = (leftChar: string, midChar: string, rightChar: string): Line => {
    let line = leftChar
    columnWidths.forEach((width, idx) => {
      line += '─'.repeat(width + 2)
      line += idx < columnWidths.length - 1 ? midChar : rightChar
    })
    return [styledSeg(line, ctx.palette.dividerFg)]
  }

  lines.push(renderSeparator('┌', '┬', '┐'))

  rows.forEach((row, rowIdx) => {
    const isHeader = rowIdx === 0
    const line: Line = []

    for (let cellIdx = 0; cellIdx < numCols; cellIdx++) {
      const cellText = row[cellIdx] || ''
      const colWidth = columnWidths[cellIdx]
      const displayText = rightPadToWidth(clipToWidth(cellText, colWidth), colWidth)

      if (cellIdx === 0) {
        line.push(styledSeg('│', ctx.palette.dividerFg))
      }

      line.push(
        styledSeg(
          ` ${displayText} `,
          isHeader ? ctx.palette.headingFg[3] : undefined,
          isHeader ? TextAttributes.BOLD : undefined,
        ),
      )

      line.push(styledSeg('│', ctx.palette.dividerFg))
    }
    lines.push(line)

    if (isHeader) {
      lines.push(renderSeparator('├', '┼', '┤'))
    }
  })

  lines.push(renderSeparator('└', '┴', '┘'))

  return lines
}

// =============================================================================
// Public API
// =============================================================================

/**
 * Parse markdown into chunks (text and code blocks separated).
 * Code blocks are returned as separate chunks so they can be rendered
 * with proper box-level backgrounds.
 */
export function parseMarkdownToChunks(
  markdown: string,
  options: MarkdownRenderOptions = {},
): MarkdownChunk[] {
  if (!markdown || markdown.trim() === '') {
    return [{ type: 'text', content: markdown || '' }]
  }

  const palette = buildMergedPalette(options.palette)
  const codeBlockWidth = options.codeBlockWidth ?? 80

  const ctx: RenderContext = {
    palette,
    codeBlockWidth,
    chunks: [],
    currentLines: [],
  }

  try {
    const reactTree = Bun.markdown.react(markdown)

    if (React.isValidElement(reactTree)) {
      const el = reactTree as React.ReactElement<{ children?: ReactNode }>
      if (el.type === React.Fragment) {
        const childArray = React.Children.toArray(el.props.children)
        for (const child of childArray) {
          processMarkdownNode(child, ctx)
        }
      } else {
        processMarkdownNode(reactTree, ctx)
      }
    } else if (Array.isArray(reactTree)) {
      for (const node of reactTree as ReactNode[]) {
        processMarkdownNode(node, ctx)
      }
    }

    // Flush any remaining lines
    flushLines(ctx)

    return ctx.chunks.length > 0 ? ctx.chunks : [{ type: 'text', content: markdown }]
  } catch (error) {
    console.error('Failed to parse markdown', error)
    return [{ type: 'text', content: markdown }]
  }
}

/**
 * Render markdown to React nodes (legacy API - flattens code blocks).
 * For proper code block rendering with backgrounds, use parseMarkdownToChunks instead.
 */
export function renderMarkdownContent(markdown: string, options: MarkdownRenderOptions = {}): ReactNode {
  const chunks = parseMarkdownToChunks(markdown, options)
  const palette = buildMergedPalette(options.palette)

  // Flatten chunks back to a single ReactNode
  const parts: ReactNode[] = []

  for (const chunk of chunks) {
    if (chunk.type === 'text') {
      parts.push(chunk.content)
    } else if (chunk.type === 'code') {
      // Render code block inline (no box background in legacy mode)
      const codeContent = convertLinesToReactNodes(chunk.lines)
      parts.push(
        <React.Fragment key={parts.length}>
          {chunk.lang && (
            <>
              <span fg={palette.codeHeaderFg}>{` ${chunk.lang} `}</span>
              {'\n'}
            </>
          )}
          {codeContent}
          {'\n'}
        </React.Fragment>,
      )
    } else if (chunk.type === 'mermaid') {
      parts.push(
        <React.Fragment key={parts.length}>
          <span fg={palette.codeHeaderFg}>{' mermaid '}</span>
          {'\n'}
          <span fg={palette.codeTextFg}>{chunk.ascii}</span>
          {'\n'}
        </React.Fragment>,
      )
    }
  }

  if (parts.length === 0) return markdown || ''
  if (parts.length === 1) return parts[0]

  return (
    <>
      {parts.map((part, idx) => (
        <React.Fragment key={idx}>{part}</React.Fragment>
      ))}
    </>
  )
}

export function looksLikeMarkdown(content: string): boolean {
  return /[*_`#>\-\+]|\[.*\]\(.*\)|```/.test(content)
}

export function hasOddFenceCount(content: string): boolean {
  let fenceCount = 0
  const fenceRegex = /```/g
  while (fenceRegex.exec(content)) {
    fenceCount += 1
  }
  return fenceCount % 2 === 1
}

// Back-compat exports
export const containsMarkdownSyntax = looksLikeMarkdown
export const hasUnclosedCodeFence = hasOddFenceCount

export function renderStreamingMarkdownContent(content: string, options: MarkdownRenderOptions = {}): ReactNode {
  if (!looksLikeMarkdown(content)) {
    return content
  }

  if (!hasOddFenceCount(content)) {
    return renderMarkdownContent(content, options)
  }

  const lastFenceIndex = content.lastIndexOf('```')
  if (lastFenceIndex === -1) {
    return renderMarkdownContent(content, options)
  }

  const completeSection = content.slice(0, lastFenceIndex)
  const pendingSection = content.slice(lastFenceIndex)

  return (
    <>
      {completeSection.length > 0 && renderMarkdownContent(completeSection, options)}
      {pendingSection}
    </>
  )
}
