import { TextAttributes } from '@opentui/core'
import React, { ReactNode } from 'react'
import stringWidth from 'string-width'
import { renderMermaidAscii } from 'beautiful-mermaid'
import { createLowlight, common } from 'lowlight'
import type { Element, Text, RootContent } from 'hast'
import { parseMarkdown } from '@magnitude/markdown-cst'
import type {
  BlankLinesNode,
  BlockquoteContentNode,
  BlockquoteItemBreakNode,
  BlockquoteItemNode,
  BlockquoteNode,
  BulletItemNode,
  BulletListNode,
  CodeBlockNode,
  DefinitionNode,
  DocumentItemNode,
  DocumentNode,
  DocumentContentNode,
  HeadingNode,
  HorizontalRuleNode,
  HtmlBlockNode,
  InlineCodeNode,
  InlineImageNode,
  InlineNode,
  LinkNode,
  ListItemBreakNode,
  ListItemContentItemNode,
  OrderedItemNode,
  OrderedListNode,
  ParagraphNode,
  RootBlockNode,
  SoftBreakNode,
  HardBreakNode,
  StrongNode,
  EmphasisNode,
  StrikethroughNode,
  TableCellNode,
  TableNode,
  TaskItemNode,
  TaskListNode,
  TextNode,
} from '@magnitude/markdown-cst/src/schema'
import { blue, slate, green, violet } from './palette'

// Create lowlight instance with common languages
const lowlight = createLowlight(common)

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
  highlightRanges?: CharacterHighlightRange[]
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

export interface StyledSegment {
  text: string
  fg?: string
  bg?: string
  attributes?: number
  ref?: { artifactName: string; section?: string; label?: string }
}

export type Line = StyledSegment[]

export interface CharacterHighlightRange {
  start: number
  end: number
  backgroundColor: string
}

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

interface BaseChunk {
  startLine: number
  endLine: number
  startChar: number
  endChar: number
}

export interface TextChunk extends BaseChunk {
  type: 'text'
  lines: Line[]
}

export interface CodeChunk extends BaseChunk {
  type: 'code'
  lang?: string
  lines: Line[]
  rawCode: string
}

export interface MermaidChunk extends BaseChunk {
  type: 'mermaid'
  ascii: string
}

export type MarkdownChunk = TextChunk | CodeChunk | MermaidChunk

export function slugify(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')
}

export function extractHeadings(content: string): Array<{ slug: string; lineNumber: number }> {
  const headings: Array<{ slug: string; lineNumber: number }> = []
  content.split('\n').forEach((line, i) => {
    const match = line.match(/^#{1,6}\s+(.+)$/)
    if (match) {
      headings.push({ slug: slugify(match[1].trim()), lineNumber: i })
    }
  })
  return headings
}

const textSeg = (text: string): StyledSegment => ({ text })

const styledSeg = (text: string, fg?: string, attributes?: number): StyledSegment => ({
  text,
  fg,
  attributes,
})

const lineWithPrefix = (prefix: Line, line: Line): Line => [...prefix, ...line]

const ARTIFACT_REF_RE = /\[\[([^\]#|]+?)(?:#([^\]|]+?))?(?:\|([^\]]+?))?\]\]/g

const processLineRefs = (line: Line): Line => {
  if (line.length === 0) return line

  const merged: StyledSegment[] = []
  for (const seg of line) {
    const prev = merged.length > 0 ? merged[merged.length - 1] : null
    if (prev && !prev.ref && prev.fg === seg.fg && prev.bg === seg.bg && prev.attributes === seg.attributes) {
      prev.text += seg.text
    } else {
      merged.push({ ...seg })
    }
  }

  const hasRef = merged.some(seg => {
    ARTIFACT_REF_RE.lastIndex = 0
    return ARTIFACT_REF_RE.test(seg.text)
  })
  if (!hasRef) return merged

  const result: StyledSegment[] = []
  for (const seg of merged) {
    ARTIFACT_REF_RE.lastIndex = 0
    if (!ARTIFACT_REF_RE.test(seg.text)) {
      result.push(seg)
      continue
    }

    const regex = new RegExp(ARTIFACT_REF_RE.source, 'g')
    let lastIndex = 0
    let match: RegExpExecArray | null
    while ((match = regex.exec(seg.text)) !== null) {
      if (match.index > lastIndex) {
        result.push({
          text: seg.text.slice(lastIndex, match.index),
          fg: seg.fg,
          bg: seg.bg,
          attributes: seg.attributes,
        })
      }
      result.push({
        text: match[0],
        fg: seg.fg,
        bg: seg.bg,
        attributes: seg.attributes,
        ref: {
          artifactName: match[1],
          section: match[2] || undefined,
          label: match[3] || undefined,
        },
      })
      lastIndex = regex.lastIndex
    }
    if (lastIndex < seg.text.length) {
      result.push({ text: seg.text.slice(lastIndex), fg: seg.fg, bg: seg.bg, attributes: seg.attributes })
    }
  }

  return result
}

export const convertLinesToReactNodes = (lines: Line[]): ReactNode => {
  let keyCounter = 0
  const nextKey = () => `md-${++keyCounter}`

  const result: ReactNode[] = []
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
              bg={seg.bg}
              attributes={seg.attributes}
              data-artifact-ref={seg.ref.artifactName}
              data-artifact-section={seg.ref.section}
              data-artifact-label={seg.ref.label}
            >
              {seg.text}
            </span>,
          )
        } else if (seg.fg || seg.bg || seg.attributes) {
          result.push(
            <span key={nextKey()} fg={seg.fg} bg={seg.bg} attributes={seg.attributes}>
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

const rightPadToWidth = (text: string, targetWidth: number): string => {
  const currentWidth = stringWidth(text)
  if (currentWidth >= targetWidth) return text
  return text + ' '.repeat(targetWidth - currentWidth)
}

interface BlockEnv {
  linePrefix?: string
  linePrefixStyle?: Pick<StyledSegment, 'fg' | 'bg' | 'attributes'>
  blockquoteDepth: number
}

interface InlineStyle {
  fg?: string
  attributes?: number
  bg?: string
}

interface CSTRenderContext {
  source: string
  palette: MarkdownPalette
  codeBlockWidth: number
  highlightRanges: CharacterHighlightRange[]
  chunks: MarkdownChunk[]
  currentLines: Line[]
  currentLineNumber: number
  minSourceOffset: number | null
  maxSourceOffset: number | null
}

function splitByHighlights(
  text: string,
  sourceStart: number,
  sourceEnd: number,
  style: { fg?: string; bg?: string; attributes?: number; ref?: StyledSegment['ref'] },
  ranges: CharacterHighlightRange[],
): StyledSegment[] {
  if (text.length === 0) return []
  if (ranges.length === 0 || sourceEnd <= sourceStart) return [{ text, ...style }]

  const result: StyledSegment[] = []
  let cursor = 0

  for (const range of ranges) {
    const overlapStart = Math.max(range.start, sourceStart)
    const overlapEnd = Math.min(range.end, sourceEnd)
    if (overlapStart >= overlapEnd) continue

    const localStart = overlapStart - sourceStart
    const localEnd = overlapEnd - sourceStart

    if (localStart > cursor) {
      result.push({ text: text.slice(cursor, localStart), ...style })
    }
    result.push({ text: text.slice(localStart, localEnd), ...style, bg: range.backgroundColor })
    cursor = localEnd
  }

  if (cursor < text.length) {
    result.push({ text: text.slice(cursor), ...style })
  }

  return result.length > 0 ? result : [{ text, ...style }]
}

function ensureCurrentLine(ctx: CSTRenderContext): Line {
  if (ctx.currentLines.length === 0) {
    ctx.currentLines.push([])
  }
  return ctx.currentLines[ctx.currentLines.length - 1]
}

function emitSourceText(
  ctx: CSTRenderContext,
  text: string,
  sourceStart: number,
  sourceEnd: number,
  style: { fg?: string; bg?: string; attributes?: number; ref?: StyledSegment['ref'] } = {},
): void {
  if (!text) return
  const line = ensureCurrentLine(ctx)
  const segments = splitByHighlights(text, sourceStart, sourceEnd, style, ctx.highlightRanges)
  line.push(...segments)

  if (ctx.minSourceOffset === null || sourceStart < ctx.minSourceOffset) ctx.minSourceOffset = sourceStart
  if (ctx.maxSourceOffset === null || sourceEnd > ctx.maxSourceOffset) ctx.maxSourceOffset = sourceEnd
}

function emitSyntheticText(
  ctx: CSTRenderContext,
  text: string,
  style: { fg?: string; bg?: string; attributes?: number; ref?: StyledSegment['ref'] } = {},
): void {
  if (!text) return
  ensureCurrentLine(ctx).push({ text, ...style })
}

function newLine(ctx: CSTRenderContext): void {
  ctx.currentLines.push([])
  ctx.currentLineNumber++
}

function emitBlankLine(ctx: CSTRenderContext): void {
  ctx.currentLines.push([])
  ctx.currentLineNumber++
}

function flushTextChunk(ctx: CSTRenderContext): void {
  if (ctx.currentLines.length === 0) return

  while (ctx.currentLines.length > 0 && ctx.currentLines[ctx.currentLines.length - 1].length === 0) {
    ctx.currentLines.pop()
    ctx.currentLineNumber--
  }

  if (ctx.currentLines.length === 0) {
    ctx.currentLines = [[]]
    return
  }

  const startLine = ctx.currentLineNumber - ctx.currentLines.length + 1
  ctx.chunks.push({
    type: 'text',
    lines: ctx.currentLines.map((line) => line.map((seg) => ({ ...seg }))),
    startLine,
    endLine: ctx.currentLineNumber,
    startChar: ctx.minSourceOffset ?? 0,
    endChar: ctx.maxSourceOffset ?? 0,
  })
  ctx.currentLines = [[]]
  ctx.minSourceOffset = null
  ctx.maxSourceOffset = null
}

function finishBlock(ctx: CSTRenderContext): void {
  emitBlankLine(ctx)
}

function extractInlinePlainText(nodes: readonly InlineNode[] | undefined): string {
  if (!nodes) return ''
  let result = ''
  for (const node of nodes) {
    switch (node.type) {
      case 'text':
        result += node.text
        break
      case 'image':
        result += `[${node.attrs.alt || 'image'}]`
        break
      case 'inlineCode':
        result += node.text
        break
      case 'softBreak':
      case 'hardBreak':
        result += ' '
        break
      case 'emphasis':
      case 'strong':
      case 'strikethrough':
      case 'link':
        result += extractInlinePlainText(node.content)
        break
    }
  }
  return result
}

function flattenTableCell(cell: TableCellNode): string {
  return extractInlinePlainText(cell.content[0]?.content).trim()
}

function renderTableNode(node: TableNode, ctx: CSTRenderContext): void {
  const rows = node.content.map((row) => row.content.map((cell) => flattenTableCell(cell)))
  if (rows.length === 0) return

  const numCols = Math.max(...rows.map((r) => r.length))
  if (numCols === 0) return

  const naturalWidths: number[] = Array(numCols).fill(3)
  for (const row of rows) {
    for (let i = 0; i < row.length; i++) {
      naturalWidths[i] = Math.max(naturalWidths[i], stringWidth(row[i] || ''))
    }
  }

  const separatorWidth = 3
  const availableWidth = Math.max(20, ctx.codeBlockWidth - 2)
  const totalNaturalWidth = naturalWidths.reduce((a, b) => a + b, 0) + (numCols - 1) * separatorWidth

  let columnWidths: number[]
  if (totalNaturalWidth <= availableWidth) {
    columnWidths = naturalWidths
  } else {
    const availableForContent = availableWidth - (numCols - 1) * separatorWidth
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

  const renderSeparator = (leftChar: string, midChar: string, rightChar: string) => {
    let line = leftChar
    columnWidths.forEach((width, idx) => {
      line += '─'.repeat(width + 2)
      line += idx < columnWidths.length - 1 ? midChar : rightChar
    })
    emitSyntheticText(ctx, line, { fg: ctx.palette.dividerFg })
    newLine(ctx)
  }

  renderSeparator('┌', '┬', '┐')

  rows.forEach((row, rowIdx) => {
    const isHeader = rowIdx === 0
    emitSyntheticText(ctx, '│', { fg: ctx.palette.dividerFg })
    for (let cellIdx = 0; cellIdx < numCols; cellIdx++) {
      const cellText = row[cellIdx] || ''
      const colWidth = columnWidths[cellIdx]
      const displayText = rightPadToWidth(clipToWidth(cellText, colWidth), colWidth)
      emitSyntheticText(ctx, ` ${displayText} `, {
        fg: isHeader ? ctx.palette.headingFg[3] : undefined,
        attributes: isHeader ? TextAttributes.BOLD : undefined,
      })
      emitSyntheticText(ctx, '│', { fg: ctx.palette.dividerFg })
    }
    newLine(ctx)

    if (isHeader) {
      renderSeparator('├', '┼', '┤')
    }
  })

  renderSeparator('└', '┴', '┘')
}

function applyPrefixToCurrentLine(ctx: CSTRenderContext, env: BlockEnv): void {
  if (env.linePrefix) {
    emitSyntheticText(ctx, env.linePrefix, env.linePrefixStyle ?? {})
  }
}

function renderInline(
  nodes: readonly InlineNode[] | undefined,
  ctx: CSTRenderContext,
  style: InlineStyle,
  env: BlockEnv,
): void {
  if (!nodes) return

  for (const node of nodes) {
    switch (node.type) {
      case 'text':
        emitSourceText(ctx, node.text, node.position.start.offset, node.position.end.offset, style)
        break
      case 'emphasis':
        renderInline(node.content, ctx, {
          ...style,
          attributes: (style.attributes ?? 0) | TextAttributes.ITALIC,
        }, env)
        break
      case 'strong':
        renderInline(node.content, ctx, {
          ...style,
          attributes: (style.attributes ?? 0) | TextAttributes.BOLD,
        }, env)
        break
      case 'strikethrough':
        renderInline(node.content, ctx, {
          ...style,
          attributes: (style.attributes ?? 0) | TextAttributes.DIM,
        }, env)
        break
      case 'inlineCode': {
        const contentStart = Math.min(node.position.end.offset, node.position.start.offset + node.meta.backticks)
        const contentEnd = Math.max(contentStart, node.position.end.offset - node.meta.backticks)
        emitSyntheticText(ctx, ' ', { fg: style.fg })
        emitSourceText(ctx, node.text, contentStart, contentEnd, {
          fg: ctx.palette.inlineCodeFg,
          attributes: TextAttributes.BOLD,
        })
        emitSyntheticText(ctx, ' ', { fg: style.fg })
        break
      }
      case 'link':
        renderInline(node.content, ctx, { ...style, fg: style.fg ?? ctx.palette.linkFg }, env)
        break
      case 'image':
        emitSourceText(ctx, `[${node.attrs.alt || 'image'}]`, node.position.start.offset, node.position.end.offset, {
          fg: ctx.palette.linkFg,
          attributes: style.attributes,
        })
        break
      case 'softBreak':
      case 'hardBreak':
        newLine(ctx)
        applyPrefixToCurrentLine(ctx, env)
        break
    }
  }
}

function renderParagraph(node: ParagraphNode, ctx: CSTRenderContext, env: BlockEnv): void {
  applyPrefixToCurrentLine(ctx, env)
  renderInline(node.content, ctx, env.blockquoteDepth > 0 ? { fg: ctx.palette.blockquoteTextFg } : {}, env)
  finishBlock(ctx)
}

function renderHeading(node: HeadingNode, ctx: CSTRenderContext, env: BlockEnv): void {
  applyPrefixToCurrentLine(ctx, env)
  renderInline(node.content, ctx, {
    fg: ctx.palette.headingFg[node.attrs.level] ?? ctx.palette.headingFg[6],
    attributes: TextAttributes.BOLD,
  }, env)
  finishBlock(ctx)
}

function highlightPlainCodeLine(
  text: string,
  lineStart: number,
  ctx: CSTRenderContext,
): Line {
  return splitByHighlights(text || ' ', lineStart, lineStart + text.length, { fg: ctx.palette.codeTextFg }, ctx.highlightRanges)
}

function highlightColoredCodeLine(
  line: Line,
  lineStart: number,
  ctx: CSTRenderContext,
): Line {
  const out: Line = []
  let cursor = lineStart
  for (const seg of line) {
    const segText = seg.text
    out.push(...splitByHighlights(segText, cursor, cursor + segText.length, seg, ctx.highlightRanges))
    cursor += segText.length
  }
  if (out.length === 0) {
    out.push({ text: ' ', fg: ctx.palette.syntax.default })
  }
  return out
}

function renderCodeBlock(node: CodeBlockNode, ctx: CSTRenderContext): void {
  flushTextChunk(ctx)

  const rawText = node.content?.map((t) => t.text).join('') ?? ''
  const codeContent = rawText.endsWith('\n') ? rawText.slice(0, -1) : rawText
  const codeStart = node.content?.[0]?.position.start.offset ?? node.position.start.offset

  if (node.attrs.language === 'mermaid') {
    try {
      const ascii = renderMermaidAscii(codeContent.trim(), {
        paddingX: 2,
        paddingY: 2,
        boxBorderPadding: 0,
      })
      const asciiLineCount = ascii.split('\n').length
      ctx.chunks.push({
        type: 'mermaid',
        ascii,
        startLine: ctx.currentLineNumber,
        endLine: ctx.currentLineNumber + asciiLineCount - 1,
        startChar: node.position.start.offset,
        endChar: node.position.end.offset,
      })
      ctx.currentLineNumber += asciiLineCount
      ctx.currentLines = [[]]
      return
    } catch {
      // fallback to normal code chunk
    }
  }

  const highlighted = node.attrs.language ? tryHighlight(codeContent, node.attrs.language, ctx.palette.syntax) : null

  let lines: Line[]
  if (highlighted) {
    let offset = codeStart
    lines = highlighted.map((line, lineIdx) => {
      const sourceLineText = codeContent.split('\n')[lineIdx] ?? ''
      const rendered = highlightColoredCodeLine(line.length === 0 ? [{ text: ' ', fg: ctx.palette.syntax.default }] : line, offset, ctx)
      offset += sourceLineText.length + 1
      return rendered
    })
  } else {
    let offset = codeStart
    lines = codeContent.split('\n').map((lineText) => {
      const line = highlightPlainCodeLine(lineText, offset, ctx)
      offset += lineText.length + 1
      return line
    })
  }

  ctx.chunks.push({
    type: 'code',
    lang: node.attrs.language ?? undefined,
    lines,
    rawCode: codeContent,
    startLine: ctx.currentLineNumber,
    endLine: ctx.currentLineNumber + Math.max(lines.length, 1) - 1,
    startChar: node.position.start.offset,
    endChar: node.position.end.offset,
  })
  ctx.currentLineNumber += Math.max(lines.length, 1)
  ctx.currentLines = [[]]
}

function renderHorizontalRule(ctx: CSTRenderContext, env: BlockEnv): void {
  applyPrefixToCurrentLine(ctx, env)
  const width = Math.max(10, Math.min(ctx.codeBlockWidth, 80))
  emitSyntheticText(ctx, '─'.repeat(width), { fg: ctx.palette.dividerFg })
  finishBlock(ctx)
}

function renderHtmlBlock(node: HtmlBlockNode, ctx: CSTRenderContext, env: BlockEnv): void {
  applyPrefixToCurrentLine(ctx, env)
  emitSourceText(ctx, node.content, node.position.start.offset, node.position.end.offset)
  finishBlock(ctx)
}

function renderDefinition(node: DefinitionNode, ctx: CSTRenderContext, env: BlockEnv): void {
  applyPrefixToCurrentLine(ctx, env)
  const title = node.title ? ` "${node.title}"` : ''
  emitSourceText(ctx, `[${node.label}]: ${node.url}${title}`, node.position.start.offset, node.position.end.offset)
  finishBlock(ctx)
}

function renderBlankLines(node: BlankLinesNode, ctx: CSTRenderContext): void {
  for (let i = 0; i < node.count; i++) {
    emitBlankLine(ctx)
  }
}

function withNestedPrefix(env: BlockEnv, prefix: string, style?: Pick<StyledSegment, 'fg' | 'bg' | 'attributes'>): BlockEnv {
  return {
    blockquoteDepth: env.blockquoteDepth,
    linePrefix: `${env.linePrefix ?? ''}${prefix}`,
    linePrefixStyle: style,
  }
}

function renderListItem(
  item: BulletItemNode | OrderedItemNode | TaskItemNode,
  marker: string,
  ctx: CSTRenderContext,
  env: BlockEnv,
): void {
  const baseStyle = env.blockquoteDepth > 0 ? { fg: ctx.palette.blockquoteTextFg } : undefined
  const firstPrefix = `${env.linePrefix ?? ''}${marker}`
  const continuationPrefix = `${env.linePrefix ?? ''}${' '.repeat(stringWidth(marker))}`

  item.content.forEach((contentItem, index) => {
    const childEnv: BlockEnv = {
      blockquoteDepth: env.blockquoteDepth,
      linePrefix: index === 0 ? firstPrefix : continuationPrefix,
      linePrefixStyle: index === 0
        ? { fg: ctx.palette.listBulletFg }
        : baseStyle,
    }
    renderListItemContentItem(contentItem, ctx, childEnv)
  })
}

function renderListItemContentItem(item: ListItemContentItemNode, ctx: CSTRenderContext, env: BlockEnv): void {
  renderBlock(item.content, ctx, env)
}

function renderListBreak(node: ListItemBreakNode, ctx: CSTRenderContext): void {
  for (let i = 0; i < node.meta.blankLines.length; i++) {
    emitBlankLine(ctx)
  }
}

function renderBulletList(node: BulletListNode, ctx: CSTRenderContext, env: BlockEnv): void {
  for (const child of node.content) {
    if (child.type === 'listItemBreak') {
      renderListBreak(child, ctx)
    } else {
      renderListItem(child, `${node.meta.marker} `, ctx, env)
    }
  }
  finishBlock(ctx)
}

function renderOrderedList(node: OrderedListNode, ctx: CSTRenderContext, env: BlockEnv): void {
  for (const child of node.content) {
    if (child.type === 'listItemBreak') {
      renderListBreak(child, ctx)
    } else {
      renderListItem(child, `${child.meta.number}${node.meta.delimiter} `, ctx, env)
    }
  }
  finishBlock(ctx)
}

function renderTaskList(node: TaskListNode, ctx: CSTRenderContext, env: BlockEnv): void {
  for (const child of node.content) {
    if (child.type === 'listItemBreak') {
      renderListBreak(child, ctx)
      continue
    }

    let marker = ''
    if (node.meta.style === 'ordered' && child.meta.number) {
      marker = `${child.meta.number}${node.meta.delimiter} `
    } else if (node.meta.style === 'bullet') {
      marker = `${node.meta.marker} `
    }
    marker += child.attrs.checked ? '[x] ' : '[ ] '
    renderListItem(child, marker, ctx, env)
  }
  finishBlock(ctx)
}

function renderBlockquote(node: BlockquoteNode, ctx: CSTRenderContext, env: BlockEnv): void {
  const quoteEnv: BlockEnv = {
    blockquoteDepth: env.blockquoteDepth + 1,
    linePrefix: `${env.linePrefix ?? ''}> `,
    linePrefixStyle: { fg: ctx.palette.blockquoteBorderFg },
  }

  for (const child of node.content) {
    if (child.type === 'blockquoteItemBreak') {
      for (let i = 0; i < child.meta.blankLines.length; i++) {
        emitBlankLine(ctx)
      }
    } else {
      renderBlockquoteItem(child, ctx, quoteEnv)
    }
  }

  finishBlock(ctx)
}

function renderBlockquoteItem(node: BlockquoteItemNode, ctx: CSTRenderContext, env: BlockEnv): void {
  renderBlock(node.content, ctx, env)
}

function renderRootBlock(node: RootBlockNode, ctx: CSTRenderContext, env: BlockEnv): void {
  switch (node.type) {
    case 'paragraph':
      renderParagraph(node, ctx, env)
      break
    case 'heading':
      renderHeading(node, ctx, env)
      break
    case 'codeBlock':
      renderCodeBlock(node, ctx)
      break
    case 'horizontalRule':
      renderHorizontalRule(ctx, env)
      break
    case 'blockquote':
      renderBlockquote(node, ctx, env)
      break
    case 'bulletList':
      renderBulletList(node, ctx, env)
      break
    case 'orderedList':
      renderOrderedList(node, ctx, env)
      break
    case 'taskList':
      renderTaskList(node, ctx, env)
      break
    case 'table':
      applyPrefixToCurrentLine(ctx, env)
      renderTableNode(node, ctx)
      finishBlock(ctx)
      break
    case 'htmlBlock':
      renderHtmlBlock(node, ctx, env)
      break
    case 'definition':
      renderDefinition(node, ctx, env)
      break
    case 'image':
      applyPrefixToCurrentLine(ctx, env)
      emitSourceText(ctx, `[${node.attrs.alt || 'image'}]`, node.position.start.offset, node.position.end.offset, {
        fg: ctx.palette.linkFg,
      })
      finishBlock(ctx)
      break
  }
}

function renderBlock(node: DocumentContentNode | BlockquoteContentNode, ctx: CSTRenderContext, env: BlockEnv): void {
  if (node.type === 'blankLines') {
    renderBlankLines(node, ctx)
    return
  }
  renderRootBlock(node as RootBlockNode, ctx, env)
}

function renderDocumentItem(item: DocumentItemNode, ctx: CSTRenderContext): void {
  renderBlock(item.content, ctx, { blockquoteDepth: 0 })
}

function renderDocument(doc: DocumentNode, ctx: CSTRenderContext): void {
  ctx.currentLines = [[]]
  for (const item of doc.content) {
    renderDocumentItem(item, ctx)
  }
  flushTextChunk(ctx)
}

export function parseMarkdownToChunks(
  markdown: string,
  options: MarkdownRenderOptions = {},
): MarkdownChunk[] {
  const palette = buildMergedPalette(options.palette)
  const codeBlockWidth = options.codeBlockWidth ?? 80
  const highlightRanges = [...(options.highlightRanges ?? [])].sort((a, b) => a.start - b.start)

  if (!markdown || markdown.trim() === '') {
    return [{ type: 'text', lines: [], startLine: 0, endLine: 0, startChar: 0, endChar: 0 }]
  }

  try {
    const doc = parseMarkdown(markdown)
    const ctx: CSTRenderContext = {
      source: markdown,
      palette,
      codeBlockWidth,
      highlightRanges,
      chunks: [],
      currentLines: [[]],
      currentLineNumber: 0,
      minSourceOffset: null,
      maxSourceOffset: null,
    }

    renderDocument(doc, ctx)

    return ctx.chunks.length > 0
      ? ctx.chunks
      : [{ type: 'text', lines: [[]], startLine: 0, endLine: 0, startChar: 0, endChar: 0 }]
  } catch (error) {
    console.error('Failed to parse markdown', error)
    return [{ type: 'text', lines: [[{ text: markdown }]], startLine: 0, endLine: 0, startChar: 0, endChar: markdown.length }]
  }
}

export function renderMarkdownContent(markdown: string, options: MarkdownRenderOptions = {}): ReactNode {
  if (!markdown || markdown.trim() === '') return markdown || ''
  const chunks = parseMarkdownToChunks(markdown, options)
  const palette = buildMergedPalette(options.palette)
  const parts: ReactNode[] = []

  for (const chunk of chunks) {
    if (chunk.type === 'text') {
      parts.push(convertLinesToReactNodes(chunk.lines))
    } else if (chunk.type === 'code') {
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