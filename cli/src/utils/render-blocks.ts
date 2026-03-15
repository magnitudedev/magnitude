import { createLowlight, common } from 'lowlight'
import type { Element, RootContent as HastRootContent, Text as HastText } from 'hast'
import { renderMermaidAscii } from 'beautiful-mermaid'
import stringWidth from 'string-width'
import type {
  Blockquote,
  Code,
  Definition,
  Heading,
  Html,
  Image,
  Link,
  List,
  ListItem as MdastListItem,
  Paragraph,
  Root,
  RootContent,
  Table,
  TableCell,
  Text,
} from 'mdast'
import type { MarkdownPalette, SyntaxColors } from './markdown-content-renderer'

const lowlight = createLowlight(common)

export interface SourceRange {
  start: number
  end: number
}

export interface Span {
  text: string
  fg?: string
  bg?: string
  bold?: boolean
  italic?: boolean
  dim?: boolean
  ref?: { name: string; section?: string; label?: string }
}

export interface HighlightRange {
  start: number
  end: number
  backgroundColor: string
}

export interface ParagraphBlock {
  type: 'paragraph'
  content: Span[]
  source: SourceRange
}

export interface HeadingBlock {
  type: 'heading'
  level: 1 | 2 | 3 | 4 | 5 | 6
  content: Span[]
  slug: string
  source: SourceRange
}

export interface CodeBlock {
  type: 'code'
  language?: string
  lines: Span[][]
  rawCode: string
  source: SourceRange
}

export interface ListItem {
  marker: string
  markerFg?: string
  checked?: boolean
  content: Block[]
}

export interface ListBlock {
  type: 'list'
  style: 'bullet' | 'ordered' | 'task'
  items: ListItem[]
  source: SourceRange
}

export interface BlockquoteBlock {
  type: 'blockquote'
  content: Block[]
  source: SourceRange
}

export interface TableBlock {
  type: 'table'
  headers: Span[][]
  rows: Span[][][]
  columnWidths: number[]
  source: SourceRange
}

export interface DividerBlock {
  type: 'divider'
  source: SourceRange
}

export interface MermaidBlock {
  type: 'mermaid'
  ascii: string
  source: SourceRange
}

export interface SpacerBlock {
  type: 'spacer'
  lines: number
}

export type Block =
  | ParagraphBlock
  | HeadingBlock
  | CodeBlock
  | ListBlock
  | BlockquoteBlock
  | TableBlock
  | DividerBlock
  | MermaidBlock
  | SpacerBlock

export interface RenderOptions {
  palette: MarkdownPalette
  codeBlockWidth?: number
  highlights?: HighlightRange[]
}

interface InlineStyle {
  fg?: string
  bold?: boolean
  italic?: boolean
  dim?: boolean
}

type PhrasingNode = Text | Link | Image | import('mdast').WikiLink | { type: 'strong' | 'emphasis' | 'delete'; children: PhrasingNode[] } | { type: 'inlineCode'; value: string; position?: any } | { type: 'break' }

function sourceOf(node: { position?: { start?: { offset?: number }; end?: { offset?: number } } }): SourceRange {
  return {
    start: node.position?.start?.offset ?? 0,
    end: node.position?.end?.offset ?? 0,
  }
}

function sourceStart(node: { position?: { start?: { offset?: number } } }): number {
  return node.position?.start?.offset ?? 0
}

function sourceEnd(node: { position?: { end?: { offset?: number } } }): number {
  return node.position?.end?.offset ?? 0
}

function lineStart(node: { position?: { start?: { line?: number } } }): number {
  return node.position?.start?.line ?? 1
}

function lineEnd(node: { position?: { end?: { line?: number } } }): number {
  return node.position?.end?.line ?? lineStart(node)
}

export function slugify(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')
}

export function spansToText(spans: Span[]): string {
  return spans.map((s) => s.text).join('')
}

function mergeAdjacentSpans(spans: Span[]): Span[] {
  const result: Span[] = []
  for (const span of spans) {
    const prev = result[result.length - 1]
    if (
      prev &&
      prev.fg === span.fg &&
      prev.bg === span.bg &&
      prev.bold === span.bold &&
      prev.italic === span.italic &&
      prev.dim === span.dim &&
      prev.ref?.name === span.ref?.name &&
      prev.ref?.section === span.ref?.section &&
      prev.ref?.label === span.ref?.label
    ) {
      prev.text += span.text
    } else {
      result.push({ ...span })
    }
  }
  return result
}

function splitByHighlights(
  text: string,
  sourceStartOffset: number,
  sourceEndOffset: number,
  style: Partial<Span>,
  highlights: HighlightRange[],
): Span[] {
  if (!text) return []
  if (highlights.length === 0 || sourceEndOffset <= sourceStartOffset) return [{ text, ...style }]

  const boundaries = new Set<number>([sourceStartOffset, sourceEndOffset])
  for (const range of highlights) {
    const overlapStart = Math.max(sourceStartOffset, range.start)
    const overlapEnd = Math.min(sourceEndOffset, range.end)
    if (overlapStart < overlapEnd) {
      boundaries.add(overlapStart)
      boundaries.add(overlapEnd)
    }
  }

  const points = [...boundaries].sort((a, b) => a - b)
  const spans: Span[] = []

  for (let i = 0; i < points.length - 1; i++) {
    const start = points[i]
    const end = points[i + 1]
    if (end <= start) continue
    const part = text.slice(start - sourceStartOffset, end - sourceStartOffset)
    if (!part) continue

    const active = highlights.find((range) => start < range.end && end > range.start)
    spans.push({
      text: part,
      ...style,
      bg: active?.backgroundColor ?? style.bg,
    })
  }

  return spans.length > 0 ? spans : [{ text, ...style }]
}

function getSourceSlice(sourceText: string | undefined, node: { position?: { start?: { offset?: number }; end?: { offset?: number } } }): string | undefined {
  if (!sourceText) return undefined
  const start = sourceStart(node)
  const end = sourceEnd(node)
  return end >= start ? sourceText.slice(start, end) : undefined
}



function renderInline(
  nodes: readonly PhrasingNode[] | undefined,
  style: InlineStyle,
  highlights: HighlightRange[],
  palette: MarkdownPalette,
  sourceText?: string,
): Span[] {
  if (!nodes) return []

  const spans: Span[] = []

  for (const node of nodes) {
    switch (node.type) {
      case 'text':
        spans.push(...splitByHighlights(node.value, sourceStart(node), sourceEnd(node), style, highlights))
        break
      case 'emphasis':
        spans.push(...renderInline(node.children, { ...style, italic: true }, highlights, palette, sourceText))
        break
      case 'strong':
        spans.push(...renderInline(node.children, { ...style, bold: true }, highlights, palette, sourceText))
        break
      case 'delete':
        spans.push(...renderInline(node.children, { ...style, dim: true }, highlights, palette, sourceText))
        break
      case 'inlineCode':
        spans.push(
          ...splitByHighlights(
            ` ${node.value} `,
            sourceStart(node),
            sourceEnd(node),
            { ...style, fg: palette.inlineCodeFg, bold: true },
            highlights,
          ),
        )
        break
      case 'link':
        spans.push(...renderInline(node.children as PhrasingNode[], { ...style, fg: style.fg ?? palette.linkFg }, highlights, palette, sourceText))
        break
      case 'image':
        spans.push(
          ...splitByHighlights(
            `[${node.alt || 'image'}]`,
            sourceStart(node),
            sourceEnd(node),
            { ...style, fg: palette.linkFg },
            highlights,
          ),
        )
        break
      case 'break':
        spans.push({ text: '\n', ...style })
        break
      case 'wikiLink': {
        const target = (node as any).value ?? ''
        const alias = (node as any).data?.alias as string | undefined
        const [name, section] = target.split('#', 2)
        const hasExplicitLabel = alias != null && alias !== target
        const displayText = hasExplicitLabel ? alias : (getSourceSlice(sourceText, node) ?? `[[${target}]]`)
        const ref = { name, section: section || undefined, label: hasExplicitLabel ? alias : undefined }
        spans.push(
          ...splitByHighlights(displayText, sourceStart(node), sourceEnd(node), { ...style, ref }, highlights),
        )
        break
      }
    }
  }

  return mergeAdjacentSpans(spans)
}

function extractInlinePlainText(nodes: readonly PhrasingNode[] | undefined, sourceText?: string): string {
  if (!nodes) return ''
  return nodes
    .map((node) => {
      switch (node.type) {
        case 'text':
          return node.value
        case 'image':
          return `[${node.alt || 'image'}]`
        case 'inlineCode':
          return node.value
        case 'break':
          return ' '
        case 'emphasis':
        case 'strong':
        case 'delete':
        case 'link':
          return extractInlinePlainText((node as any).children, sourceText)
        case 'wikiLink':
          return getSourceSlice(sourceText, node) ?? `[[${node.value ?? ''}]]`
        default:
          return ''
      }
    })
    .join('')
}

function hljsClassToColor(classNames: string[], syntax: SyntaxColors): string {
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

function highlightToLines(nodes: HastRootContent[], syntax: SyntaxColors): Span[][] {
  const lines: Span[][] = [[]]

  const walk = (node: HastRootContent, inheritedColor?: string): void => {
    if (node.type === 'text') {
      const textNode = node as HastText
      const parts = textNode.value.split('\n')
      parts.forEach((part, idx) => {
        if (idx > 0) lines.push([])
        if (part) lines[lines.length - 1].push({ text: part, fg: inheritedColor ?? syntax.default })
      })
      return
    }

    if (node.type === 'element') {
      const el = node as Element
      const classNames = (el.properties?.className as string[]) ?? []
      const color = hljsClassToColor(classNames, syntax)
      for (const child of el.children) walk(child as HastRootContent, color)
    }
  }

  for (const node of nodes) walk(node)
  return lines
}

function tryHighlight(code: string, lang: string, syntax: SyntaxColors): Span[][] | null {
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
  if (!lowlight.registered(normalizedLang)) return null

  try {
    const result = lowlight.highlight(normalizedLang, code)
    return highlightToLines(result.children as HastRootContent[], syntax)
  } catch {
    return null
  }
}

function applyHighlightsToCodeLine(
  line: Span[],
  lineStart: number,
  lineText: string,
  highlights: HighlightRange[],
  fallbackFg: string,
): Span[] {
  if (line.length === 0) {
    return splitByHighlights(lineText || ' ', lineStart, lineStart + lineText.length, { fg: fallbackFg }, highlights)
  }

  if (highlights.length > 0) {
    const boundaries = new Set<number>([lineStart, lineStart + lineText.length])
    for (const range of highlights) {
      const start = Math.max(lineStart, range.start)
      const end = Math.min(lineStart + lineText.length, range.end)
      if (start < end) {
        boundaries.add(start)
        boundaries.add(end)
      }
    }

    const points = [...boundaries].sort((a, b) => a - b)
    const spans: Span[] = []

    const colorAt = (offset: number): string => {
      let cursor = lineStart
      for (const seg of line) {
        const end = cursor + seg.text.length
        if (offset < end) return seg.fg ?? fallbackFg
        cursor = end
      }
      return line[line.length - 1]?.fg ?? fallbackFg
    }

    for (let i = 0; i < points.length - 1; i++) {
      const start = points[i]
      const end = points[i + 1]
      if (end <= start) continue
      const text = lineText.slice(start - lineStart, end - lineStart)
      if (!text) continue
      const active = highlights.find((range) => start < range.end && end > range.start)
      spans.push({
        text,
        fg: colorAt(start),
        bg: active?.backgroundColor,
      })
    }

    return mergeAdjacentSpans(spans)
  }

  const out: Span[] = []
  let cursor = lineStart
  for (const seg of line) {
    out.push(...splitByHighlights(seg.text, cursor, cursor + seg.text.length, seg, highlights))
    cursor += seg.text.length
  }
  return mergeAdjacentSpans(out)
}

function renderCodeBlockStart(node: Code, sourceText?: string): number {
  const blockStart = sourceStart(node)
  const raw = getSourceSlice(sourceText, node)
  if (!raw) return blockStart
  const firstNewline = raw.indexOf('\n')
  return firstNewline === -1 ? blockStart : blockStart + firstNewline + 1
}

function renderCodeBlockNode(node: Code, options: RenderOptions, sourceText?: string): CodeBlock | MermaidBlock {
  const rawCode = node.value.endsWith('\n') ? node.value.slice(0, -1) : node.value
  const codeStart = renderCodeBlockStart(node, sourceText)

  if (node.lang === 'mermaid') {
    try {
      return {
        type: 'mermaid',
        ascii: renderMermaidAscii(rawCode.trim(), {
          paddingX: 2,
          paddingY: 2,
          boxBorderPadding: 0,
        }),
        source: sourceOf(node),
      }
    } catch {}
  }

  const syntaxLines = node.lang ? tryHighlight(rawCode, node.lang, options.palette.syntax) : null
  const rawLines = rawCode.split('\n')
  let offset = codeStart

  const lines = rawLines.map((lineText, idx) => {
    const line = syntaxLines?.[idx] ?? [{ text: lineText || ' ', fg: options.palette.codeTextFg }]
    const rendered = applyHighlightsToCodeLine(line, offset, lineText, options.highlights ?? [], options.palette.codeTextFg)
    offset += lineText.length + 1
    return rendered
  })

  return {
    type: 'code',
    language: node.lang ?? undefined,
    lines,
    rawCode,
    source: sourceOf(node),
  }
}

function renderParagraphNode(node: Paragraph, options: RenderOptions, sourceText?: string, fg?: string): ParagraphBlock {
  return {
    type: 'paragraph',
    content: renderInline(node.children as PhrasingNode[], { fg }, options.highlights ?? [], options.palette, sourceText),
    source: sourceOf(node),
  }
}

function renderHeadingNode(node: Heading, options: RenderOptions, sourceText?: string): HeadingBlock {
  const level = node.depth as 1 | 2 | 3 | 4 | 5 | 6
  const content = renderInline(
    node.children as PhrasingNode[],
    { fg: options.palette.headingFg[level], bold: true },
    options.highlights ?? [],
    options.palette,
    sourceText,
  )
  return {
    type: 'heading',
    level,
    content,
    slug: slugify(spansToText(content)),
    source: sourceOf(node),
  }
}

function renderTableCell(cell: TableCell, options: RenderOptions, sourceText?: string): Span[] {
  return renderInline(cell.children as PhrasingNode[], {}, options.highlights ?? [], options.palette, sourceText)
}

function computeTableColumnWidths(allRows: Span[][][], availableWidth: number): number[] {
  const numCols = Math.max(...allRows.map((row) => row.length), 0)
  if (numCols === 0) return []

  const naturalWidths: number[] = Array(numCols).fill(3)
  for (const row of allRows) {
    for (let i = 0; i < row.length; i++) {
      const cellWidth = stringWidth(spansToText(row[i] ?? []))
      naturalWidths[i] = Math.max(naturalWidths[i], cellWidth)
    }
  }

  const separatorWidth = 3
  const numSeparators = numCols - 1
  const totalNatural = naturalWidths.reduce((a, b) => a + b, 0) + numSeparators * separatorWidth
  const effectiveAvailable = Math.max(20, availableWidth - 2)

  if (totalNatural <= effectiveAvailable) return naturalWidths

  const availableForContent = effectiveAvailable - numSeparators * separatorWidth
  const totalNaturalContent = naturalWidths.reduce((a, b) => a + b, 0)
  const scale = availableForContent / totalNaturalContent
  const columnWidths = naturalWidths.map((w) => Math.max(3, Math.floor(w * scale)))

  let usedWidth = columnWidths.reduce((a, b) => a + b, 0)
  let remaining = availableForContent - usedWidth
  for (let i = 0; i < columnWidths.length && remaining > 0; i++) {
    if (columnWidths[i] < naturalWidths[i]) {
      const add = Math.min(remaining, naturalWidths[i] - columnWidths[i])
      columnWidths[i] += add
      remaining -= add
    }
  }

  return columnWidths
}

function renderTableNode(node: Table, options: RenderOptions, sourceText?: string): TableBlock {
  const rows = node.children.map((row) => row.children.map((cell) => renderTableCell(cell, options, sourceText)))
  const availableWidth = options.codeBlockWidth ?? 80
  const columnWidths = computeTableColumnWidths(rows, availableWidth)

  return {
    type: 'table',
    headers: rows[0] ?? [],
    rows: rows.slice(1),
    columnWidths,
    source: sourceOf(node),
  }
}

function getListStyle(node: List): 'bullet' | 'ordered' | 'task' {
  if (node.children.some((item) => typeof item.checked === 'boolean')) return 'task'
  return node.ordered ? 'ordered' : 'bullet'
}

function renderListItemContent(node: MdastListItem, options: RenderOptions, sourceText?: string): Block[] {
  return renderNodesToBlocks(node.children as RootContent[], options, sourceText)
}

function renderListNode(node: List, options: RenderOptions, sourceText?: string): ListBlock {
  const style = getListStyle(node)
  return {
    type: 'list',
    style,
    items: node.children.map((item, index) => ({
      marker:
        style === 'task'
          ? `${node.ordered ? `${(node.start ?? 1) + index}. ` : '- '}${item.checked ? '[x] ' : '[ ] '}`
          : node.ordered
            ? `${(node.start ?? 1) + index}. `
            : '- ',
      markerFg: options.palette.listBulletFg,
      checked: typeof item.checked === 'boolean' ? item.checked : undefined,
      content: renderListItemContent(item, options, sourceText),
    })),
    source: sourceOf(node),
  }
}

function renderBlockquoteNode(node: Blockquote, options: RenderOptions, sourceText?: string): BlockquoteBlock {
  const content = renderNodesToBlocks(node.children as RootContent[], options, sourceText, options.palette.blockquoteTextFg).flatMap((block) =>
    block.type === 'table'
      ? [{
          type: 'paragraph' as const,
          content: splitByHighlights(
            extractPlainTextFromBlock(block),
            block.source.start,
            block.source.end,
            { fg: options.palette.blockquoteTextFg },
            options.highlights ?? [],
          ),
          source: block.source,
        }]
      : [block],
  )

  return {
    type: 'blockquote',
    content,
    source: sourceOf(node),
  }
}

function renderNodeToBlocks(node: RootContent, options: RenderOptions, sourceText?: string, inheritedParagraphFg?: string): Block[] {
  switch (node.type) {
    case 'paragraph':
      return [renderParagraphNode(node, options, sourceText, inheritedParagraphFg)]
    case 'heading':
      return [renderHeadingNode(node, options, sourceText)]
    case 'code':
      return [renderCodeBlockNode(node, options, sourceText)]
    case 'thematicBreak':
      return [{ type: 'divider', source: sourceOf(node) }]
    case 'list':
      return [renderListNode(node, options, sourceText)]
    case 'blockquote':
      return [renderBlockquoteNode(node, options, sourceText)]
    case 'table':
      return [renderTableNode(node, options, sourceText)]
    case 'html':
      return [{
        type: 'paragraph',
        content: splitByHighlights(node.value, sourceStart(node), sourceEnd(node), {}, options.highlights ?? []),
        source: sourceOf(node),
      }]
    case 'definition':
      return []
    case 'image':
      return [{
        type: 'paragraph',
        content: splitByHighlights(`[${node.alt || 'image'}]`, sourceStart(node), sourceEnd(node), { fg: options.palette.linkFg }, options.highlights ?? []),
        source: sourceOf(node),
      }]
    default:
      return []
  }
}

function countBlankLinesBetween(previous: RootContent, next: RootContent): number {
  return Math.max(0, lineStart(next) - lineEnd(previous) - 1)
}

function renderNodesToBlocks(
  nodes: readonly RootContent[] | undefined,
  options: RenderOptions,
  sourceText?: string,
  inheritedParagraphFg?: string,
): Block[] {
  if (!nodes) return []
  const blocks: Block[] = []
  let previousRenderable: RootContent | null = null

  for (const node of nodes) {
    const rendered = renderNodeToBlocks(node, options, sourceText, inheritedParagraphFg)
    if (rendered.length === 0) continue

    if (previousRenderable) {
      const blankLines = countBlankLinesBetween(previousRenderable, node)
      if (blankLines > 0) blocks.push({ type: 'spacer', lines: blankLines })
    }

    blocks.push(...rendered)
    previousRenderable = node
  }

  return blocks
}

export function renderDocumentItemToBlocks(item: RootContent, options: RenderOptions): Block[] {
  return renderNodeToBlocks(item, options)
}

export function renderDocumentToBlocks(doc: Root, options: RenderOptions): Block[] {
  const sourceText = (doc as any).data?.source as string | undefined
  return renderNodesToBlocks(doc.children, options, sourceText)
}

export function extractHeadingSlugsFromBlocks(blocks: Block[]): string[] {
  return blocks.filter((b): b is HeadingBlock => b.type === 'heading').map((b) => b.slug)
}

export function extractPlainTextFromBlock(block: Block): string {
  switch (block.type) {
    case 'paragraph':
    case 'heading':
      return spansToText(block.content)
    case 'code':
      return block.rawCode
    case 'mermaid':
      return block.ascii
    case 'divider':
      return '---'
    case 'spacer':
      return '\n'.repeat(block.lines)
    case 'blockquote':
      return block.content.map(extractPlainTextFromBlock).join('\n')
    case 'list':
      return block.items.map((item) => item.content.map(extractPlainTextFromBlock).join('\n')).join('\n')
    case 'table':
      return [block.headers.map(spansToText).join(' | '), ...block.rows.map((row) => row.map(spansToText).join(' | '))].join('\n')
  }
}

export function extractInlineText(nodes: readonly PhrasingNode[] | undefined): string {
  return extractInlinePlainText(nodes)
}

export type { MarkdownPalette }