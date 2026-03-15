import { createLowlight, common } from 'lowlight'
import type { Element, RootContent, Text } from 'hast'
import { renderMermaidAscii } from 'beautiful-mermaid'
import stringWidth from 'string-width'
import type {
  BlankLinesNode,
  BlockquoteContentNode,
  BlockquoteItemNode,
  BlockquoteNode,
  BulletItemNode,
  BulletListNode,
  CodeBlockNode,
  DefinitionNode,
  DocumentContentNode,
  DocumentItemNode,
  DocumentNode,
  HeadingNode,
  HorizontalRuleNode,
  HtmlBlockNode,
  InlineNode,
  LinkNode,
  OrderedItemNode,
  OrderedListNode,
  ParagraphNode,
  RootBlockNode,
  TableCellNode,
  TableNode,
  TaskItemNode,
  TaskListNode,
} from '@magnitude/markdown-cst/src/schema'
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

const ARTIFACT_REF_RE = /\[\[([^\]|#]+?)(?:#([^\]|]+?))?(?:\|([^\]]+?))?\]\]/g

function sourceOf(node: { position: { start: { offset: number }; end: { offset: number } } }): SourceRange {
  return {
    start: node.position.start.offset,
    end: node.position.end.offset,
  }
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

function splitTextForArtifactRefs(
  text: string,
  baseStart: number,
): Array<{ text: string; start: number; end: number; ref?: Span['ref'] }> {
  const result: Array<{ text: string; start: number; end: number; ref?: Span['ref'] }> = []
  const regex = new RegExp(ARTIFACT_REF_RE.source, 'g')
  let lastIndex = 0
  let match: RegExpExecArray | null

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      result.push({
        text: text.slice(lastIndex, match.index),
        start: baseStart + lastIndex,
        end: baseStart + match.index,
      })
    }
    result.push({
      text: match[0],
      start: baseStart + match.index,
      end: baseStart + regex.lastIndex,
      ref: {
        name: match[1],
        section: match[2] || undefined,
        label: match[3] || undefined,
      },
    })
    lastIndex = regex.lastIndex
  }

  if (lastIndex < text.length) {
    result.push({
      text: text.slice(lastIndex),
      start: baseStart + lastIndex,
      end: baseStart + text.length,
    })
  }

  return result.length > 0 ? result : [{ text, start: baseStart, end: baseStart + text.length }]
}

function splitByHighlights(
  text: string,
  sourceStart: number,
  sourceEnd: number,
  style: Partial<Span>,
  highlights: HighlightRange[],
): Span[] {
  if (!text) return []
  if (highlights.length === 0 || sourceEnd <= sourceStart) return [{ text, ...style }]

  const boundaries = new Set<number>([sourceStart, sourceEnd])
  for (const range of highlights) {
    const overlapStart = Math.max(sourceStart, range.start)
    const overlapEnd = Math.min(sourceEnd, range.end)
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
    const part = text.slice(start - sourceStart, end - sourceStart)
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

function applyTextWithRefsAndHighlights(
  text: string,
  start: number,
  style: Partial<Span>,
  highlights: HighlightRange[],
): Span[] {
  const pieces = splitTextForArtifactRefs(text, start)
  return mergeAdjacentSpans(
    pieces.flatMap((piece) =>
      splitByHighlights(piece.text, piece.start, piece.end, { ...style, ref: piece.ref ?? style.ref }, highlights),
    ),
  )
}

function extractInlinePlainText(nodes: readonly InlineNode[] | undefined): string {
  if (!nodes) return ''
  return nodes
    .map((node) => {
      switch (node.type) {
        case 'text':
          return node.text
        case 'image':
          return `[${node.attrs.alt || 'image'}]`
        case 'inlineCode':
          return node.text
        case 'softBreak':
        case 'hardBreak':
          return ' '
        case 'emphasis':
        case 'strong':
        case 'strikethrough':
        case 'link':
          return extractInlinePlainText(node.content)
      }
    })
    .join('')
}

function renderInline(
  nodes: readonly InlineNode[] | undefined,
  style: InlineStyle,
  highlights: HighlightRange[],
  palette: MarkdownPalette,
): Span[] {
  if (!nodes) return []

  const spans: Span[] = []

  for (const node of nodes) {
    switch (node.type) {
      case 'text':
        spans.push(
          ...applyTextWithRefsAndHighlights(node.text, node.position.start.offset, style, highlights),
        )
        break
      case 'emphasis':
        spans.push(...renderInline(node.content, { ...style, italic: true }, highlights, palette))
        break
      case 'strong':
        spans.push(...renderInline(node.content, { ...style, bold: true }, highlights, palette))
        break
      case 'strikethrough':
        spans.push(...renderInline(node.content, { ...style, dim: true }, highlights, palette))
        break
      case 'inlineCode':
        spans.push(
          ...splitByHighlights(
            ` ${node.text} `,
            node.position.start.offset,
            node.position.end.offset,
            { ...style, fg: palette.inlineCodeFg, bold: true },
            highlights,
          ),
        )
        break
      case 'link':
        spans.push(
          ...renderInline(
            node.content,
            { ...style, fg: style.fg ?? palette.linkFg },
            highlights,
            palette,
          ),
        )
        break
      case 'image':
        spans.push(
          ...splitByHighlights(
            `[${node.attrs.alt || 'image'}]`,
            node.position.start.offset,
            node.position.end.offset,
            { ...style, fg: palette.linkFg },
            highlights,
          ),
        )
        break
      case 'softBreak':
        spans.push({ text: ' ', ...style })
        break
      case 'hardBreak':
        spans.push({ text: '\n', ...style })
        break
    }
  }

  return mergeAdjacentSpans(spans)
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

function highlightToLines(nodes: RootContent[], syntax: SyntaxColors): Span[][] {
  const lines: Span[][] = [[]]

  const walk = (node: RootContent, inheritedColor?: string): void => {
    if (node.type === 'text') {
      const textNode = node as Text
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
      for (const child of el.children) {
        walk(child as RootContent, color)
      }
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
    return highlightToLines(result.children as RootContent[], syntax)
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

  const out: Span[] = []
  let cursor = lineStart
  for (const seg of line) {
    out.push(...splitByHighlights(seg.text, cursor, cursor + seg.text.length, seg, highlights))
    cursor += seg.text.length
  }
  return mergeAdjacentSpans(out)
}

function renderCodeBlockNode(node: CodeBlockNode, options: RenderOptions): CodeBlock | MermaidBlock {
  const rawText = node.content?.map((t) => t.text).join('') ?? ''
  const rawCode = rawText.endsWith('\n') ? rawText.slice(0, -1) : rawText
  const codeStart = node.content?.[0]?.position.start.offset ?? node.position.start.offset

  if (node.attrs.language === 'mermaid') {
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
    } catch {
      // fall through to normal code block
    }
  }

  const syntaxLines = node.attrs.language ? tryHighlight(rawCode, node.attrs.language, options.palette.syntax) : null
  const rawLines = rawCode.split('\n')
  let offset = codeStart

  const lines = rawLines.map((lineText, idx) => {
    const line = syntaxLines?.[idx] ?? [{ text: lineText || ' ', fg: options.palette.codeTextFg }]
    const rendered = applyHighlightsToCodeLine(
      line,
      offset,
      lineText,
      options.highlights ?? [],
      options.palette.codeTextFg,
    )
    offset += lineText.length + 1
    return rendered
  })

  return {
    type: 'code',
    language: node.attrs.language ?? undefined,
    lines,
    rawCode,
    source: sourceOf(node),
  }
}

function renderParagraphNode(node: ParagraphNode, options: RenderOptions, fg?: string): ParagraphBlock {
  return {
    type: 'paragraph',
    content: renderInline(node.content, { fg }, options.highlights ?? [], options.palette),
    source: sourceOf(node),
  }
}

function renderHeadingNode(node: HeadingNode, options: RenderOptions): HeadingBlock {
  const content = renderInline(
    node.content,
    { fg: options.palette.headingFg[node.attrs.level], bold: true },
    options.highlights ?? [],
    options.palette,
  )
  return {
    type: 'heading',
    level: node.attrs.level,
    content,
    slug: slugify(spansToText(content)),
    source: sourceOf(node),
  }
}

function renderTableCell(cell: TableCellNode, options: RenderOptions): Span[] {
  return renderInline(cell.content[0]?.content, {}, options.highlights ?? [], options.palette)
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

  if (totalNatural <= effectiveAvailable) {
    return naturalWidths
  }

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

function renderTableNode(node: TableNode, options: RenderOptions): TableBlock {
  const rows = node.content.map((row) => row.content.map((cell) => renderTableCell(cell, options)))
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

function renderListItemContent(
  content: readonly { content: ParagraphNode | BulletListNode | OrderedListNode | TaskListNode | BlankLinesNode }[],
  options: RenderOptions,
): Block[] {
  return content.flatMap((item) => renderNodeToBlocks(item.content, options))
}

function renderBulletListNode(node: BulletListNode, options: RenderOptions): ListBlock {
  return {
    type: 'list',
    style: 'bullet',
    items: node.content
      .filter((item): item is BulletItemNode => item.type === 'bulletItem')
      .map((item) => ({
        marker: `${node.meta.marker} `,
        markerFg: options.palette.listBulletFg,
        content: renderListItemContent(item.content, options),
      })),
    source: sourceOf(node),
  }
}

function renderOrderedListNode(node: OrderedListNode, options: RenderOptions): ListBlock {
  return {
    type: 'list',
    style: 'ordered',
    items: node.content
      .filter((item): item is OrderedItemNode => item.type === 'orderedItem')
      .map((item) => ({
        marker: `${item.meta.number}${node.meta.delimiter} `,
        markerFg: options.palette.listBulletFg,
        content: renderListItemContent(item.content, options),
      })),
    source: sourceOf(node),
  }
}

function renderTaskListNode(node: TaskListNode, options: RenderOptions): ListBlock {
  return {
    type: 'list',
    style: 'task',
    items: node.content
      .filter((item): item is TaskItemNode => item.type === 'taskItem')
      .map((item) => {
        const prefix =
          node.meta.style === 'ordered' && item.meta.number
            ? `${item.meta.number}${node.meta.delimiter} `
            : node.meta.style === 'bullet'
              ? `${node.meta.marker} `
              : ''
        return {
          marker: `${prefix}${item.attrs.checked ? '[x] ' : '[ ] '}`,
          markerFg: options.palette.listBulletFg,
          checked: item.attrs.checked,
          content: renderListItemContent(item.content, options),
        }
      }),
    source: sourceOf(node),
  }
}

function renderBlockquoteNode(node: BlockquoteNode, options: RenderOptions): BlockquoteBlock {
  const content: Block[] = []
  for (const child of node.content) {
    if (child.type === 'blockquoteItem') {
      content.push(...renderNodeToBlocks(child.content, options, options.palette.blockquoteTextFg))
    } else {
      content.push({ type: 'spacer', lines: child.meta.blankLines.length || 1 })
    }
  }

  return {
    type: 'blockquote',
    content,
    source: sourceOf(node),
  }
}

function renderNodeToBlocks(
  node: DocumentContentNode | BlockquoteContentNode,
  options: RenderOptions,
  inheritedParagraphFg?: string,
): Block[] {
  switch (node.type) {
    case 'blankLines':
      return [{ type: 'spacer', lines: node.count }]
    case 'paragraph':
      return [renderParagraphNode(node, options, inheritedParagraphFg)]
    case 'heading':
      return [renderHeadingNode(node, options)]
    case 'codeBlock':
      return [renderCodeBlockNode(node, options)]
    case 'horizontalRule':
      return [{ type: 'divider', source: sourceOf(node) }]
    case 'bulletList':
      return [renderBulletListNode(node, options)]
    case 'orderedList':
      return [renderOrderedListNode(node, options)]
    case 'taskList':
      return [renderTaskListNode(node, options)]
    case 'blockquote':
      return [renderBlockquoteNode(node, options)]
    case 'table':
      return [renderTableNode(node, options)]
    case 'htmlBlock':
      return [
        {
          type: 'paragraph',
          content: splitByHighlights(
            node.content,
            node.position.start.offset,
            node.position.end.offset,
            {},
            options.highlights ?? [],
          ),
          source: sourceOf(node),
        },
      ]
    case 'definition':
      return []
    case 'image':
      return [
        {
          type: 'paragraph',
          content: splitByHighlights(
            `[${node.attrs.alt || 'image'}]`,
            node.position.start.offset,
            node.position.end.offset,
            { fg: options.palette.linkFg },
            options.highlights ?? [],
          ),
          source: sourceOf(node),
        },
      ]
  }
}

export function renderDocumentItemToBlocks(item: DocumentItemNode, options: RenderOptions): Block[] {
  return renderNodeToBlocks(item.content, options)
}

export function renderDocumentToBlocks(doc: DocumentNode, options: RenderOptions): Block[] {
  return doc.content.flatMap((item) => renderDocumentItemToBlocks(item, options))
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
      return [
        block.headers.map(spansToText).join(' | '),
        ...block.rows.map((row) => row.map(spansToText).join(' | ')),
      ].join('\n')
  }
}

export function extractInlineText(nodes: readonly InlineNode[] | undefined): string {
  return extractInlinePlainText(nodes)
}

export type { MarkdownPalette }