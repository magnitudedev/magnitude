import { parseMarkdownToMdast } from '../markdown/parse'
import {
  renderDocumentToBlocks,
  spansToText,
  type Block,
  type RenderOptions,
  type Span,
} from '../markdown/blocks'
import { buildMarkdownColorPalette, chatThemes } from './theme'

export const theme = chatThemes.dark
export const palette = buildMarkdownColorPalette(theme)
export const baseOptions: RenderOptions = { palette, codeBlockWidth: 80 }

export function renderBlocks(markdown: string, overrides?: Partial<RenderOptions>): Block[] {
  return renderDocumentToBlocks(parseMarkdownToMdast(markdown), { ...baseOptions, ...overrides })
}

export function getSingleBlock(markdown: string): Block {
  const blocks = renderBlocks(markdown)
  if (blocks.length !== 1) {
    throw new Error(`expected exactly 1 block, got ${blocks.length}`)
  }
  return blocks[0]!
}

export function findBlocks(markdown: string, type: Block['type']): Block[] {
  return renderBlocks(markdown).filter((block) => block.type === type)
}

export function blockTypes(blocks: Block[]): string[] {
  return blocks.map((block) => block.type)
}

export function collectText(blockOrBlocks: Block | Block[]): string {
  const blocks = Array.isArray(blockOrBlocks) ? blockOrBlocks : [blockOrBlocks]
  return blocks.map(extractTextFromBlock).join('\n')
}

function extractTextFromBlock(block: Block): string {
  switch (block.type) {
    case 'paragraph':
    case 'heading':
      return spansToText(block.content)
    case 'code':
      return block.lines.map(spansToText).join('\n')
    case 'table':
      return [
        block.headers.map(spansToText).join(' | '),
        ...block.rows.map((row) => row.map(spansToText).join(' | ')),
      ].join('\n')
    case 'list':
      return block.items.map((item) => collectText(item.content)).join('\n')
    case 'blockquote':
      return collectText(block.content)
    case 'divider':
      return '---'
    case 'mermaid':
      return block.ascii
    case 'spacer':
      return '\n'.repeat(block.lines)
  }
}

export function normalizeSpan(span: Span) {
  return {
    text: span.text,
    fg: span.fg,
    bg: span.bg,
    bold: span.bold,
    italic: span.italic,
    dim: span.dim,
    ref: span.ref,
  }
}

export function normalizeBlock(block: Block): unknown {
  switch (block.type) {
    case 'paragraph':
      return {
        type: block.type,
        source: block.source,
        content: block.content.map(normalizeSpan),
      }
    case 'heading':
      return {
        type: block.type,
        level: block.level,
        slug: block.slug,
        source: block.source,
        content: block.content.map(normalizeSpan),
      }
    case 'code':
      return {
        type: block.type,
        language: block.language,
        rawCode: block.rawCode,
        source: block.source,
        lines: block.lines.map((line) => line.map(normalizeSpan)),
      }
    case 'list':
      return {
        type: block.type,
        style: block.style,
        source: block.source,
        items: block.items.map((item) => ({
          marker: item.marker,
          markerFg: item.markerFg,
          checked: item.checked,
          content: item.content.map(normalizeBlock),
        })),
      }
    case 'blockquote':
      return {
        type: block.type,
        source: block.source,
        content: block.content.map(normalizeBlock),
      }
    case 'table':
      return {
        type: block.type,
        source: block.source,
        alignments: block.alignments,
        headers: block.headers.map((row) => row.map(normalizeSpan)),
        rows: block.rows.map((row) => row.map((cell) => cell.map(normalizeSpan))),
      }
    case 'divider':
      return { type: block.type, source: block.source }
    case 'mermaid':
      return { type: block.type, source: block.source, ascii: block.ascii }
    case 'spacer':
      return { type: block.type, lines: block.lines }
  }
}

export function hasHighlight(block: Block): boolean {
  if (block.type === 'paragraph' || block.type === 'heading') {
    return block.content.some((span) => !!span.bg)
  }
  if (block.type === 'table') {
    return [...block.headers.flat(), ...block.rows.flat(2)].some((span) => !!span.bg)
  }
  if (block.type === 'code') {
    return block.lines.flat().some((span) => !!span.bg)
  }
  if (block.type === 'list') {
    return block.items.some((item) => item.content.some(hasHighlight))
  }
  if (block.type === 'blockquote') {
    return block.content.some(hasHighlight)
  }
  return false
}

export function collectHighlightedText(blocks: Block[]): string[] {
  const text: string[] = []

  const visit = (block: Block): void => {
    if (block.type === 'paragraph' || block.type === 'heading') {
      text.push(...block.content.filter((span) => !!span.bg).map((span) => span.text))
      return
    }
    if (block.type === 'table') {
      text.push(
        ...[...block.headers.flat(), ...block.rows.flat(2)]
          .filter((span) => !!span.bg)
          .map((span) => span.text),
      )
      return
    }
    if (block.type === 'code') {
      text.push(...block.lines.flat().filter((span) => !!span.bg).map((span) => span.text))
      return
    }
    if (block.type === 'list') {
      block.items.forEach((item) => item.content.forEach(visit))
      return
    }
    if (block.type === 'blockquote') {
      block.content.forEach(visit)
    }
  }

  blocks.forEach(visit)
  return text
}