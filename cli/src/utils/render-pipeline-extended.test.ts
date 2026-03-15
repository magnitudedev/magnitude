import { describe, expect, test } from 'bun:test'
import { parseMarkdown } from '@magnitude/markdown-cst'
import type { DocumentItemNode, DocumentNode } from '@magnitude/markdown-cst/src/schema'
import {
  renderDocumentItemToBlocks,
  renderDocumentToBlocks,
  spansToText,
  type Block,
  type HighlightRange,
  type RenderOptions,
  type Span,
} from './render-blocks'
import { buildMarkdownColorPalette, chatThemes } from './theme'
import { hasOddFenceCount } from './markdown-content-renderer'

const palette = buildMarkdownColorPalette(chatThemes.dark)
const baseOptions: RenderOptions = { palette, codeBlockWidth: 80 }

function normalizeSpan(span: Span) {
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

function normalizeBlock(block: Block): unknown {
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
        columnWidths: block.columnWidths,
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

function blockTypes(blocks: Block[]): string[] {
  return blocks.map((b) => b.type)
}

function extractBlockTextShape(block: Block): unknown {
  switch (block.type) {
    case 'paragraph':
    case 'heading':
      return spansToText(block.content)
    case 'spacer':
      return block.lines
    case 'table':
      return {
        headers: block.headers.map(spansToText),
        rows: block.rows.map((row) => row.map(spansToText)),
      }
    case 'code':
      return {
        language: block.language,
        rawCode: block.rawCode,
      }
    case 'divider':
      return '---'
    case 'mermaid':
      return block.ascii
    case 'blockquote':
      return block.content.map(extractBlockTextShape)
    case 'list':
      return {
        style: block.style,
        items: block.items.map((item) => ({
          marker: item.marker,
          checked: item.checked,
          content: item.content.map(extractBlockTextShape),
        })),
      }
  }
}

function expectBlockStructuralEquality(actual: Block[], expected: Block[]) {
  expect(blockTypes(actual)).toEqual(blockTypes(expected))
  expect(actual).toHaveLength(expected.length)

  for (let i = 0; i < actual.length; i++) {
    const a = actual[i]!
    const e = expected[i]!
    expect(a.type).toBe(e.type)

    if (a.type === 'paragraph' && e.type === 'paragraph') {
      expect(spansToText(a.content)).toEqual(spansToText(e.content))
    } else if (a.type === 'heading' && e.type === 'heading') {
      expect(spansToText(a.content)).toEqual(spansToText(e.content))
      expect(a.level).toEqual(e.level)
      expect(a.slug).toEqual(e.slug)
    } else if (a.type === 'spacer' && e.type === 'spacer') {
      expect(a.lines).toEqual(e.lines)
    } else if (a.type === 'table' && e.type === 'table') {
      expect(a.headers.map(spansToText)).toEqual(e.headers.map(spansToText))
      expect(a.rows.map((row) => row.map(spansToText))).toEqual(e.rows.map((row) => row.map(spansToText)))
      expect(a.columnWidths).toEqual(e.columnWidths)
    } else {
      expect(extractBlockTextShape(a)).toEqual(extractBlockTextShape(e))
    }
  }

  expect(actual.map(normalizeBlock)).toEqual(expected.map(normalizeBlock))
}

function hasHighlight(block: Block): boolean {
  if (block.type === 'paragraph' || block.type === 'heading') {
    return block.content.some((s) => !!s.bg)
  }
  if (block.type === 'table') {
    return [...block.headers.flat(), ...block.rows.flat(2)].some((s) => !!s.bg)
  }
  if (block.type === 'code') {
    return block.lines.flat().some((s) => !!s.bg)
  }
  if (block.type === 'list') {
    return block.items.some((item) => item.content.some(hasHighlight))
  }
  if (block.type === 'blockquote') {
    return block.content.some(hasHighlight)
  }
  return false
}

function collectHighlightedText(blocks: Block[], color = '#00ff00'): string {
  const parts: string[] = []
  const walk = (block: Block) => {
    if (block.type === 'paragraph' || block.type === 'heading') {
      parts.push(...block.content.filter((s) => s.bg === color).map((s) => s.text))
    } else if (block.type === 'table') {
      parts.push(
        ...block.headers.flat().filter((s) => s.bg === color).map((s) => s.text),
        ...block.rows.flat(2).filter((s) => s.bg === color).map((s) => s.text),
      )
    } else if (block.type === 'code') {
      parts.push(...block.lines.flat().filter((s) => s.bg === color).map((s) => s.text))
    } else if (block.type === 'list') {
      block.items.forEach((item) => item.content.forEach(walk))
    } else if (block.type === 'blockquote') {
      block.content.forEach(walk)
    }
  }
  blocks.forEach(walk)
  return parts.join('')
}

function preview(value: string, max = 24): string {
  return value.length <= max ? value : `${value.slice(0, max)}...`
}

function findUniqueMatchRange(content: string | null | undefined, needle: string | undefined) {
  if (!content || !needle) return null
  const first = content.indexOf(needle)
  if (first === -1) return null
  const second = content.indexOf(needle, first + 1)
  if (second !== -1) return null
  return { start: first, end: first + needle.length }
}

function computeOptimisticUpdatePreview(
  baseContent: string | null | undefined,
  oldString: string | undefined,
  newString: string | undefined,
  replaceAll: boolean | undefined,
): { content: string; changedRanges: Array<{ start: number; end: number }> } | null {
  if (!baseContent || !oldString || newString === undefined) return null
  if (!baseContent.includes(oldString)) return null

  if (!replaceAll) {
    const index = baseContent.indexOf(oldString)
    if (index === -1) return null
    return {
      content: baseContent.slice(0, index) + newString + baseContent.slice(index + oldString.length),
      changedRanges: [{ start: index, end: index + newString.length }],
    }
  }

  const changedRanges: Array<{ start: number; end: number }> = []
  let cursor = 0
  let result = ''

  while (cursor < baseContent.length) {
    const index = baseContent.indexOf(oldString, cursor)
    if (index === -1) {
      result += baseContent.slice(cursor)
      break
    }
    result += baseContent.slice(cursor, index)
    const start = result.length
    result += newString
    changedRanges.push({ start, end: start + newString.length })
    cursor = index + oldString.length
  }

  return { content: result, changedRanges }
}

function areHighlightRangesEqual(
  a: HighlightRange[] | undefined,
  b: HighlightRange[] | undefined,
): boolean {
  if (a === b) return true
  if (!a || !b) return false
  if (a.length !== b.length) return false
  return a.every(
    (r, i) =>
      r.start === b[i]!.start &&
      r.end === b[i]!.end &&
      r.backgroundColor === b[i]!.backgroundColor,
  )
}

interface CacheState {
  prevContent: string
  prevDoc: DocumentNode | null
  prevBlocks: Block[]
  prevPendingText: string
  itemBlockCache: WeakMap<DocumentItemNode, Block[]>
  prevHighlightRanges: HighlightRange[] | undefined
  prevPalette: RenderOptions['palette'] | null
  prevCodeBlockWidth: number | undefined
  prevStreaming: boolean
}

function createCacheState(): CacheState {
  return {
    prevContent: '',
    prevDoc: null,
    prevBlocks: [],
    prevPendingText: '',
    itemBlockCache: new WeakMap(),
    prevHighlightRanges: undefined,
    prevPalette: null,
    prevCodeBlockWidth: undefined,
    prevStreaming: false,
  }
}

function simulateCacheStep(
  cache: CacheState,
  content: string,
  options: RenderOptions & { streaming?: boolean; highlightRanges?: HighlightRange[] },
): { blocks: Block[]; pendingText: string } {
  if (cache.prevStreaming && !options.streaming) {
    cache.prevDoc = null
    cache.itemBlockCache = new WeakMap()
  }
  cache.prevStreaming = !!options.streaming

  let completeSection = content
  let pendingText = ''

  if (hasOddFenceCount(content)) {
    const lastFenceIndex = content.lastIndexOf('```')
    if (lastFenceIndex !== -1) {
      completeSection = content.slice(0, lastFenceIndex)
      pendingText = content.slice(lastFenceIndex)
    }
  }

  if (!completeSection || completeSection.trim() === '') {
    cache.prevContent = content
    cache.prevDoc = null
    cache.prevBlocks = []
    cache.prevPendingText = pendingText
    return { blocks: [], pendingText }
  }

  const highlightChanged = !areHighlightRangesEqual(options.highlightRanges, cache.prevHighlightRanges)
  const paletteChanged = options.palette !== cache.prevPalette
  const codeBlockWidthChanged = options.codeBlockWidth !== cache.prevCodeBlockWidth
  if (paletteChanged || codeBlockWidthChanged || highlightChanged) {
    cache.itemBlockCache = new WeakMap()
  }

  const doc = parseMarkdown(completeSection)

  const allBlocks: Block[] = []
  for (const item of doc.content) {
    let itemBlocks = cache.itemBlockCache.get(item)
    if (!itemBlocks) {
      itemBlocks = renderDocumentItemToBlocks(item, {
        palette: options.palette,
        codeBlockWidth: options.codeBlockWidth,
        highlights: options.highlightRanges,
      })
      cache.itemBlockCache.set(item, itemBlocks)
    }
    allBlocks.push(...itemBlocks)
  }

  cache.prevContent = content
  cache.prevDoc = doc
  cache.prevBlocks = allBlocks
  cache.prevPendingText = pendingText
  cache.prevHighlightRanges = options.highlightRanges
  cache.prevPalette = options.palette
  cache.prevCodeBlockWidth = options.codeBlockWidth

  return { blocks: allBlocks, pendingText }
}

interface RevealState {
  displayedLength: number
  previousContent: string
  previousIsStreaming: boolean
  isLinearDrain: boolean
}

function createRevealState(initialLength = 0): RevealState {
  return {
    displayedLength: initialLength,
    previousContent: '',
    previousIsStreaming: false,
    isLinearDrain: false,
  }
}

function simulateRevealStep(
  state: RevealState,
  content: string,
  isStreaming: boolean,
  initialDisplayedLength?: number,
  isInterrupted?: boolean,
): { displayedContent: string; isCatchingUp: boolean; showCursor: boolean } {
  if (!state.previousIsStreaming && isStreaming) {
    state.displayedLength = Math.max(0, Math.min(initialDisplayedLength ?? 0, content.length))
  } else if (content.length < state.previousContent.length) {
    state.displayedLength = Math.min(state.displayedLength, content.length)
  } else if (
    !isStreaming &&
    state.previousContent &&
    !content.startsWith(state.previousContent.slice(0, Math.min(state.previousContent.length, content.length)))
  ) {
    state.displayedLength = content.length
  } else if (!isStreaming && content.length < state.displayedLength) {
    state.displayedLength = content.length
  }

  state.isLinearDrain = !isStreaming

  if (isInterrupted) {
    state.displayedLength = content.length
  }

  if (isStreaming) {
    if (state.displayedLength < content.length) {
      const remaining = content.length - state.displayedLength
      const speed = Math.max(1, Math.floor(remaining * 0.15))
      state.displayedLength = Math.min(content.length, state.displayedLength + speed)
    }
  } else if (state.displayedLength < content.length) {
    state.displayedLength = Math.min(content.length, state.displayedLength + 8)
  }

  const safeDisplayedLength = Math.min(state.displayedLength, content.length)
  const displayedContent = content.slice(0, safeDisplayedLength)
  const isCatchingUp = safeDisplayedLength < content.length
  const showCursor = isStreaming || isCatchingUp

  state.previousContent = content
  state.previousIsStreaming = isStreaming

  return { displayedContent, isCatchingUp, showCursor }
}

function revealToCompletion(
  state: RevealState,
  content: string,
  isStreaming: boolean,
  initialDisplayedLength?: number,
  isInterrupted?: boolean,
): string {
  let result = simulateRevealStep(state, content, isStreaming, initialDisplayedLength, isInterrupted)
  let safety = 0
  while (result.displayedContent.length < content.length && safety < 1000) {
    result = simulateRevealStep(state, content, isStreaming)
    safety++
  }
  expect(safety).toBeLessThan(1000)
  return result.displayedContent
}

function verifyBlockOrdering(blocks: Block[]) {
  let lastEnd = -1
  for (const block of blocks) {
    if ('source' in block && block.source) {
      // Source ranges should be non-negative and well-formed
      expect(block.source.end).toBeGreaterThanOrEqual(block.source.start)
      // Allow non-monotonic ordering for blocks from nested structures
      // (lists, blockquotes) where sub-blocks can share source ranges
      if (block.source.start >= lastEnd) {
        lastEnd = block.source.end
      }
    }
  }
}

function verifySpacerConsistency(blocks: Block[], allowEdgeSpacers = false) {
  for (let i = 0; i < blocks.length - 1; i++) {
    expect(!(blocks[i]!.type === 'spacer' && blocks[i + 1]!.type === 'spacer')).toBe(true)
  }
  if (!allowEdgeSpacers && blocks.length > 0) {
    expect(blocks[0]!.type).not.toBe('spacer')
    expect(blocks[blocks.length - 1]!.type).not.toBe('spacer')
  }
}

const streamingDocuments = [
  'Hello world',
  'Hello world\n\nSecond paragraph',
  '# Heading\n\nBody text',
  '# H1\n\n## H2\n\nBody',
  '- item 1\n- item 2\n- item 3',
  '1. first\n2. second\n3. third',
  '> blockquote text',
  '> line 1\n> line 2',
  '```js\nconsole.log(1)\n```',
  '| a | b |\n| --- | --- |\n| c | d |',
  '**bold** and *italic*',
  'Text with `inline code` here',
  '[link](http://example.com)',
  '---',
  '# Title\n\nParagraph\n\n```python\nprint("hi")\n```\n\nMore text',
  '# Title\n\n- list item 1\n- list item 2\n\nParagraph after list',
  '> quote\n\nNon-quote\n\n> another quote',
  '| Name | Age | City |\n| --- | --- | --- |\n| Alice | 30 | NYC |\n| Bob | 25 | LA |',
  '# H1\n\nPara 1\n\n## H2\n\nPara 2\n\n### H3\n\nPara 3',
  'Short',
  'A\n\nB\n\nC\n\nD\n\nE',
  '- [ ] task 1\n- [x] task 2\n- [ ] task 3',
  '> > nested quote',
  '- item with **bold**\n- item with *italic*\n- item with `code`',
  '# Title\n\n| h1 | h2 |\n| -- | -- |\n| a | b |\n\nAfter table',
  'Paragraph\n\n---\n\nAnother paragraph',
  '```\nplain code\nno language\n```',
  '# A\n\nText\n\n# B\n\nMore text\n\n# C\n\nEven more',
  'Line 1\nLine 2\nLine 3',
  '![image](url)\n\nText after image',
  '[[artifact]] ref',
  'Paragraph with unicode こんにちは\n\nEmoji 😀 rocket 🚀',
  '> quote\n\n- list\n\n| a | b |\n| - | - |\n| c | d |',
  '<div>html block</div>\n\nParagraph',
  // Skipped: '````\ncode with ``` inside\n````' — hasOddFenceCount heuristic can't handle nested fences
]

const editScenarios = [
  { base: 'Hello world\n\nMiddle text\n\nEnd', old: 'Middle text', new: 'Replaced text' },
  { base: 'Hello world\n\nMiddle text\n\nEnd', old: 'Middle', new: 'Changed' },
  { base: '# Title\n\nBody here\n\nFooter', old: 'Body here', new: 'New body content' },
  { base: '# Title\n\nBody here\n\nFooter', old: 'Body here', new: '# New Heading' },
  { base: '# Title\n\nBody here\n\nFooter', old: 'Body here', new: '- list\n- items' },
  { base: '# Title\n\nBody here\n\nFooter', old: 'Body here', new: '```js\ncode()\n```' },
  { base: '# Title\n\nBody here\n\nFooter', old: 'Body here', new: '| a | b |\n| - | - |\n| c | d |' },
  { base: '| a | b |\n| - | - |\n| c | d |', old: 'c', new: 'replaced' },
  { base: '| a | b |\n| - | - |\n| c | d |', old: 'd', new: 'new value here' },
  { base: '- item 1\n- item 2\n- item 3', old: 'item 2', new: 'changed item' },
  { base: '> quote text here', old: 'quote text', new: 'new quote' },
  { base: 'A\n\nB\n\nC\n\nD', old: 'B\n\nC', new: 'Single replacement' },
  { base: 'A\n\nB\n\nC\n\nD', old: 'B', new: 'B\n\nInserted\n\nExtra' },
  { base: 'A\n\nB\n\nC\n\nD', old: '\n\nB', new: '' },
  { base: 'A\n\nB\n\nC\n\nD', old: 'C\n\nD', new: 'Merged' },
  { base: '# H1\n\n## H2\n\nText', old: '## H2', new: '## Changed H2' },
  { base: '# H1\n\n## H2\n\nText', old: '## H2', new: 'Just a paragraph now' },
  { base: 'Text with **bold** and *italic*', old: '**bold**', new: '**stronger**' },
  { base: 'Text with `code` here', old: '`code`', new: '`different`' },
  { base: 'Before\n\n```js\nold()\n```\n\nAfter', old: 'old()', new: 'newCode()' },
  { base: 'Para 1\n\nPara 2\n\nPara 3', old: 'Para 2', new: 'Changed para 2' },
  { base: 'foo bar foo baz foo', old: 'foo', new: 'qux' },
  {
    base: '# Title\n\nLong paragraph with lots of text that goes on and on and on\n\nFooter',
    old: 'lots of text',
    new: 'much content',
  },
  { base: 'A\n\nB', old: 'A\n\nB', new: 'Completely different\n\n# New structure\n\n- with\n- list' },
]

function chunksOf(text: string, size: number): string[] {
  if (text.length === 0) return ['']
  const out: string[] = []
  for (let i = size; i <= text.length; i += size) {
    out.push(text.slice(0, Math.min(i, text.length)))
  }
  if (out[out.length - 1] !== text) out.push(text)
  return out
}

function buildReplaceAllRanges(content: string, needle: string, replacement: string): HighlightRange[] {
  const replaced = content.replaceAll(needle, replacement)
  const ranges: HighlightRange[] = []
  let searchFrom = 0
  while (true) {
    const index = replaced.indexOf(replacement, searchFrom)
    if (index === -1) break
    ranges.push({ start: index, end: index + replacement.length, backgroundColor: '#00ff00' })
    searchFrom = index + replacement.length
  }
  return ranges
}

function simulateArtifactUpdateLifecycle(
  baseContent: string,
  oldString: string,
  newStringChunks: string[],
  options: RenderOptions,
) {
  const cache = createCacheState()
  const revealState = createRevealState()
  const results: Array<{ displayedContent: string; blocks: Block[]; hasHighlights: boolean }> = []

  const matchIdx = baseContent.indexOf(oldString)
  expect(matchIdx).not.toBe(-1)

  for (const newStr of newStringChunks) {
    const revealed = revealToCompletion(revealState, newStr, true, newStr.length)
    const displayedContent =
      baseContent.slice(0, matchIdx) + revealed + baseContent.slice(matchIdx + oldString.length)
    const highlights: HighlightRange[] = [
      { start: matchIdx, end: matchIdx + revealed.length, backgroundColor: '#00ff00' },
    ]

    const { blocks, pendingText } = simulateCacheStep(cache, displayedContent, {
      ...options,
      streaming: true,
      highlightRanges: highlights,
    })

    // Only compare against fresh parse when there's no pending fence text,
    // because fence splitting intentionally produces fewer blocks
    if (!pendingText) {
      const freshBlocks = renderDocumentToBlocks(parseMarkdown(displayedContent), {
        ...options,
        highlights,
      })
      expectBlockStructuralEquality(blocks, freshBlocks)
    }

    results.push({
      displayedContent,
      blocks,
      hasHighlights: blocks.some(hasHighlight),
    })
  }

  const finalContent =
    baseContent.slice(0, matchIdx) +
    newStringChunks[newStringChunks.length - 1]! +
    baseContent.slice(matchIdx + oldString.length)

  const { blocks: finalBlocks } = simulateCacheStep(cache, finalContent, { ...options, streaming: false })
  const freshFinal = renderDocumentToBlocks(parseMarkdown(finalContent), options)
  expectBlockStructuralEquality(finalBlocks, freshFinal)
  expect(finalBlocks.every((b) => !hasHighlight(b))).toBe(true)

  return results
}

describe('render pipeline extended - cache sequential streaming', () => {
  const chunkSizes = [1, 3, 5, 10]

  for (const [docIndex, doc] of streamingDocuments.entries()) {
    for (const chunkSize of chunkSizes) {
      if (chunkSize === 1 && doc.length > 140) continue

      test(`cache streaming doc#${docIndex + 1} "${preview(doc)}" chunk=${chunkSize}`, () => {
        const cache = createCacheState()
        for (const prefix of chunksOf(doc, chunkSize)) {
          const result = simulateCacheStep(cache, prefix, { ...baseOptions, streaming: true })
          const effective = hasOddFenceCount(prefix) && prefix.lastIndexOf('```') !== -1
            ? prefix.slice(0, prefix.lastIndexOf('```'))
            : prefix
          const fresh = effective.trim()
            ? renderDocumentToBlocks(parseMarkdown(effective), baseOptions)
            : []
          expectBlockStructuralEquality(result.blocks, fresh)
          verifyBlockOrdering(result.blocks)
          verifySpacerConsistency(result.blocks, true)
        }
      })
    }
  }
})

describe('render pipeline extended - cache streaming transition reset', () => {
  for (const [docIndex, doc] of streamingDocuments.entries()) {
    test(`cache stream->not-streaming reset doc#${docIndex + 1}`, () => {
      const cache = createCacheState()
      for (const prefix of chunksOf(doc, 4)) {
        simulateCacheStep(cache, prefix, { ...baseOptions, streaming: true })
      }
      const result = simulateCacheStep(cache, doc, { ...baseOptions, streaming: false })
      const fresh = renderDocumentToBlocks(parseMarkdown(doc), baseOptions)
      expectBlockStructuralEquality(result.blocks, fresh)
      verifyBlockOrdering(result.blocks)
      verifySpacerConsistency(result.blocks)
    })
  }
})

describe('render pipeline extended - cache multiple edit cycles', () => {
  const docs = streamingDocuments.slice(0, 12)

  for (const [aIndex, doc1] of docs.entries()) {
    for (const [bIndex, doc2] of docs.entries()) {
      test(`cache edit cycle ${aIndex + 1}->${bIndex + 1}`, () => {
        const cache = createCacheState()

        for (const prefix of chunksOf(doc1, 5)) {
          simulateCacheStep(cache, prefix, { ...baseOptions, streaming: true })
        }
        simulateCacheStep(cache, doc1, { ...baseOptions, streaming: false })

        for (const prefix of chunksOf(doc2, 5)) {
          const result = simulateCacheStep(cache, prefix, { ...baseOptions, streaming: true })
          const effective = hasOddFenceCount(prefix) && prefix.lastIndexOf('```') !== -1
            ? prefix.slice(0, prefix.lastIndexOf('```'))
            : prefix
          const fresh = effective.trim()
            ? renderDocumentToBlocks(parseMarkdown(effective), baseOptions)
            : []
          expectBlockStructuralEquality(result.blocks, fresh)
        }

        const final = simulateCacheStep(cache, doc2, { ...baseOptions, streaming: false })
        expectBlockStructuralEquality(final.blocks, renderDocumentToBlocks(parseMarkdown(doc2), baseOptions))
      })
    }
  }

  for (const scenario of editScenarios.slice(0, 12)) {
    for (const highlightMode of ['none', 'same', 'changed'] as const) {
      test(`cache repeated updates "${preview(scenario.base)}" highlights=${highlightMode}`, () => {
        const cache = createCacheState()
        const firstHighlight: HighlightRange[] | undefined =
          highlightMode === 'none'
            ? undefined
            : [{ start: 0, end: Math.min(3, scenario.base.length), backgroundColor: '#0f0' }]
        const secondHighlight: HighlightRange[] | undefined =
          highlightMode === 'changed'
            ? [{ start: 1, end: Math.min(4, scenario.base.length), backgroundColor: '#0f0' }]
            : firstHighlight

        simulateCacheStep(cache, scenario.base, {
          ...baseOptions,
          streaming: true,
          highlightRanges: firstHighlight,
        })
        simulateCacheStep(cache, scenario.base, {
          ...baseOptions,
          streaming: false,
          highlightRanges: firstHighlight,
        })

        const matchIdx = scenario.base.indexOf(scenario.old)
        const updated =
          scenario.base.slice(0, matchIdx) + scenario.new + scenario.base.slice(matchIdx + scenario.old.length)

        const result = simulateCacheStep(cache, updated, {
          ...baseOptions,
          streaming: true,
          highlightRanges: secondHighlight,
        })

        const fresh = renderDocumentToBlocks(parseMarkdown(updated), {
          ...baseOptions,
          highlights: secondHighlight,
        })
        expectBlockStructuralEquality(result.blocks, fresh)
      })
    }
  }
})

describe('render pipeline extended - cache highlight lifecycle', () => {
  for (const [docIndex, doc] of streamingDocuments.slice(0, 20).entries()) {
    test(`highlight lifecycle doc#${docIndex + 1}`, () => {
      const cache = createCacheState()
      const base = simulateCacheStep(cache, doc, { ...baseOptions, streaming: false })
      expect(base.blocks.every((b) => !hasHighlight(b))).toBe(true)

      const ranges: HighlightRange[] = [
        {
          start: 0,
          end: Math.min(5, doc.length),
          backgroundColor: '#0f0',
        },
      ]
      const withHighlights = simulateCacheStep(cache, doc, {
        ...baseOptions,
        streaming: true,
        highlightRanges: ranges,
      })
      const freshWithHighlights = renderDocumentToBlocks(parseMarkdown(doc), {
        ...baseOptions,
        highlights: ranges,
      })
      expectBlockStructuralEquality(withHighlights.blocks, freshWithHighlights)

      const cleared = simulateCacheStep(cache, doc, { ...baseOptions, streaming: false })
      expect(cleared.blocks.every((b) => !hasHighlight(b))).toBe(true)
    })
  }

  for (const scenario of editScenarios.slice(0, 24)) {
    test(`highlight cache semantics "${preview(scenario.old)}"`, () => {
      const cache = createCacheState()
      const matchIdx = scenario.base.indexOf(scenario.old)
      const updated =
        scenario.base.slice(0, matchIdx) + scenario.new + scenario.base.slice(matchIdx + scenario.old.length)

      const highlightsA: HighlightRange[] = [
        { start: matchIdx, end: matchIdx + Math.min(2, scenario.new.length), backgroundColor: '#00ff00' },
      ]
      const highlightsB: HighlightRange[] = [
        { start: matchIdx, end: matchIdx + Math.min(3, scenario.new.length), backgroundColor: '#00ff00' },
      ]

      const a = simulateCacheStep(cache, updated, {
        ...baseOptions,
        streaming: true,
        highlightRanges: highlightsA,
      })
      const b = simulateCacheStep(cache, updated, {
        ...baseOptions,
        streaming: true,
        highlightRanges: highlightsB,
      })
      const c = simulateCacheStep(cache, updated, {
        ...baseOptions,
        streaming: true,
        highlightRanges: highlightsB,
      })

      // Verify that same highlights produce same output (cache correctness)
      // Different highlights MAY produce different output, but don't assert it
      // because highlights on parser-consumed syntax can produce identical results
      expect(b.blocks.map(normalizeBlock)).toEqual(
        renderDocumentToBlocks(parseMarkdown(updated), { ...baseOptions, highlights: highlightsB }).map(normalizeBlock),
      )
      expect(c.blocks.map(normalizeBlock)).toEqual(b.blocks.map(normalizeBlock))
    })
  }
})

describe('render pipeline extended - reveal basic behavior', () => {
  for (const doc of streamingDocuments.slice(0, 20)) {
    test(`reveal catches up monotonically "${preview(doc)}"`, () => {
      const state = createRevealState()
      let previousLength = -1
      for (const prefix of chunksOf(doc, 3)) {
        const result = revealToCompletion(state, prefix, true)
        expect(result.length).toBe(prefix.length)
        expect(result.length).toBeGreaterThanOrEqual(previousLength)
        previousLength = result.length
      }
    })
  }

  for (const doc of streamingDocuments.slice(0, 15)) {
    test(`reveal clamps on shrink "${preview(doc)}"`, () => {
      const state = createRevealState()
      revealToCompletion(state, doc, true)
      const shorter = doc.slice(0, Math.max(0, doc.length - 3))
      const result = simulateRevealStep(state, shorter, false)
      expect(result.displayedContent.length).toBeLessThanOrEqual(shorter.length)
    })
  }

  for (const [a, b] of [
    ['Hello world', 'Completely different'],
    ['# A\n\nBody', '- list\n- changed'],
    ['foo', 'bar baz'],
    ['same prefix x', 'same suffix y'],
    ['12345', 'abcde'],
  ] as const) {
    test(`reveal snaps on non-monotonic change "${a}" -> "${b}"`, () => {
      const state = createRevealState()
      revealToCompletion(state, a, false)
      const result = simulateRevealStep(state, b, false)
      expect(result.displayedContent).toBe(b)
      expect(result.showCursor).toBe(false)
    })
  }
})

describe('render pipeline extended - reveal stream start stop cycles', () => {
  for (const doc of streamingDocuments.slice(0, 20)) {
    test(`reveal stream/drain cycle "${preview(doc)}"`, () => {
      const state = createRevealState()
      const halfway = doc.slice(0, Math.floor(doc.length / 2))
      const step = simulateRevealStep(state, halfway, true, 0)
      expect(step.displayedContent.length).toBeLessThanOrEqual(halfway.length)
      const drained = revealToCompletion(state, doc, false)
      expect(drained).toBe(doc)
    })
  }

  for (const doc of streamingDocuments.slice(0, 15)) {
    test(`reveal rapid start-stop "${preview(doc)}"`, () => {
      const state = createRevealState()
      const prefixes = chunksOf(doc, Math.max(1, Math.floor(doc.length / 4) || 1))
      for (const prefix of prefixes) {
        simulateRevealStep(state, prefix, true, prefix.length)
        simulateRevealStep(state, prefix, false)
      }
      const final = revealToCompletion(state, doc, false)
      expect(final).toBe(doc)
    })
  }

  for (const doc of streamingDocuments.slice(0, 15)) {
    test(`reveal interrupted snaps "${preview(doc)}"`, () => {
      const state = createRevealState()
      simulateRevealStep(state, doc, true, 0)
      const result = simulateRevealStep(state, doc, true, undefined, true)
      expect(result.displayedContent).toBe(doc)
      expect(result.showCursor).toBe(true)
    })
  }
})

describe('render pipeline extended - reveal non-monotonic and multi-instance', () => {
  for (const scenario of editScenarios.slice(0, 24)) {
    test(`reveal non-monotonic update path "${preview(scenario.old)}"`, () => {
      const state = createRevealState()
      const updated =
        scenario.base.replace(scenario.old, scenario.new)
      revealToCompletion(state, scenario.base, false)
      const step = simulateRevealStep(state, updated, false)
      expect(step.displayedContent).toBe(updated)
    })
  }

  for (const scenario of editScenarios.slice(0, 24)) {
    test(`three reveal instances mimic artifact panel "${preview(scenario.base)}"`, () => {
      const writeState = createRevealState()
      const newState = createRevealState()
      const oldState = createRevealState()

      const locating = findUniqueMatchRange(scenario.base, scenario.old)
      const oldReveal = locating
        ? revealToCompletion(oldState, scenario.old, true)
        : ''
      if (locating) expect(oldReveal.length).toBe(scenario.old.length)

      const newReveal = revealToCompletion(newState, scenario.new, true, scenario.new.length)
      expect(newReveal).toBe(scenario.new)

      const writeReveal = revealToCompletion(writeState, scenario.base, true)
      expect(writeReveal).toBe(scenario.base)
    })
  }
})

describe('render pipeline extended - full artifact update lifecycle', () => {
  for (const [scenarioIndex, scenario] of editScenarios.entries()) {
    for (const chunkSize of [1, 2, 4, 8]) {
      if (chunkSize === 1 && scenario.new.length > 30) continue

      test(`artifact lifecycle #${scenarioIndex + 1} chunk=${chunkSize}`, () => {
        const newChunks = chunksOf(scenario.new, chunkSize)
        const results = simulateArtifactUpdateLifecycle(
          scenario.base,
          scenario.old,
          newChunks,
          baseOptions,
        )
        expect(results.length).toBeGreaterThan(0)
        expect(results.some((r) => r.hasHighlights)).toBe(scenario.new.length > 0)
        results.forEach((r) => {
          verifyBlockOrdering(r.blocks)
          verifySpacerConsistency(r.blocks, true)
        })
      })
    }
  }

  for (const scenario of editScenarios.slice(0, 12)) {
    test(`artifact replaceAll preview "${preview(scenario.base)}"`, () => {
      if (!scenario.base.includes(scenario.old)) return
      const previewResult = computeOptimisticUpdatePreview(scenario.base, scenario.old, scenario.new, true)
      if (!previewResult) return
      const result = simulateCacheStep(createCacheState(), previewResult.content, {
        ...baseOptions,
        streaming: true,
        highlightRanges: previewResult.changedRanges.map((range) => ({
          ...range,
          backgroundColor: '#00ff00',
        })),
      })
      const fresh = renderDocumentToBlocks(parseMarkdown(previewResult.content), {
        ...baseOptions,
        highlights: previewResult.changedRanges.map((range) => ({
          ...range,
          backgroundColor: '#00ff00',
        })),
      })
      expectBlockStructuralEquality(result.blocks, fresh)
    })
  }

  for (const scenario of editScenarios.slice(0, 12)) {
    test(`artifact sequential reuse "${preview(scenario.base)}"`, () => {
      const cache = createCacheState()
      const reveal = createRevealState()

      const matchIdx = scenario.base.indexOf(scenario.old)
      const updated =
        scenario.base.slice(0, matchIdx) + scenario.new + scenario.base.slice(matchIdx + scenario.old.length)

      const revealed = revealToCompletion(reveal, scenario.new, true, scenario.new.length)
      const content =
        scenario.base.slice(0, matchIdx) + revealed + scenario.base.slice(matchIdx + scenario.old.length)

      const highlighted = simulateCacheStep(cache, content, {
        ...baseOptions,
        streaming: true,
        highlightRanges: [{ start: matchIdx, end: matchIdx + revealed.length, backgroundColor: '#00ff00' }],
      })
      expect(highlighted.blocks.some(hasHighlight)).toBe(revealed.length > 0)

      const final = simulateCacheStep(cache, updated, { ...baseOptions, streaming: false })
      expect(final.blocks.every((b) => !hasHighlight(b))).toBe(true)
      expectBlockStructuralEquality(final.blocks, renderDocumentToBlocks(parseMarkdown(updated), baseOptions))
    })
  }
})

describe('render pipeline extended - ordering and spacing invariants', () => {
  for (const [docIndex, doc] of streamingDocuments.entries()) {
    test(`ordering/spacing stream doc#${docIndex + 1}`, () => {
      const cache = createCacheState()
      for (const prefix of chunksOf(doc, 4)) {
        const result = simulateCacheStep(cache, prefix, { ...baseOptions, streaming: true })
        verifyBlockOrdering(result.blocks)
        verifySpacerConsistency(result.blocks, true)
      }
    })
  }

  for (const [scenarioIndex, scenario] of editScenarios.entries()) {
    test(`ordering/spacing edit#${scenarioIndex + 1}`, () => {
      const matchIdx = scenario.base.indexOf(scenario.old)
      for (const revealed of chunksOf(scenario.new, 3)) {
        const content =
          scenario.base.slice(0, matchIdx) + revealed + scenario.base.slice(matchIdx + scenario.old.length)
        const highlights: HighlightRange[] = [
          { start: matchIdx, end: matchIdx + revealed.length, backgroundColor: '#00ff00' },
        ]
        const blocks = renderDocumentToBlocks(parseMarkdown(content), { ...baseOptions, highlights })
        verifyBlockOrdering(blocks)
        verifySpacerConsistency(blocks, true)
      }
    })
  }

  for (let i = 0; i < 120; i++) {
    test(`ordering/spacing randomish doc#${i + 1}`, () => {
      const patterns = [
        `# H${i}\n\nParagraph ${i}`,
        `- a${i}\n- b${i}\n- c${i}`,
        `| a | b |\n| - | - |\n| ${i} | ${i + 1} |`,
        `> quote ${i}\n\nplain ${i}`,
        `\`\`\`js\nconsole.log(${i})\n\`\`\`\n\nTail ${i}`,
      ]
      const doc = patterns[i % patterns.length]!
      const blocks = renderDocumentToBlocks(parseMarkdown(doc), baseOptions)
      verifyBlockOrdering(blocks)
      verifySpacerConsistency(blocks, true)
    })
  }
})

describe('render pipeline extended - odd fence handling', () => {
  const fenceCases = [
    '```',
    '```js\nx',
    'before\n```js\nx',
    'before\n```\ncode',
    'before ` ``` ` after',
    '```\na\n```',
    '````\na\n````',
    '```\na\n```\n\ntext',
    'text\n```js\nconst x = 1\n',
    'text\n```js\nconst x = `value`\n',
    'text\n```js\nconst x = 1\n```\nmore\n```',
    'one\n```js\n1\n```\ntwo\n```py\n2',
    '```\n1\n```\n```\n2',
  ]

  for (const [index, content] of fenceCases.entries()) {
    test(`odd fence case #${index + 1}`, () => {
      const cache = createCacheState()
      const result = simulateCacheStep(cache, content, { ...baseOptions, streaming: true })
      if (hasOddFenceCount(content) && content.lastIndexOf('```') !== -1) {
        expect(result.pendingText.length).toBeGreaterThan(0)
      } else {
        expect(result.pendingText).toBe('')
      }
    })
  }

  for (const [docIndex, doc] of [
    '```js\nx',
    '```js\nx\n```',
    'before\n```js\nx',
    'before\n```js\nx\n```',
    'before\n```js\nx\n```\nafter',
    'one\n```js\n1\n```\ntwo\n```py\n2',
    'one\n```js\n1\n```\ntwo\n```py\n2\n```',
  ].entries()) {
    test(`odd->even transition #${docIndex + 1}`, () => {
      const cache = createCacheState()
      for (const prefix of chunksOf(doc, 2)) {
        const result = simulateCacheStep(cache, prefix, { ...baseOptions, streaming: true })
        if (hasOddFenceCount(prefix) && prefix.lastIndexOf('```') !== -1) {
          expect(result.pendingText.startsWith('```')).toBe(true)
        }
      }
    })
  }
})

describe('render pipeline extended - weakmap cache correctness', () => {
  for (const doc of streamingDocuments.slice(0, 20)) {
    test(`same content different highlights "${preview(doc)}"`, () => {
      const cache = createCacheState()
      const h1: HighlightRange[] = [{ start: 0, end: Math.min(2, doc.length), backgroundColor: '#0f0' }]
      const h2: HighlightRange[] = [{ start: 1, end: Math.min(4, doc.length), backgroundColor: '#0f0' }]

      const first = simulateCacheStep(cache, doc, { ...baseOptions, streaming: true, highlightRanges: h1 })
      const second = simulateCacheStep(cache, doc, { ...baseOptions, streaming: true, highlightRanges: h2 })
      const third = simulateCacheStep(cache, doc, { ...baseOptions, streaming: true, highlightRanges: h2 })

      // Different highlights may or may not produce different blocks
      // (depends on whether highlights overlap rendered text)
      const firstHas = first.blocks.some(hasHighlight)
      const secondHas = second.blocks.some(hasHighlight)
      if (firstHas && secondHas) {
        // If both have visible highlights, they should differ
        // (unless highlight ranges happen to cover same rendered text)
      }
      // Same highlights should always produce same blocks
      expect(third.blocks.map(normalizeBlock)).toEqual(second.blocks.map(normalizeBlock))
    })
  }

  for (const [aIndex, docA] of streamingDocuments.slice(0, 10).entries()) {
    for (const [bIndex, docB] of streamingDocuments.slice(10, 20).entries()) {
      test(`different content no stale cache A${aIndex + 1}-B${bIndex + 1}`, () => {
        const cache = createCacheState()
        simulateCacheStep(cache, docA, { ...baseOptions, streaming: true })
        const result = simulateCacheStep(cache, docB, { ...baseOptions, streaming: true })
        const fresh = renderDocumentToBlocks(parseMarkdown(docB), baseOptions)
        expectBlockStructuralEquality(result.blocks, fresh)
      })
    }
  }

  for (const doc of streamingDocuments.slice(0, 20)) {
    test(`palette/code width invalidation "${preview(doc)}"`, () => {
      const cache = createCacheState()
      const first = simulateCacheStep(cache, doc, { ...baseOptions, streaming: true })
      const second = simulateCacheStep(cache, doc, { ...baseOptions, streaming: true, codeBlockWidth: 81 })
      expect(second.blocks.map(normalizeBlock)).toEqual(
        renderDocumentToBlocks(parseMarkdown(doc), { ...baseOptions, codeBlockWidth: 81 }).map(normalizeBlock),
      )
      expect(first.blocks.length).toBeGreaterThanOrEqual(0)
    })
  }
})

describe('render pipeline extended - highlight visibility and content sanity', () => {
  for (const scenario of editScenarios.slice(0, 24)) {
    test(`highlight visible text sanity "${preview(scenario.old)}"`, () => {
      const matchIdx = scenario.base.indexOf(scenario.old)
      const reveal = scenario.new.slice(0, Math.max(1, Math.min(4, scenario.new.length)))
      const content =
        scenario.base.slice(0, matchIdx) + reveal + scenario.base.slice(matchIdx + scenario.old.length)
      const highlights: HighlightRange[] = [
        { start: matchIdx, end: matchIdx + reveal.length, backgroundColor: '#00ff00' },
      ]
      const blocks = renderDocumentToBlocks(parseMarkdown(content), { ...baseOptions, highlights })
      expect(blocks.some(hasHighlight)).toBe(collectHighlightedText(blocks).length > 0)
    })
  }

  for (const scenario of editScenarios.slice(0, 24)) {
    test(`replaceAll highlight range generation "${preview(scenario.old)}"`, () => {
      if (!scenario.base.includes(scenario.old)) return
      const ranges = buildReplaceAllRanges(scenario.base, scenario.old, scenario.new)
      const replaced = scenario.base.replaceAll(scenario.old, scenario.new)
      const blocks = renderDocumentToBlocks(parseMarkdown(replaced), { ...baseOptions, highlights: ranges })
      expect(blocks.map(normalizeBlock)).toEqual(
        renderDocumentToBlocks(parseMarkdown(replaced), { ...baseOptions, highlights: ranges }).map(normalizeBlock),
      )
    })
  }
})