import { describe, expect, it, test } from 'bun:test'
import type { Root, RootContent } from 'mdast'
import { parseMarkdownToMdast } from './markdown-parser'
import {
  renderDocumentToBlocks,
  renderDocumentItemToBlocks,
  spansToText,
  type Block,
  type HighlightRange,
  type Span,
} from './render-blocks'
import { buildMarkdownColorPalette, chatThemes } from './theme'
import { hasOddFenceCount } from './markdown-content-renderer'

const palette = buildMarkdownColorPalette(chatThemes.dark)
const baseOptions = { palette, codeBlockWidth: 80 }

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
        lines: block.lines.map(line => line.map(normalizeSpan)),
      }
    case 'list':
      return {
        type: block.type,
        style: block.style,
        source: block.source,
        items: block.items.map(item => ({
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
        headers: block.headers.map(row => row.map(normalizeSpan)),
        rows: block.rows.map(row => row.map(cell => cell.map(normalizeSpan))),
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
  return blocks.map(b => b.type)
}

function paragraphText(block: Block): string {
  if (block.type !== 'paragraph') throw new Error('not a paragraph')
  return block.content.map(s => s.text).join('')
}

function hasHighlight(block: Block): boolean {
  if (block.type === 'paragraph' || block.type === 'heading') {
    return block.content.some(s => !!s.bg)
  }
  if (block.type === 'table') {
    return [...block.headers.flat(), ...block.rows.flat(2)].some(s => !!s.bg)
  }
  if (block.type === 'code') {
    return block.lines.flat().some(s => !!s.bg)
  }
  if (block.type === 'list') {
    return block.items.some(item => item.content.some(hasHighlight))
  }
  if (block.type === 'blockquote') {
    return block.content.some(hasHighlight)
  }
  return false
}

function paragraphCount(blocks: Block[]): number {
  return blocks.filter(b => b.type === 'paragraph').length
}

function final<T>(arr: T[]): T {
  const value = arr[arr.length - 1]
  if (!value) throw new Error('empty array')
  return value
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
        rows: block.rows.map(row => row.map(spansToText)),
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
        items: block.items.map(item => ({
          marker: item.marker,
          checked: item.checked,
          content: item.content.map(extractBlockTextShape),
        })),
      }
  }
}

function expectBlockStructuralEquality(incrementalBlocks: Block[], freshBlocks: Block[]) {
  expect(blockTypes(incrementalBlocks)).toEqual(blockTypes(freshBlocks))
  expect(incrementalBlocks).toHaveLength(freshBlocks.length)

  for (let i = 0; i < incrementalBlocks.length; i++) {
    const ib = incrementalBlocks[i]!
    const fb = freshBlocks[i]!
    expect(ib.type).toBe(fb.type)

    if (ib.type === 'paragraph' && fb.type === 'paragraph') {
      expect(spansToText(ib.content)).toEqual(spansToText(fb.content))
    } else if (ib.type === 'heading' && fb.type === 'heading') {
      expect(spansToText(ib.content)).toEqual(spansToText(fb.content))
      expect(ib.level).toEqual(fb.level)
      expect(ib.slug).toEqual(fb.slug)
    } else if (ib.type === 'spacer' && fb.type === 'spacer') {
      expect(ib.lines).toEqual(fb.lines)
    } else if (ib.type === 'table' && fb.type === 'table') {
      expect(ib.headers.map(spansToText)).toEqual(fb.headers.map(spansToText))
      expect(ib.rows).toHaveLength(fb.rows.length)
      expect(ib.rows.map(row => row.map(spansToText))).toEqual(fb.rows.map(row => row.map(spansToText)))
      expect(ib.columnWidths).toEqual(fb.columnWidths)
    } else {
      expect(extractBlockTextShape(ib)).toEqual(extractBlockTextShape(fb))
    }
  }

  expect(incrementalBlocks.map(normalizeBlock)).toEqual(freshBlocks.map(normalizeBlock))
}

function expectIncrementalBlocksToMatchFresh(
  source: string,
  _previous: Root | undefined,
  options: typeof baseOptions | (typeof baseOptions & { highlights?: HighlightRange[] }),
) {
  const incremental = parseMarkdownToMdast(source)
  const fresh = parseMarkdownToMdast(source)
  const incrementalBlocks = renderDocumentToBlocks(incremental, options)
  const freshBlocks = renderDocumentToBlocks(fresh, options)
  expectBlockStructuralEquality(incrementalBlocks, freshBlocks)
  return incremental
}

function simulateStreaming(
  fullText: string,
  chunkSizes: number[],
  options: typeof baseOptions,
): Array<{ prefix: string; blocks: Block[]; doc: Root }> {
  const results: Array<{ prefix: string; blocks: Block[]; doc: Root }> = []
  let previous: Root | undefined
  let offset = 0

  for (const size of chunkSizes) {
    offset = Math.min(offset + size, fullText.length)
    const prefix = fullText.slice(0, offset)
    const doc = expectIncrementalBlocksToMatchFresh(prefix, previous, options)
    const blocks = renderDocumentToBlocks(doc, options)
    results.push({ prefix, blocks, doc })
    previous = doc
    if (offset >= fullText.length) break
  }

  if (results.length === 0 || results[results.length - 1]?.prefix !== fullText) {
    const doc = expectIncrementalBlocksToMatchFresh(fullText, previous, options)
    results.push({ prefix: fullText, blocks: renderDocumentToBlocks(doc, options), doc })
  }

  return results
}

function simulateArtifactUpdate(
  baseContent: string,
  oldString: string,
  newString: string,
  chunkSizes: number[],
  options: typeof baseOptions,
): Array<{ content: string; blocks: Block[]; highlights: HighlightRange[]; doc: Root }> {
  const matchIndex = baseContent.indexOf(oldString)
  if (matchIndex === -1) throw new Error('oldString not found in baseContent')

  const results: Array<{ content: string; blocks: Block[]; highlights: HighlightRange[]; doc: Root }> = []
  let previous: Root | undefined
  let revealedLength = 0

  for (const size of chunkSizes) {
    revealedLength = Math.min(revealedLength + size, newString.length)
    const revealed = newString.slice(0, revealedLength)
    const content =
      baseContent.slice(0, matchIndex) +
      revealed +
      baseContent.slice(matchIndex + oldString.length)
    const highlights: HighlightRange[] = [{
      start: matchIndex,
      end: matchIndex + revealed.length,
      backgroundColor: '#00ff00',
    }]
    const doc = expectIncrementalBlocksToMatchFresh(content, previous, { ...options, highlights })
    const blocks = renderDocumentToBlocks(doc, { ...options, highlights })
    results.push({ content, blocks, highlights, doc })
    previous = doc
    if (revealedLength >= newString.length) break
  }

  if (results.length === 0 || results[results.length - 1]?.content !== baseContent.replace(oldString, newString)) {
    const finalContent = baseContent.replace(oldString, newString)
    const highlights: HighlightRange[] = [{
      start: matchIndex,
      end: matchIndex + newString.length,
      backgroundColor: '#00ff00',
    }]
    const doc = expectIncrementalBlocksToMatchFresh(finalContent, previous, { ...options, highlights })
    results.push({
      content: finalContent,
      blocks: renderDocumentToBlocks(doc, { ...options, highlights }),
      highlights,
      doc,
    })
  }

  return results
}

function simulateStreamingMarkdownCache(
  contentSteps: Array<{ content: string; highlightRanges?: HighlightRange[] }>,
  options: typeof baseOptions & { streaming?: boolean } = { ...baseOptions, streaming: true },
) {
  let prevDoc: Root | null = null
  let itemBlockCache = new WeakMap<RootContent, Block[]>()
  let prevHighlightRanges: HighlightRange[] | undefined
  let prevPalette = options.palette
  let prevCodeBlockWidth = options.codeBlockWidth
  const rendersPerStep: number[] = []

  for (const step of contentSteps) {
    let completeSection = step.content
    if (hasOddFenceCount(step.content)) {
      const lastFenceIndex = step.content.lastIndexOf('```')
      if (lastFenceIndex !== -1) {
        completeSection = step.content.slice(0, lastFenceIndex)
      }
    }

    if (!completeSection || completeSection.trim() === '') {
      prevDoc = null
      itemBlockCache = new WeakMap()
      rendersPerStep.push(0)
      continue
    }

    const highlightChanged = step.highlightRanges !== prevHighlightRanges
    const paletteChanged = options.palette !== prevPalette
    const codeBlockWidthChanged = options.codeBlockWidth !== prevCodeBlockWidth
    if (highlightChanged || paletteChanged || codeBlockWidthChanged) {
      itemBlockCache = new WeakMap()
    }

    const doc = parseMarkdownToMdast(completeSection, { previous: prevDoc ?? undefined })
    let renderedCount = 0

    for (const item of doc.children) {
      let itemBlocks = itemBlockCache.get(item)
      if (!itemBlocks) {
        renderedCount++
        itemBlocks = renderDocumentItemToBlocks(item, {
          palette: options.palette,
          codeBlockWidth: options.codeBlockWidth,
          highlights: step.highlightRanges,
        })
        itemBlockCache.set(item, itemBlocks)
      }
    }

    rendersPerStep.push(renderedCount)
    prevDoc = doc
    prevHighlightRanges = step.highlightRanges
    prevPalette = options.palette
    prevCodeBlockWidth = options.codeBlockWidth
  }

  return rendersPerStep
}

function chunkPlanForDocument(doc: string, chunkSize: number): number[] {
  if (doc.length === 0) return [1]
  return Array(Math.ceil(doc.length / chunkSize)).fill(chunkSize)
}

function preview(value: string, max = 24): string {
  return value.length <= max ? value : `${value.slice(0, max)}...`
}

function collectHighlightedText(blocks: Block[], color = '#00ff00'): string {
  const highlightedText: string[] = []

  const collect = (block: Block) => {
    if (block.type === 'paragraph' || block.type === 'heading') {
      highlightedText.push(...block.content.filter(s => s.bg === color).map(s => s.text))
    } else if (block.type === 'table') {
      highlightedText.push(
        ...block.headers.flat().filter(s => s.bg === color).map(s => s.text),
        ...block.rows.flat(2).filter(s => s.bg === color).map(s => s.text),
      )
    } else if (block.type === 'code') {
      highlightedText.push(...block.lines.flat().filter(s => s.bg === color).map(s => s.text))
    } else if (block.type === 'list') {
      block.items.forEach(item => item.content.forEach(collect))
    } else if (block.type === 'blockquote') {
      block.content.forEach(collect)
    }
  }

  blocks.forEach(collect)
  return highlightedText.join('')
}

function editTargetsCodeBlock(base: string, oldString: string): boolean {
  const fencePattern = /^```/gm
  const fenceOffsets: number[] = []

  for (const match of base.matchAll(fencePattern)) {
    if (typeof match.index === 'number') fenceOffsets.push(match.index)
  }

  const matchIdx = base.indexOf(oldString)
  if (matchIdx === -1) return false

  return fenceOffsets.filter(index => index < matchIdx).length % 2 === 1
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

const sequentialEditScenarios = [
  {
    initial: '# Title\n\nFirst paragraph\n\nSecond paragraph\n\nThird paragraph',
    edits: [
      { old: 'First paragraph', new: 'Changed first' },
      { old: 'Second paragraph', new: 'Changed second' },
      { old: 'Third paragraph', new: 'Changed third' },
    ],
  },
  {
    initial: '| a | b |\n| - | - |\n| c | d |\n| e | f |',
    edits: [
      { old: 'c', new: 'x' },
      { old: 'd', new: 'y' },
      { old: 'e', new: 'z' },
    ],
  },
  {
    initial: '- item 1\n- item 2\n- item 3',
    edits: [
      { old: 'item 1', new: 'changed 1' },
      { old: 'item 2', new: 'changed 2' },
    ],
  },
  {
    initial: 'A\n\nB\n\nC\n\nD',
    edits: [
      { old: 'B', new: '# Heading B' },
      { old: 'C', new: '- list\n- item' },
      { old: 'D', new: '```js\nfinal()\n```' },
    ],
  },
  {
    initial: '> quote one\n\n> quote two\n\nplain',
    edits: [
      { old: 'quote one', new: 'replaced one' },
      { old: 'quote two', new: 'replaced two' },
      { old: 'plain', new: 'tail paragraph' },
    ],
  },
  {
    initial: '# A\n\nText\n\n# B\n\nMore text',
    edits: [
      { old: '# A', new: '# Alpha' },
      { old: 'Text', new: 'Body Alpha' },
      { old: '# B', new: '## Beta' },
      { old: 'More text', new: 'Body Beta' },
    ],
  },
  {
    initial: 'Before\n\n```ts\none()\n```\n\nAfter',
    edits: [
      { old: 'one()', new: 'two()' },
      { old: 'After', new: 'Later' },
      { old: 'Before', new: 'Start' },
    ],
  },
  {
    initial: '| h1 | h2 |\n| -- | -- |\n| a | b |\n\nParagraph',
    edits: [
      { old: 'a', new: 'alpha' },
      { old: 'b', new: 'beta' },
      { old: 'Paragraph', new: 'Footer paragraph' },
    ],
  },
  {
    initial: 'foo\n\nbar\n\nbaz',
    edits: [
      { old: 'foo', new: 'one' },
      { old: 'bar', new: 'two' },
      { old: 'baz', new: 'three' },
    ],
  },
  {
    initial: '- [ ] task 1\n- [x] task 2\n- [ ] task 3',
    edits: [
      { old: 'task 1', new: 'alpha' },
      { old: 'task 2', new: 'beta' },
      { old: 'task 3', new: 'gamma' },
    ],
  },
]

describe('render pipeline - streaming append', () => {
  it('plain paragraph streaming', () => {
    const full = 'Hello world, this is a complete sentence.'
    const results = simulateStreaming(full, Array(20).fill(5), baseOptions)

    for (const step of results) {
      expect(blockTypes(step.blocks)).toEqual(['paragraph'])
    }

    expect(paragraphText(final(results).blocks[0]!)).toBe(full)
  })

  it('multiple paragraphs streaming', () => {
    const full = 'First paragraph.\n\nSecond paragraph.\n\nThird paragraph.'
    const results = simulateStreaming(full, Array(20).fill(8), baseOptions)

    for (const step of results) {
      const expectedMax = step.prefix.split('\n\n').filter(Boolean).length
      expect(paragraphCount(step.blocks)).toBeLessThanOrEqual(expectedMax)
    }

    expect(blockTypes(final(results).blocks)).toEqual([
      'paragraph',
      'spacer',
      'paragraph',
      'spacer',
      'paragraph',
    ])
  })

  it('heading then paragraph streaming', () => {
    const full = '# My Heading\n\nSome body text here.'
    const results = simulateStreaming(full, Array(20).fill(5), baseOptions)

    expect(results.some(step => step.blocks.some(block => block.type === 'heading'))).toBe(true)

    const afterParagraphStarts = results.find(step => step.prefix.includes('\n\nS'))
    expect(afterParagraphStarts?.blocks.some(block => block.type === 'paragraph')).toBe(true)

    expect(blockTypes(final(results).blocks)).toEqual(['heading', 'spacer', 'paragraph'])
  })

  it('table streaming', () => {
    const full = '| Name | Age |\n| --- | --- |\n| Alice | 30 |\n| Bob | 25 |\n'
    const results = simulateStreaming(full, Array(20).fill(10), baseOptions)

    expect(blockTypes(results[0]!.blocks)).toEqual(['paragraph'])
    expect(results.some(step => step.blocks.some(block => block.type === 'table'))).toBe(true)

    const lastBlock = final(results).blocks[0]
    expect(lastBlock?.type).toBe('table')
    if (lastBlock?.type === 'table') {
      expect(lastBlock.rows).toHaveLength(2)
      expect(lastBlock.headers.map(spansToText)).toEqual(['Name', 'Age'])
    }
  })

  it('code block streaming', () => {
    const full = 'Some text.\n\n```javascript\nconsole.log("hello")\n```\n\nMore text.'
    const results = simulateStreaming(full, Array(20).fill(10), baseOptions)

    const incompleteStep = results.find(step => step.prefix.includes('```javascript') && !step.prefix.includes('```javascript\nconsole.log("hello")\n```'))
    expect(incompleteStep).toBeDefined()

    expect(blockTypes(final(results).blocks)).toEqual([
      'paragraph',
      'spacer',
      'code',
      'spacer',
      'paragraph',
    ])
  })

  it('list streaming', () => {
    const full = 'Shopping list:\n\n- Apples\n- Bananas\n- Oranges'
    const results = simulateStreaming(full, Array(20).fill(8), baseOptions)
    expect(blockTypes(final(results).blocks)).toEqual(['paragraph', 'spacer', 'list'])

    const list = final(results).blocks[2]
    expect(list?.type).toBe('list')
    if (list?.type === 'list') {
      expect(list.items).toHaveLength(3)
    }
  })

  it('mixed content streaming', () => {
    const full = [
      '# Mixed',
      '',
      'Body paragraph.',
      '',
      '```ts',
      'console.log("x")',
      '```',
      '',
      '| Name | Age |',
      '| --- | --- |',
      '| Alice | 30 |',
      '',
      '- One',
      '- Two',
    ].join('\n')

    const results = simulateStreaming(full, Array(40).fill(7), baseOptions)
    expect(blockTypes(final(results).blocks)).toEqual([
      'heading',
      'spacer',
      'paragraph',
      'spacer',
      'code',
      'spacer',
      'table',
      'spacer',
      'list',
    ])
  })

  it('single character streaming always stays one paragraph', () => {
    const results = simulateStreaming('Hello', [1, 1, 1, 1, 1], baseOptions)
    for (const step of results) {
      expect(blockTypes(step.blocks)).toEqual(['paragraph'])
    }
  })

  it('rapid block type changes', () => {
    const prefixes = ['>', '> quote', '> quote\n\nnot quote']
    let previous: Root | undefined

    for (const prefix of prefixes) {
      previous = expectIncrementalBlocksToMatchFresh(prefix, previous, baseOptions)
    }

    const blocks = renderDocumentToBlocks(previous!, baseOptions)
    expect(blockTypes(blocks)).toEqual(['blockquote', 'spacer', 'paragraph'])
  })
})

describe('render pipeline - artifact update', () => {
  it('replace word in paragraph', () => {
    const results = simulateArtifactUpdate(
      'Hello world\n\nThis is old text\n\nGoodbye',
      'old text',
      'new text',
      Array(20).fill(1),
      baseOptions,
    )

    for (const step of results) {
      expect(blockTypes(step.blocks)).toEqual(['paragraph', 'spacer', 'paragraph', 'spacer', 'paragraph'])
      expect(hasHighlight(step.blocks[2]!)).toBe(true)
    }

    expect(paragraphText(final(results).blocks[2]!)).toBe('This is new text')
  })

  it('replace text in table cell', () => {
    const results = simulateArtifactUpdate(
      '| Name | Status |\n| --- | --- |\n| Alice | Active |\n| Bob | Inactive |\n',
      'Active',
      'Retired',
      Array(20).fill(1),
      baseOptions,
    )

    for (const step of results) {
      expect(blockTypes(step.blocks)).toEqual(['table'])
      expect(hasHighlight(step.blocks[0]!)).toBe(true)
    }

    const table = final(results).blocks[0]
    expect(table?.type).toBe('table')
    if (table?.type === 'table') {
      expect(table.rows[0]?.map(spansToText)).toEqual(['Alice', 'Retired'])
    }
  })

  it('replace paragraph with heading', () => {
    const results = simulateArtifactUpdate(
      'Hello\n\nold paragraph\n\nFooter',
      'old paragraph',
      '# New Heading',
      Array(20).fill(1),
      baseOptions,
    )

    expect(results.some(step => step.blocks[2]?.type === 'heading')).toBe(true)
    expect(blockTypes(final(results).blocks)).toEqual(['paragraph', 'spacer', 'heading', 'spacer', 'paragraph'])
  })

  it('replace content spanning block boundary', () => {
    const results = simulateArtifactUpdate(
      'Para one\n\nPara two\n\nPara three',
      'Para two\n\nPara three',
      'Single replacement paragraph',
      [3, 4, 5, 100],
      baseOptions,
    )

    expect(blockTypes(final(results).blocks)).toEqual(['paragraph', 'spacer', 'paragraph'])
    expect(paragraphText(final(results).blocks[2]!)).toBe('Single replacement paragraph')
  })

  it('insert new content', () => {
    const results = simulateArtifactUpdate(
      'Before\n\nAfter',
      'After',
      'Middle\n\nAfter',
      [2, 2, 2, 100],
      baseOptions,
    )

    expect(blockTypes(final(results).blocks)).toEqual(['paragraph', 'spacer', 'paragraph', 'spacer', 'paragraph'])
  })

  it('delete content', () => {
    const results = simulateArtifactUpdate(
      'Keep this\n\nDelete this\n\nKeep this too',
      '\n\nDelete this',
      '',
      [1],
      baseOptions,
    )

    expect(blockTypes(final(results).blocks)).toEqual(['paragraph', 'spacer', 'paragraph'])
  })

  it('replaceAll simulation', () => {
    const base = 'foo bar foo baz foo'
    const replaced = base.replaceAll('foo', 'qux')
    const ranges: HighlightRange[] = []
    let searchFrom = 0
    while (true) {
      const index = replaced.indexOf('qux', searchFrom)
      if (index === -1) break
      ranges.push({ start: index, end: index + 3, backgroundColor: '#00ff00' })
      searchFrom = index + 3
    }

    const blocks = renderDocumentToBlocks(parseMarkdownToMdast(replaced), { ...baseOptions, highlights: ranges })
    expect(blockTypes(blocks)).toEqual(['paragraph'])
    expect(paragraphText(blocks[0]!)).toBe(replaced)
    const highlightedSpans = blocks[0]!.type === 'paragraph'
      ? blocks[0]!.content.filter(span => span.bg === '#00ff00')
      : []
    expect(highlightedSpans).toHaveLength(3)
  })
})

describe('render pipeline - post-streaming consistency', () => {
  it('final incremental parse matches fresh parse', () => {
    const full = '# Done\n\nBody\n\n- one\n- two'
    const results = simulateStreaming(full, [2, 3, 4, 5, 100], baseOptions)
    const finalDoc = final(results).doc
    const fresh = parseMarkdownToMdast(full)
    expect(renderDocumentToBlocks(finalDoc, baseOptions).map(normalizeBlock))
      .toEqual(renderDocumentToBlocks(fresh, baseOptions).map(normalizeBlock))
  })

  it('block output stability', () => {
    const content = '# Stable\n\nParagraph'
    const first = renderDocumentToBlocks(parseMarkdownToMdast(content), baseOptions)
    const second = renderDocumentToBlocks(parseMarkdownToMdast(content), baseOptions)
    expect(first.map(normalizeBlock)).toEqual(second.map(normalizeBlock))
  })

  it('cache simulation reuses stable items and clears on highlight change', () => {
    const content1 = 'Alpha\n\nBravo'
    const content2 = 'Alpha\n\nBravo extended'
    const h1: HighlightRange[] = [{ start: 0, end: 5, backgroundColor: '#0f0' }]
    const h2: HighlightRange[] = [{ start: 0, end: 6, backgroundColor: '#0f0' }]

    const renders = simulateStreamingMarkdownCache([
      { content: content1, highlightRanges: h1 },
      { content: content2, highlightRanges: h1 },
      { content: content2, highlightRanges: h2 },
    ])

    expect(renders[0]).toBe(2)
    expect(renders[1]).toBe(2)
    expect(renders[2]).toBe(2)
  })
})

describe('render pipeline - edge cases', () => {
  it('empty content', () => {
    expect(renderDocumentToBlocks(parseMarkdownToMdast(''), baseOptions)).toEqual([])
  })

  it('whitespace only', () => {
    const blocks = renderDocumentToBlocks(parseMarkdownToMdast('   \n\n   '), baseOptions)
    expect(Array.isArray(blocks)).toBe(true)
  })

  it('table with styled content and highlights', () => {
    const source = '| Name | Notes |\n| --- | --- |\n| **Alice** | *Active* |\n'
    const start = source.indexOf('Alice')
    const blocks = renderDocumentToBlocks(parseMarkdownToMdast(source), {
      ...baseOptions,
      highlights: [{ start, end: start + 5, backgroundColor: '#00ff00' }],
    })

    expect(blockTypes(blocks)).toEqual(['table'])
    const table = blocks[0]
    expect(table?.type).toBe('table')
    if (table?.type === 'table') {
      const aliceCell = table.rows[0]?.[0] ?? []
      const activeCell = table.rows[0]?.[1] ?? []
      expect(aliceCell.some(span => span.bold)).toBe(true)
      expect(activeCell.some(span => span.italic)).toBe(true)
      expect(aliceCell.some(span => span.bg === '#00ff00')).toBe(true)
    }
  })
})

describe('render pipeline - exhaustive streaming append invariant', () => {
  const chunkSizes = [1, 3, 5, 10, 20]

  for (const [docIndex, doc] of streamingDocuments.entries()) {
    for (const chunkSize of chunkSizes) {
      if (chunkSize === 1 && doc.length > 120) continue

      test(`streaming doc#${docIndex + 1} "${preview(doc)}" chunk=${chunkSize} matches fresh parse at every step`, () => {
        let previous: Root | undefined

        for (let i = chunkSize; i <= doc.length; i += chunkSize) {
          const prefix = doc.slice(0, Math.min(i, doc.length))
          const incremental = parseMarkdownToMdast(prefix)
          const fresh = parseMarkdownToMdast(prefix)

          const incBlocks = renderDocumentToBlocks(incremental, baseOptions)
          const freshBlocks = renderDocumentToBlocks(fresh, baseOptions)

          expectBlockStructuralEquality(incBlocks, freshBlocks)
          previous = incremental
        }

        const finalIncremental = parseMarkdownToMdast(doc)
        const finalFresh = parseMarkdownToMdast(doc)
        expectBlockStructuralEquality(
          renderDocumentToBlocks(finalIncremental, baseOptions),
          renderDocumentToBlocks(finalFresh, baseOptions),
        )
      })
    }
  }
})

describe('render pipeline - exhaustive mid-document replacement invariant', () => {
  for (const [scenarioIndex, scenario] of editScenarios.entries()) {
    test(`replacement #${scenarioIndex + 1}: "${preview(scenario.old)}" -> "${preview(scenario.new)}" inside "${preview(scenario.base)}"`, () => {
      const matchIdx = scenario.base.indexOf(scenario.old)
      expect(matchIdx).not.toBe(-1)

      let previous: Root | undefined

      for (let revealLen = 0; revealLen <= scenario.new.length; revealLen++) {
        const revealed = scenario.new.slice(0, revealLen)
        const composed =
          scenario.base.slice(0, matchIdx) +
          revealed +
          scenario.base.slice(matchIdx + scenario.old.length)

        const incremental = parseMarkdownToMdast(composed)
        const fresh = parseMarkdownToMdast(composed)

        const incBlocks = renderDocumentToBlocks(incremental, baseOptions)
        const freshBlocks = renderDocumentToBlocks(fresh, baseOptions)

        expectBlockStructuralEquality(incBlocks, freshBlocks)
        previous = incremental
      }
    })
  }
})

describe('render pipeline - exhaustive highlight correctness', () => {
  for (const [scenarioIndex, scenario] of editScenarios.slice(0, 15).entries()) {
    test(`highlight correctness #${scenarioIndex + 1}: "${preview(scenario.old)}" -> "${preview(scenario.new)}"`, () => {
      const matchIdx = scenario.base.indexOf(scenario.old)
      expect(matchIdx).not.toBe(-1)

      for (let revealLen = 1; revealLen <= scenario.new.length; revealLen++) {
        const revealed = scenario.new.slice(0, revealLen)
        const composed =
          scenario.base.slice(0, matchIdx) +
          revealed +
          scenario.base.slice(matchIdx + scenario.old.length)
        const highlights: HighlightRange[] = [{
          start: matchIdx,
          end: matchIdx + revealed.length,
          backgroundColor: '#00ff00',
        }]

        const doc = parseMarkdownToMdast(composed)
        const blocks = renderDocumentToBlocks(doc, { ...baseOptions, highlights })

        let foundHighlight = false
        for (const block of blocks) {
          if (hasHighlight(block)) {
            foundHighlight = true
            break
          }
        }

        const visibleHighlightedText = collectHighlightedText(blocks)
        expect(foundHighlight).toBe(visibleHighlightedText.length > 0)
      }
    })
  }

  for (const [scenarioIndex, scenario] of editScenarios.slice(0, 15).entries()) {
    test(`highlight exact text preservation #${scenarioIndex + 1}: "${preview(scenario.base)}"`, () => {
      const matchIdx = scenario.base.indexOf(scenario.old)
      expect(matchIdx).not.toBe(-1)

      const revealLen = Math.max(1, Math.min(3, scenario.new.length))
      const revealed = scenario.new.slice(0, revealLen)
      const composed =
        scenario.base.slice(0, matchIdx) +
        revealed +
        scenario.base.slice(matchIdx + scenario.old.length)
      const highlights: HighlightRange[] = [{
        start: matchIdx,
        end: matchIdx + revealed.length,
        backgroundColor: '#00ff00',
      }]

      const blocks = renderDocumentToBlocks(parseMarkdownToMdast(composed), { ...baseOptions, highlights })
      const highlightedText = collectHighlightedText(blocks)

      if (!editTargetsCodeBlock(scenario.base, scenario.old) && !scenario.new.startsWith('```')) {
        expect(revealed).toContain(highlightedText)
      }
    })
  }

  for (const [scenarioIndex, scenario] of editScenarios.slice(0, 15).entries()) {
    test(`highlighted render still matches fresh structure #${scenarioIndex + 1}`, () => {
      const matchIdx = scenario.base.indexOf(scenario.old)
      expect(matchIdx).not.toBe(-1)

      let previous: Root | undefined

      for (let revealLen = 1; revealLen <= scenario.new.length; revealLen++) {
        const revealed = scenario.new.slice(0, revealLen)
        const composed =
          scenario.base.slice(0, matchIdx) +
          revealed +
          scenario.base.slice(matchIdx + scenario.old.length)
        const highlights: HighlightRange[] = [{
          start: matchIdx,
          end: matchIdx + revealed.length,
          backgroundColor: '#00ff00',
        }]

        const incremental = parseMarkdownToMdast(composed)
        const fresh = parseMarkdownToMdast(composed)

        expectBlockStructuralEquality(
          renderDocumentToBlocks(incremental, { ...baseOptions, highlights }),
          renderDocumentToBlocks(fresh, { ...baseOptions, highlights }),
        )
        previous = incremental
      }
    })
  }

  for (const [scenarioIndex, scenario] of editScenarios.slice(0, 5).entries()) {
    for (const revealLen of [1, 2, 3, 4]) {
      test(`highlight range split behavior #${scenarioIndex + 1} reveal=${revealLen}`, () => {
        const matchIdx = scenario.base.indexOf(scenario.old)
        const boundedReveal = Math.min(revealLen, scenario.new.length)
        const revealed = scenario.new.slice(0, boundedReveal)
        const composed =
          scenario.base.slice(0, matchIdx) +
          revealed +
          scenario.base.slice(matchIdx + scenario.old.length)
        const highlights: HighlightRange[] = [{
          start: matchIdx,
          end: matchIdx + boundedReveal,
          backgroundColor: '#00ff00',
        }]
        const blocks = renderDocumentToBlocks(parseMarkdownToMdast(composed), { ...baseOptions, highlights })
        expect(blocks.some(hasHighlight)).toBe(collectHighlightedText(blocks).length > 0)
      })
    }
  }
})

describe('render pipeline - sequential edits', () => {
  for (const [scenarioIndex, scenario] of sequentialEditScenarios.entries()) {
    test(`sequential edits scenario #${scenarioIndex + 1}: "${preview(scenario.initial)}"`, () => {
      let currentContent = scenario.initial
      let previous: Root | undefined

      for (const edit of scenario.edits) {
        const matchIdx = currentContent.indexOf(edit.old)
        expect(matchIdx).not.toBe(-1)

        currentContent =
          currentContent.slice(0, matchIdx) +
          edit.new +
          currentContent.slice(matchIdx + edit.old.length)

        const incremental = parseMarkdownToMdast(currentContent)
        const fresh = parseMarkdownToMdast(currentContent)

        expectBlockStructuralEquality(
          renderDocumentToBlocks(incremental, baseOptions),
          renderDocumentToBlocks(fresh, baseOptions),
        )

        previous = incremental
      }
    })
  }

  for (const [scenarioIndex, scenario] of sequentialEditScenarios.entries()) {
    test(`sequential edits with highlights scenario #${scenarioIndex + 1}`, () => {
      let currentContent = scenario.initial
      let previous: Root | undefined

      for (const edit of scenario.edits) {
        const matchIdx = currentContent.indexOf(edit.old)
        expect(matchIdx).not.toBe(-1)

        currentContent =
          currentContent.slice(0, matchIdx) +
          edit.new +
          currentContent.slice(matchIdx + edit.old.length)

        const highlightIndex = currentContent.indexOf(edit.new)
        const highlights: HighlightRange[] = [{
          start: highlightIndex,
          end: highlightIndex + edit.new.length,
          backgroundColor: '#00ff00',
        }]

        const incremental = parseMarkdownToMdast(currentContent)
        const fresh = parseMarkdownToMdast(currentContent)

        expectBlockStructuralEquality(
          renderDocumentToBlocks(incremental, { ...baseOptions, highlights }),
          renderDocumentToBlocks(fresh, { ...baseOptions, highlights }),
        )

        const freshBlocksWithHighlights = renderDocumentToBlocks(fresh, { ...baseOptions, highlights })
        const expectVisibleHighlight = collectHighlightedText(freshBlocksWithHighlights).length > 0
        expect(freshBlocksWithHighlights.some(hasHighlight)).toBe(expectVisibleHighlight)

        previous = incremental
      }
    })
  }

  for (const [scenarioIndex, scenario] of sequentialEditScenarios.entries()) {
    test(`sequential edits final output stability #${scenarioIndex + 1}`, () => {
      let currentContent = scenario.initial
      for (const edit of scenario.edits) {
        const matchIdx = currentContent.indexOf(edit.old)
        expect(matchIdx).not.toBe(-1)
        currentContent =
          currentContent.slice(0, matchIdx) +
          edit.new +
          currentContent.slice(matchIdx + edit.old.length)
      }

      const first = renderDocumentToBlocks(parseMarkdownToMdast(currentContent), baseOptions)
      const second = renderDocumentToBlocks(parseMarkdownToMdast(currentContent), baseOptions)
      expectBlockStructuralEquality(first, second)
    })
  }
})

describe('render pipeline - highlight lifecycle', () => {
  it('highlight lifecycle: appear during streaming, persist during executing, clear on success', () => {
    const base = 'Hello\n\nOld text\n\nFooter'
    const replacement = 'New text'
    const matchIdx = base.indexOf('Old text')

    const beforeDoc = parseMarkdownToMdast(base)
    const beforeBlocks = renderDocumentToBlocks(beforeDoc, baseOptions)
    for (const block of beforeBlocks) {
      if (block.type === 'paragraph') {
        expect(block.content.every(s => !s.bg)).toBe(true)
      }
    }

    const composed = base.slice(0, matchIdx) + replacement + base.slice(matchIdx + 'Old text'.length)
    const highlights: HighlightRange[] = [{ start: matchIdx, end: matchIdx + replacement.length, backgroundColor: '#00ff00' }]
    const streamDoc = parseMarkdownToMdast(composed)
    const streamBlocks = renderDocumentToBlocks(streamDoc, { ...baseOptions, highlights })
    expect(streamBlocks.some(b => hasHighlight(b))).toBe(true)

    const successBlocks = renderDocumentToBlocks(streamDoc, baseOptions)
    for (const block of successBlocks) {
      if (block.type === 'paragraph') {
        expect(block.content.every(s => !s.bg)).toBe(true)
      }
    }
  })

  const lifecycleScenarios = editScenarios.slice(0, 24)
  for (const [scenarioIndex, scenario] of lifecycleScenarios.entries()) {
    test(`highlight lifecycle scenario #${scenarioIndex + 1}`, () => {
      const startDoc = parseMarkdownToMdast(scenario.base)
      const startBlocks = renderDocumentToBlocks(startDoc, baseOptions)
      expect(startBlocks.some(hasHighlight)).toBe(false)

      const matchIdx = scenario.base.indexOf(scenario.old)
      expect(matchIdx).not.toBe(-1)

      const midReveal = Math.max(1, Math.floor(scenario.new.length / 2))
      const revealed = scenario.new.slice(0, midReveal)
      const midContent =
        scenario.base.slice(0, matchIdx) +
        revealed +
        scenario.base.slice(matchIdx + scenario.old.length)
      const midHighlights: HighlightRange[] = [{
        start: matchIdx,
        end: matchIdx + revealed.length,
        backgroundColor: '#00ff00',
      }]

      const midBlocks = renderDocumentToBlocks(parseMarkdownToMdast(midContent), { ...baseOptions, highlights: midHighlights })
      if (!editTargetsCodeBlock(scenario.base, scenario.old)) {
        expect(midBlocks.some(hasHighlight)).toBe(collectHighlightedText(midBlocks).length > 0)
      }

      const finalContent = scenario.base.slice(0, matchIdx) + scenario.new + scenario.base.slice(matchIdx + scenario.old.length)
      const finalBlocksWithHighlights = renderDocumentToBlocks(parseMarkdownToMdast(finalContent), {
        ...baseOptions,
        highlights: [{ start: matchIdx, end: matchIdx + scenario.new.length, backgroundColor: '#00ff00' }],
      })
      if (!editTargetsCodeBlock(scenario.base, scenario.old)) {
        expect(finalBlocksWithHighlights.some(hasHighlight)).toBe(
          collectHighlightedText(finalBlocksWithHighlights).length > 0,
        )
      }

      const successBlocks = renderDocumentToBlocks(parseMarkdownToMdast(finalContent), baseOptions)
      expect(successBlocks.some(hasHighlight)).toBe(false)
    })
  }
})

describe('render pipeline - additional edge cases and invariant matrix', () => {
  const edgeCaseDocs = [
    '',
    'a',
    ' ',
    '   \n',
    '\n\n',
    '#',
    '# Heading only',
    '```js\n```',
    '```\nplain\n```',
    '| a | b |\n| - | - |\n|  |  |',
    '> > > deeply nested',
    '- one\n1. two\n- three',
    'Paragraph\n---\nParagraph',
    'Ends without newline',
    'Ends with newline\n',
    'こんにちは世界',
    '😀 emoji\n\n🚀 rocket',
    '# 同じ\n\n## レベル',
    'A very long paragraph '.repeat(40),
    '> quote\n> with\n> lines\n>\n> end',
    '````\ncode with ``` inside\n````',
    '![alt](url)',
    '[link](https://example.com) and `code` and **bold**',
    '---\n---\n---',
    '# A\n# B\n# C',
    '| col1 | col2 |\n| --- | --- |\n| left | |\n| | right |',
    '- [ ] unchecked\n- [x] checked',
    '<div>html</div>',
    'Line 1\nLine 2\nLine 3\n',
    '> quote\n\n- list\n\n| a | b |\n| - | - |\n| c | d |',
  ]

  for (const [docIndex, doc] of edgeCaseDocs.entries()) {
    for (const chunkSize of [1, 2, 4, 8]) {
      if (chunkSize === 1 && doc.length > 80) continue
      test(`edge streaming doc#${docIndex + 1} chunk=${chunkSize}`, () => {
        let previous: Root | undefined
        const plan = chunkPlanForDocument(doc, chunkSize)

        for (const size of plan) {
          const prevLen = (((previous as any)?.data?.source as string | undefined)?.length) ?? 0
          const prefix = doc.slice(0, Math.min(prevLen + size, doc.length))
          const incremental = parseMarkdownToMdast(prefix)
          const fresh = parseMarkdownToMdast(prefix)

          expectBlockStructuralEquality(
            renderDocumentToBlocks(incremental, baseOptions),
            renderDocumentToBlocks(fresh, baseOptions),
          )
          previous = incremental
          if (prefix.length === doc.length) break
        }

        const finalInc = parseMarkdownToMdast(doc)
        const finalFresh = parseMarkdownToMdast(doc)
        expectBlockStructuralEquality(
          renderDocumentToBlocks(finalInc, baseOptions),
          renderDocumentToBlocks(finalFresh, baseOptions),
        )
      })
    }
  }

  const whitespaceEdits = [
    { base: '', old: '', new: 'A' },
    { base: 'A', old: 'A', new: '' },
    { base: '   ', old: ' ', new: 'x' },
    { base: '\n\n', old: '\n', new: 'a' },
    { base: 'A\n', old: '\n', new: '\n\n' },
  ]

  for (const [caseIndex, scenario] of whitespaceEdits.entries()) {
    test(`whitespace mutation case #${caseIndex + 1}`, () => {
      const matchIdx = scenario.base.indexOf(scenario.old)
      expect(matchIdx).not.toBe(-1)
      let previous: Root | undefined

      for (let revealLen = 0; revealLen <= scenario.new.length; revealLen++) {
        const composed =
          scenario.base.slice(0, matchIdx) +
          scenario.new.slice(0, revealLen) +
          scenario.base.slice(matchIdx + scenario.old.length)

        const incremental = parseMarkdownToMdast(composed)
        const fresh = parseMarkdownToMdast(composed)
        expectBlockStructuralEquality(
          renderDocumentToBlocks(incremental, baseOptions),
          renderDocumentToBlocks(fresh, baseOptions),
        )
        previous = incremental
      }
    })
  }

  for (const [docIndex, doc] of edgeCaseDocs.slice(0, 20).entries()) {
    test(`fresh parse rendering stability edge doc#${docIndex + 1}`, () => {
      const first = renderDocumentToBlocks(parseMarkdownToMdast(doc), baseOptions)
      const second = renderDocumentToBlocks(parseMarkdownToMdast(doc), baseOptions)
      expectBlockStructuralEquality(first, second)
    })
  }
})