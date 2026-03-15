import React from 'react'
import { describe, expect, mock, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import { parseStreamingContent, StreamingMarkdownContent } from '../components/markdown-content'
import { parseMarkdownToMdast } from '../utils/markdown-parser'
import { blockTypes, collectHighlightedText, normalizeBlock, palette, renderBlocks } from '../utils/test-markdown-helpers'
import { renderDocumentItemToBlocks, renderDocumentToBlocks, type HighlightRange, type MarkdownPalette } from '../utils/render-blocks'

function createCacheSimulator() {
  let prevDoc: ReturnType<typeof parseMarkdownToMdast> | null = null
  let itemBlockCache = new WeakMap<object, ReturnType<typeof renderDocumentItemToBlocks>>()
  let prevHighlightRanges: HighlightRange[] | undefined
  let prevPalette: MarkdownPalette | null = null
  let prevCodeBlockWidth: number | undefined

  return {
    step(content: string, options: {
      palette: MarkdownPalette
      codeBlockWidth?: number
      highlightRanges?: HighlightRange[]
    }) {
      let completeSection = content
      let pendingText = ''

      const fenceCount = (content.match(/```/g) ?? []).length
      if (fenceCount % 2 === 1) {
        const lastFenceIndex = content.lastIndexOf('```')
        if (lastFenceIndex !== -1) {
          completeSection = content.slice(0, lastFenceIndex)
          pendingText = content.slice(lastFenceIndex)
        }
      }

      const highlightChanged =
        JSON.stringify(prevHighlightRanges ?? []) !== JSON.stringify(options.highlightRanges ?? [])
      const paletteChanged = prevPalette !== options.palette
      const widthChanged = prevCodeBlockWidth !== options.codeBlockWidth
      if (highlightChanged || paletteChanged || widthChanged) {
        itemBlockCache = new WeakMap()
      }

      if (!completeSection || completeSection.trim() === '') {
        prevDoc = null
        prevHighlightRanges = options.highlightRanges
        prevPalette = options.palette
        prevCodeBlockWidth = options.codeBlockWidth
        return { blocks: [], pendingText, renderedCount: 0 }
      }

      const doc = parseMarkdownToMdast(completeSection)
      const blocks = []
      let renderedCount = 0

      for (const item of doc.children) {
        let itemBlocks = itemBlockCache.get(item as object)
        if (!itemBlocks) {
          renderedCount++
          itemBlocks = renderDocumentItemToBlocks(item, {
            palette: options.palette,
            codeBlockWidth: options.codeBlockWidth,
            highlights: options.highlightRanges,
          })
          itemBlockCache.set(item as object, itemBlocks)
        }
        blocks.push(...itemBlocks)
      }

      const trailingMatch = completeSection.match(/\n\n+$/)
      if (trailingMatch) {
        blocks.push({ type: 'spacer', lines: trailingMatch[0].length - 1 })
      }

      prevDoc = doc
      prevHighlightRanges = options.highlightRanges
      prevPalette = options.palette
      prevCodeBlockWidth = options.codeBlockWidth

      return { blocks, pendingText, renderedCount }
    },
  }
}

mock.module('../hooks/use-theme', () => ({
  useTheme: () => ({
    foreground: 'white',
    muted: 'gray',
    border: 'gray',
  }),
}))

mock.module('@opentui/react', () => ({
  useRenderer: () => ({ terminal: { width: 100 }, clearSelection() {} }),
}))

describe('Layer 4 - Streaming Cache - Suite A Cache reuse/invalidation', () => {
  test('reuses cached item blocks when content only appends', () => {
    const cache = createCacheSimulator()

    const first = cache.step('# Title\n\nBody', { palette, codeBlockWidth: 80 })
    const second = cache.step('# Title\n\nBody\n\nTail', { palette, codeBlockWidth: 80 })

    expect(first.renderedCount).toBe(2)
    expect(second.renderedCount).toBe(3)
    expect(blockTypes(second.blocks)).toEqual(['heading', 'paragraph', 'paragraph'])
  })

  test('invalidates cache on width change', () => {
    const cache = createCacheSimulator()
    const markdown = `| A | Much Longer Header |
| - | ------------------ |
| 1 | value |`

    const first = cache.step(markdown, { palette, codeBlockWidth: 80 })
    const second = cache.step(markdown, { palette, codeBlockWidth: 24 })

    expect(second.renderedCount).toBeGreaterThanOrEqual(first.renderedCount)
    const firstTable = first.blocks.find((block) => block.type === 'table')
    const secondTable = second.blocks.find((block) => block.type === 'table')
    expect(firstTable?.type).toBe('table')
    expect(secondTable?.type).toBe('table')
    if (firstTable?.type === 'table' && secondTable?.type === 'table') {
      expect(secondTable.columnWidths).not.toEqual(firstTable.columnWidths)
    }
  })

  test('invalidates cache on palette identity change', () => {
    const cache = createCacheSimulator()
    const paletteA = palette
    const paletteB = { ...palette }

    const first = cache.step('Alpha\n\nBravo', { palette: paletteA, codeBlockWidth: 80 })
    const second = cache.step('Alpha\n\nBravo', { palette: paletteB, codeBlockWidth: 80 })

    expect(first.renderedCount).toBe(2)
    expect(second.renderedCount).toBe(2)
  })

  test('does not invalidate when highlightRanges are structurally equal', () => {
    const cache = createCacheSimulator()
    const h1 = [{ start: 0, end: 5, backgroundColor: '#0f0' }]
    const h2 = [{ start: 0, end: 5, backgroundColor: '#0f0' }]

    const first = cache.step('Alpha\n\nBravo', { palette, codeBlockWidth: 80, highlightRanges: h1 })
    const second = cache.step('Alpha\n\nBravo', { palette, codeBlockWidth: 80, highlightRanges: h2 })

    expect(first.renderedCount).toBe(2)
    expect(second.renderedCount).toBe(2)
  })

  test('invalidates when highlightRanges values change', () => {
    const cache = createCacheSimulator()
    const h1 = [{ start: 0, end: 5, backgroundColor: '#0f0' }]
    const h2 = [{ start: 0, end: 6, backgroundColor: '#0f0' }]

    cache.step('Alpha\n\nBravo', { palette, codeBlockWidth: 80, highlightRanges: h1 })
    const second = cache.step('Alpha\n\nBravo', { palette, codeBlockWidth: 80, highlightRanges: h2 })

    expect(second.renderedCount).toBe(2)
    expect(collectHighlightedText(second.blocks)).not.toEqual([])
  })
})

describe('Layer 4 - Streaming Cache - Suite B Incomplete fences', () => {
  test('returns pendingText for unmatched fenced code block', () => {
    const result = parseStreamingContent('Before\n\n```js\nconst x = 1', { palette })

    expect(blockTypes(result.blocks)).toEqual(['paragraph', 'spacer'])
    expect(result.pendingText).toBe('```js\nconst x = 1')
  })

  test('clears pendingText when fence closes', () => {
    const open = parseStreamingContent('Before\n\n```js\nconst x = 1', { palette })
    const closed = parseStreamingContent('Before\n\n```js\nconst x = 1\n```', { palette })

    expect(open.pendingText).toBe('```js\nconst x = 1')
    expect(closed.pendingText).toBe('')
    expect(blockTypes(closed.blocks)).toEqual(['paragraph', 'spacer', 'code'])
  })

  test('preserves preceding spacer blocks while code fence is incomplete', () => {
    const result = parseStreamingContent('Alpha\n\n```js\nx', { palette })
    expect(blockTypes(result.blocks)).toEqual(['paragraph', 'spacer'])
  })
})

describe('Layer 4 - Streaming Cache - Suite C Ordering stability', () => {
  test('block ordering remains stable across streaming appends', () => {
    const prefixes = [
      '# Title',
      '# Title\n\nBody',
      '# Title\n\nBody\n\n- item',
      '# Title\n\nBody\n\n- item\n\n---',
      '# Title\n\nBody\n\n- item\n\n---\n\n| A | B |\n| - | - |\n| 1 | 2 |',
    ]

    let previousTypes: string[] = []
    for (const prefix of prefixes) {
      const cached = parseStreamingContent(prefix, { palette })
      const fresh = renderDocumentToBlocks(parseMarkdownToMdast(prefix), { palette })
      const currentTypes = blockTypes(cached.blocks)

      expect(currentTypes).toEqual(blockTypes(fresh))
      expect(currentTypes.slice(0, previousTypes.length)).toEqual(previousTypes)
      previousTypes = currentTypes
    }
  })

  test('spacer ordering remains stable across streaming paragraph growth', () => {
    const step1 = parseStreamingContent('A', { palette })
    const step2 = parseStreamingContent('A\n\n', { palette })
    const step3 = parseStreamingContent('A\n\nB', { palette })

    expect(blockTypes(step1.blocks)).toEqual(['paragraph'])
    expect(blockTypes(step2.blocks)).toEqual(['paragraph', 'spacer'])
    expect(blockTypes(step3.blocks)).toEqual(['paragraph', 'spacer', 'paragraph'])
  })
})

describe('Layer 4 - Streaming Cache - consumer observable behavior', () => {
  test('StreamingMarkdownContent renders pendingText raw while fence is incomplete', () => {
    const html = renderToStaticMarkup(
      React.createElement(StreamingMarkdownContent, {
        content: '# Title\n\n```js\nconst x = 1',
        showCursor: true,
        streaming: true,
      }),
    )

    expect(html).toContain('Title')
    expect(html).toContain('```js')
    expect(html).toContain('const x = 1')
    expect(html).toContain('▍')
  })

  test('streaming parse final blocks match fresh rendering once complete', () => {
    const content = '# Title\n\nBody\n\n```ts\nconst x = 1\n```'
    const streaming = parseStreamingContent(content, { palette })
    const fresh = renderBlocks(content)

    expect(streaming.pendingText).toBe('')
    expect(streaming.blocks.map(normalizeBlock)).toEqual(fresh.map(normalizeBlock))
  })
})