import { useMemo, useRef } from 'react'
import { parseMarkdown } from '@magnitude/markdown-cst'
import type { DocumentNode, DocumentItemNode } from '@magnitude/markdown-cst/src/schema'
import { hasOddFenceCount } from '../utils/markdown-content-renderer'
import { renderDocumentItemToBlocks, type Block, type HighlightRange, type MarkdownPalette } from '../utils/render-blocks'

interface StreamingMarkdownCacheOptions {
  palette: MarkdownPalette
  codeBlockWidth?: number
  highlightRanges?: HighlightRange[]
  streaming?: boolean
}

interface StreamingMarkdownCacheResult {
  blocks: Block[]
  pendingText: string
}

function areHighlightRangesEqual(
  previous: HighlightRange[] | undefined,
  next: HighlightRange[] | undefined,
): boolean {
  if (previous === next) return true
  if (!previous || !next) return !previous && !next
  if (previous.length !== next.length) return false

  return previous.every((range, index) => {
    const other = next[index]
    return (
      range.start === other.start &&
      range.end === other.end &&
      range.backgroundColor === other.backgroundColor
    )
  })
}

export function useStreamingMarkdownCache(
  content: string,
  options: StreamingMarkdownCacheOptions,
): StreamingMarkdownCacheResult {
  const cacheRef = useRef<{
    prevContent: string
    prevDoc: DocumentNode | null
    prevBlocks: Block[]
    prevPendingText: string
    itemBlockCache: WeakMap<DocumentItemNode, Block[]>
    prevHighlightRanges: HighlightRange[] | undefined
    prevPalette: MarkdownPalette | null
    prevCodeBlockWidth: number | undefined
    prevStreaming: boolean
  }>({
    prevContent: '',
    prevDoc: null,
    prevBlocks: [],
    prevPendingText: '',
    itemBlockCache: new WeakMap(),
    prevHighlightRanges: undefined,
    prevPalette: null,
    prevCodeBlockWidth: undefined,
    prevStreaming: false,
  })

  return useMemo(() => {
    const cache = cacheRef.current
    const { palette, codeBlockWidth, highlightRanges } = options

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

    const highlightChanged = !areHighlightRangesEqual(highlightRanges, cache.prevHighlightRanges)
    const paletteChanged = palette !== cache.prevPalette
    const codeBlockWidthChanged = codeBlockWidth !== cache.prevCodeBlockWidth
    if (paletteChanged || codeBlockWidthChanged) {
      cache.itemBlockCache = new WeakMap()
    } else if (highlightChanged) {
      cache.itemBlockCache = new WeakMap()
    }

    const doc = parseMarkdown(completeSection, {
      previous: cache.prevDoc ?? undefined,
    })

    const allBlocks: Block[] = []
    for (const item of doc.content) {
      let itemBlocks = cache.itemBlockCache.get(item)
      if (!itemBlocks) {
        itemBlocks = renderDocumentItemToBlocks(item, {
          palette,
          codeBlockWidth,
          highlights: highlightRanges,
        })
        cache.itemBlockCache.set(item, itemBlocks)
      }
      allBlocks.push(...itemBlocks)
    }

    cache.prevContent = content
    cache.prevDoc = doc
    cache.prevBlocks = allBlocks
    cache.prevPendingText = pendingText
    cache.prevHighlightRanges = highlightRanges
    cache.prevPalette = palette
    cache.prevCodeBlockWidth = codeBlockWidth

    return { blocks: allBlocks, pendingText }
  }, [content, options.palette, options.codeBlockWidth, options.highlightRanges, options.streaming])
}