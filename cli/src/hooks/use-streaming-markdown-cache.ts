import { useMemo, useRef } from 'react'
import type { Root } from 'mdast'
import { hasOddFenceCount } from '../utils/markdown-content-renderer'
import { parseMarkdownToMdast } from '../utils/markdown-parser'
import { renderDocumentToBlocks, type Block, type HighlightRange, type MarkdownPalette } from '../utils/render-blocks'

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

function appendTrailingSpacer(blocks: Block[], source: string): Block[] {
  const match = source.match(/\n\n+$/)
  if (!match) return blocks
  const lines = match[0].length - 1
  if (lines <= 0) return blocks
  return [...blocks, { type: 'spacer', lines }]
}

export function useStreamingMarkdownCache(
  content: string,
  options: StreamingMarkdownCacheOptions,
): StreamingMarkdownCacheResult {
  const cacheRef = useRef<{
    prevContent: string
    prevDoc: Root | null
    prevBlocks: Block[]
    prevPendingText: string
    prevHighlightRanges: HighlightRange[] | undefined
    prevPalette: MarkdownPalette | null
    prevCodeBlockWidth: number | undefined
    prevStreaming: boolean
  }>({
    prevContent: '',
    prevDoc: null,
    prevBlocks: [],
    prevPendingText: '',
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

    const doc = parseMarkdownToMdast(completeSection)
    const rendered = renderDocumentToBlocks(doc, {
      palette,
      codeBlockWidth,
      highlights: highlightRanges,
    })
    const finalBlocks =
      paletteChanged || codeBlockWidthChanged || highlightChanged || content !== cache.prevContent
        ? appendTrailingSpacer(rendered, completeSection)
        : cache.prevBlocks

    cache.prevContent = content
    cache.prevDoc = doc
    cache.prevBlocks = finalBlocks
    cache.prevPendingText = pendingText
    cache.prevHighlightRanges = highlightRanges
    cache.prevPalette = palette
    cache.prevCodeBlockWidth = codeBlockWidth

    return { blocks: finalBlocks, pendingText }
  }, [content, options.palette, options.codeBlockWidth, options.highlightRanges, options.streaming])
}