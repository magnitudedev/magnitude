import { describe, expect, it } from 'bun:test'
import { hasOddFenceCount } from './streaming'
import { parseMarkdownToMdast } from './parse'
import { renderDocumentToBlocks } from './blocks'
import { baseOptions, blockTypes } from '../utils/test-markdown-helpers'

const splitStreaming = (content: string) => {
  let completeSection = content
  let pendingText = ''
  if (hasOddFenceCount(content)) {
    const lastFenceIndex = content.lastIndexOf('```')
    if (lastFenceIndex !== -1) {
      completeSection = content.slice(0, lastFenceIndex)
      pendingText = content.slice(lastFenceIndex)
    }
  }
  const blocks = completeSection.trim()
    ? renderDocumentToBlocks(parseMarkdownToMdast(completeSection), baseOptions)
    : []
  const match = completeSection.match(/\n\n+$/)
  if (match) {
    const lines = match[0].length - 1
    if (lines > 0) blocks.push({ type: 'spacer', lines } as any)
  }
  return { blocks, pendingText }
}

describe('markdown/streaming', () => {
  describe('hasOddFenceCount', () => {
    it('returns false for no fences', () => expect(hasOddFenceCount('hello')).toBeFalse())
    it('returns true for one opening fence', () => expect(hasOddFenceCount('```')).toBeTrue())
    it('returns false for matched pair', () => expect(hasOddFenceCount('```\na\n```')).toBeFalse())
    it('returns true for three fences', () => expect(hasOddFenceCount('```\na\n```\n```')).toBeTrue())
    it('counts language-tag fence', () => expect(hasOddFenceCount('```python\nx')).toBeTrue())
    it('returns false for multiple pairs', () => expect(hasOddFenceCount('```\na\n```\n```\nb\n```')).toBeFalse())
    it('returns false for empty string', () => expect(hasOddFenceCount('')).toBeFalse())
  })

  describe('split behavior', () => {
    it('complete content yields blocks and no pending', () => {
      const out = splitStreaming('# hi\n\ntext')
      expect(out.pendingText).toBe('')
      expect(blockTypes(out.blocks)).toContain('heading')
      expect(blockTypes(out.blocks)).toContain('paragraph')
    })

    it('unclosed fence yields complete blocks and pending tail', () => {
      const out = splitStreaming('before\n\n```ts\nconst x = 1')
      expect(out.pendingText.startsWith('```')).toBeTrue()
      expect(blockTypes(out.blocks)).toContain('paragraph')
      expect(blockTypes(out.blocks)).not.toContain('code')
    })

    it('only unclosed fence yields no blocks and all pending', () => {
      const out = splitStreaming('```ts\nconst x = 1')
      expect(out.blocks).toHaveLength(0)
      expect(out.pendingText).toBe('```ts\nconst x = 1')
    })

    it('preserves trailing newlines as spacer', () => {
      const out = splitStreaming('hello\n\n')
      expect(blockTypes(out.blocks)).toContain('spacer')
      const spacer = out.blocks[out.blocks.length - 1] as any
      expect(spacer.type).toBe('spacer')
      expect(spacer.lines).toBe(1)
    })
  })
})
