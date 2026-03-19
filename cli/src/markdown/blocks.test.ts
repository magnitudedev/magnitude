import { describe, expect, it } from 'bun:test'
import { parseMarkdownToMdast } from './parse'
import {
  extractHeadingSlugsFromBlocks,
  renderDocumentToBlocks,
  slugify,
  spansToText,
  type Span,
} from './blocks'
import { baseOptions, blockTypes, collectText, findBlocks, getSingleBlock, normalizeBlock, renderBlocks } from '../utils/test-markdown-helpers'

describe('markdown/blocks', () => {
  describe('paragraphs', () => {
    it('renders basic text', () => {
      const block = getSingleBlock('hello world')
      expect(block.type).toBe('paragraph')
      if (block.type === 'paragraph') expect(spansToText(block.content)).toBe('hello world')
    })

    it('renders inline formatting and links', () => {
      const block = getSingleBlock('**bold** *italic* ~~gone~~ `code` [link](https://example.com)')
      expect(block.type).toBe('paragraph')
      if (block.type !== 'paragraph') return
      expect(spansToText(block.content)).toContain('bold italic gone  code  link')
      expect(block.content.some((s) => s.bold)).toBeTrue()
      expect(block.content.some((s) => s.italic)).toBeTrue()
      expect(block.content.some((s) => s.dim)).toBeTrue()
      expect(block.content.some((s) => s.text.includes('code'))).toBeTrue()
      expect(block.content.some((s) => s.text.includes('link') && !!s.fg)).toBeTrue()
    })
  })

  describe('headings', () => {
    it('renders levels 1-6 with slugs', () => {
      const md = '# h1\n## h2\n### h3\n#### h4\n##### h5\n###### h6'
      const blocks = renderBlocks(md).filter((b) => b.type === 'heading')
      expect(blocks).toHaveLength(6)
      expect((blocks[0] as any).level).toBe(1)
      expect((blocks[5] as any).level).toBe(6)
      expect(extractHeadingSlugsFromBlocks(blocks)).toEqual(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    })

    it('supports inline formatting in heading content', () => {
      const block = getSingleBlock('# **Hello** *World*')
      expect(block.type).toBe('heading')
      if (block.type !== 'heading') return
      expect(spansToText(block.content)).toBe('Hello World')
      expect(block.content.some((s) => s.bold)).toBeTrue()
      expect(block.content.some((s) => s.italic)).toBeTrue()
    })
  })

  describe('code blocks', () => {
    it('renders language and multiline code', () => {
      const block = getSingleBlock('```ts\nconst a = 1\nconsole.log(a)\n```')
      expect(block.type).toBe('code')
      if (block.type !== 'code') return
      expect(block.language).toBe('ts')
      expect(block.rawCode).toBe('const a = 1\nconsole.log(a)')
      expect(block.lines.length).toBe(2)
    })

    it('renders code block without language', () => {
      const block = getSingleBlock('```\njust text\n```')
      expect(block.type).toBe('code')
      if (block.type !== 'code') return
      expect(block.language).toBeUndefined()
      expect(block.rawCode).toBe('just text')
    })
  })

  describe('lists', () => {
    it('renders bullet, ordered, task, and nested lists', () => {
      const md = '- a\n  - nested\n1. first\n- [x] done\n- [ ] todo'
      const lists = findBlocks(md, 'list')
      expect(lists.length).toBeGreaterThanOrEqual(3)
      const styles = lists.map((l: any) => l.style)
      expect(styles).toContain('bullet')
      expect(styles).toContain('ordered')
      expect(styles).toContain('task')

      const first = lists[0] as any
      expect(first.items[0].content.some((c: any) => c.type === 'list')).toBeTrue()
    })
  })

  describe('tables', () => {
    it('renders headers, rows, and alignments without width fields', () => {
      const block = getSingleBlock('| a | b |\n| :-- | :-: |\n| 1 | 2 |')
      expect(block.type).toBe('table')
      if (block.type !== 'table') return
      expect(block.headers).toHaveLength(2)
      expect(block.rows).toHaveLength(1)
      expect(block.alignments).toEqual(['left', 'center'])
      expect((block as any).widths).toBeUndefined()
    })
  })

  describe('blockquotes', () => {
    it('renders simple, nested, and mixed content', () => {
      const block = getSingleBlock('> quote\n>\n> - item\n> > nested')
      expect(block.type).toBe('blockquote')
      if (block.type !== 'blockquote') return
      expect(block.content.some((b) => b.type === 'paragraph')).toBeTrue()
      expect(block.content.some((b) => b.type === 'list')).toBeTrue()
      expect(block.content.some((b) => b.type === 'blockquote')).toBeTrue()
    })
  })

  describe('dividers', () => {
    it('renders ---, ***, ___ as dividers', () => {
      const blocks = renderBlocks('---\n***\n___').filter((b) => b.type === 'divider')
      expect(blocks).toHaveLength(3)
    })
  })

  describe('mermaid', () => {
    it('renders mermaid code fence as mermaid block', () => {
      const block = getSingleBlock('```mermaid\ngraph TD\nA-->B\n```')
      expect(['mermaid', 'code']).toContain(block.type)
    })
  })

  describe('spacers', () => {
    it('inserts spacer blocks between document blocks', () => {
      const blocks = renderBlocks('one\n\ntwo')
      expect(blockTypes(blocks)).toEqual(['paragraph', 'spacer', 'paragraph'])
    })
  })

  describe('helpers', () => {
    it('slugify handles various heading text', () => {
      expect(slugify('Hello, World!')).toBe('hello-world')
      expect(slugify('  A__B  ')).toBe('a-b')
      expect(slugify('123 Title')).toBe('123-title')
    })

    it('spansToText concatenates span text', () => {
      const spans: Span[] = [{ text: 'a' }, { text: 'b' }, { text: 'c' }]
      expect(spansToText(spans)).toBe('abc')
    })

    it('extractHeadingSlugsFromBlocks returns only heading slugs', () => {
      const blocks = renderBlocks('# A\ntext\n## B')
      expect(extractHeadingSlugsFromBlocks(blocks)).toEqual(['a', 'b'])
    })
  })

  describe('wiki links and source ranges', () => {
    it('maps wiki links to span.ref with name/section/label', () => {
      const block = getSingleBlock('[[artifact-name]] [[name#section]] [[name|label]]')
      expect(block.type).toBe('paragraph')
      if (block.type !== 'paragraph') return
      const refs = block.content.filter((s) => s.ref).map((s) => s.ref)
      expect(refs).toEqual([
        { name: 'artifact-name', section: undefined, label: undefined },
        { name: 'name', section: 'section', label: undefined },
        { name: 'name', section: undefined, label: 'label' },
      ])
    })

    it('includes source start/end for blocks', () => {
      const doc = parseMarkdownToMdast('# head\n\npara')
      const blocks = renderDocumentToBlocks(doc, baseOptions)
      const concrete = blocks.filter((b) => b.type !== 'spacer') as Array<{ source: { start: number; end: number } }>
      expect(concrete.every((b) => b.source.start >= 0 && b.source.end >= b.source.start)).toBeTrue()
    })
  })

  describe('inline html', () => {
    it('renders inline html as text spans', () => {
      const block = getSingleBlock('before <span>mid</span> after')
      expect(collectText(block)).toContain('<span>mid</span>')
      expect(normalizeBlock(block)).toBeDefined()
    })
  })
})
