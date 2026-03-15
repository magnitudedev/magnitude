import { describe, expect, test } from 'bun:test'
import { renderDocumentToBlocks, slugify, spansToText, type HighlightRange, type Span } from './render-blocks'
import {
  baseOptions,
  blockTypes,
  collectText,
  getSingleBlock,
  palette,
  renderBlocks,
} from './test-markdown-helpers'

describe('render-blocks layer - Suite A: spacer generation / block boundaries', () => {
  test('renders spacer block between adjacent paragraphs separated by one blank line', () => {
    const blocks = renderBlocks('Alpha\n\nBeta')
    expect(blockTypes(blocks)).toEqual(['paragraph', 'spacer', 'paragraph'])
    expect(blocks[1]).toEqual({ type: 'spacer', lines: 1 })
  })

  test('renders spacer block with exact count for multiple blank lines', () => {
    const blocks = renderBlocks('Alpha\n\n\nBeta')
    expect(blocks[1]?.type).toBe('spacer')
    expect(blocks[1]).toEqual({ type: 'spacer', lines: 2 })
  })

  test('preserves spacers between heading and paragraph', () => {
    expect(blockTypes(renderBlocks('# Title\n\nBody'))).toEqual(['heading', 'spacer', 'paragraph'])
  })

  test('preserves spacers around divider', () => {
    expect(blockTypes(renderBlocks('Before\n\n---\n\nAfter'))).toEqual([
      'paragraph',
      'spacer',
      'divider',
      'spacer',
      'paragraph',
    ])
  })

  test('preserves spacers around table', () => {
    expect(blockTypes(renderBlocks('Before\n\n| A | B |\n| - | - |\n| 1 | 2 |\n\nAfter'))).toEqual([
      'paragraph',
      'spacer',
      'table',
      'spacer',
      'paragraph',
    ])
  })

  test('preserves spacers around code block', () => {
    expect(blockTypes(renderBlocks('Before\n\n```js\nx()\n```\n\nAfter'))).toEqual([
      'paragraph',
      'spacer',
      'code',
      'spacer',
      'paragraph',
    ])
  })

  test('preserves spacers around mermaid block', () => {
    expect(blockTypes(renderBlocks('Before\n\n```mermaid\ngraph TD\nA --> B\n```\n\nAfter'))).toEqual([
      'paragraph',
      'spacer',
      'mermaid',
      'spacer',
      'paragraph',
    ])
  })
})

describe('render-blocks layer - Suite B: table block data', () => {
  test('renders table headers rows and column widths', () => {
    const block = getSingleBlock('| Name | Age |\n| ---- | --- |\n| Alice | 30 |\n| Bob | 25 |')
    expect(block.type).toBe('table')
    if (block.type !== 'table') return
    expect(block.headers.map(spansToText)).toEqual(['Name', 'Age'])
    expect(block.rows.map((row) => row.map(spansToText))).toEqual([
      ['Alice', '30'],
      ['Bob', '25'],
    ])
    expect(block.columnWidths).toHaveLength(2)
  })

  test('uses larger natural width for longer cells before clipping', () => {
    const block = getSingleBlock('| Short | Much Longer Header |\n| ----- | ------------------ |\n| x | value |')
    expect(block.type).toBe('table')
    if (block.type !== 'table') return
    expect(block.columnWidths[1]).toBeGreaterThan(block.columnWidths[0]!)
  })

  test('shrinks table column widths when available width is constrained', () => {
    const wide = renderBlocks('| Short | Much Longer Header |\n| ----- | ------------------ |\n| x | value |', {
      codeBlockWidth: 80,
    })[0]
    const narrow = renderBlocks('| Short | Much Longer Header |\n| ----- | ------------------ |\n| x | value |', {
      codeBlockWidth: 24,
    })[0]

    expect(wide?.type).toBe('table')
    expect(narrow?.type).toBe('table')
    if (wide?.type !== 'table' || narrow?.type !== 'table') return

    expect(narrow.columnWidths).not.toEqual(wide.columnWidths)
    expect(narrow.columnWidths.reduce((a, b) => a + b, 0)).toBeLessThan(
      wide.columnWidths.reduce((a, b) => a + b, 0),
    )
  })

  test('currently drops parsed table alignment metadata', () => {
    const block = getSingleBlock('| Left | Center | Right |\n|:-----|:------:|------:|\n| a | b | c |')
    expect(block.type).toBe('table')
    if (block.type !== 'table') return
    expect(block.headers.every((cell) => cell.every((span) => !('align' in span)))).toBe(true)
    expect(block.rows.every((row) => row.every((cell) => cell.every((span) => !('align' in span))))).toBe(true)
  })

  test('table cell inline formatting is preserved semantically', () => {
    const block = getSingleBlock('| A | B |\n| - | - |\n| **bold** | *italic* |\n| `code` | [link](x) |')
    expect(block.type).toBe('table')
    if (block.type !== 'table') return

    expect(block.rows[0]?.[0]?.some((span) => span.bold)).toBe(true)
    expect(block.rows[0]?.[1]?.some((span) => span.italic)).toBe(true)
    expect(block.rows[1]?.[0]?.some((span) => span.fg === palette.inlineCodeFg)).toBe(true)
    expect(block.rows[1]?.[1]?.some((span) => span.fg === palette.linkFg)).toBe(true)
  })
})

describe('render-blocks layer - Suite C: palette color application to spans', () => {
  test('applies heading palette color by level', () => {
    const block = getSingleBlock('### Heading')
    expect(block.type).toBe('heading')
    if (block.type !== 'heading') return
    expect(block.content[0]?.fg).toBe(palette.headingFg[3])
    expect(block.content[0]?.bold).toBe(true)
  })

  test('applies inline code palette color', () => {
    const block = getSingleBlock('Use `npm test`')
    expect(block.type).toBe('paragraph')
    if (block.type !== 'paragraph') return
    const codeSpan = block.content.find((span) => span.fg === palette.inlineCodeFg)
    expect(codeSpan?.text).toContain('npm test')
    expect(codeSpan?.bold).toBe(true)
  })

  test('applies link palette color', () => {
    const block = getSingleBlock('[Magnitude](https://example.com)')
    expect(block.type).toBe('paragraph')
    if (block.type !== 'paragraph') return
    expect(block.content.some((span) => span.fg === palette.linkFg)).toBe(true)
  })

  test('applies blockquote paragraph color to quoted text', () => {
    const block = getSingleBlock('> quoted text')
    expect(block.type).toBe('blockquote')
    if (block.type !== 'blockquote') return
    const paragraph = block.content[0]
    expect(paragraph?.type).toBe('paragraph')
    if (paragraph?.type !== 'paragraph') return
    expect(paragraph.content.every((span) => span.fg === palette.blockquoteTextFg)).toBe(true)
  })

  test('applies list marker palette color', () => {
    const block = getSingleBlock('- item')
    expect(block.type).toBe('list')
    if (block.type !== 'list') return
    expect(block.items[0]?.marker).toBe('- ')
    expect(block.items[0]?.markerFg).toBe(palette.listBulletFg)
  })
})

describe('render-blocks layer - Suite D: divider and mermaid generation', () => {
  test('renders horizontal rule as divider block', () => {
    expect(getSingleBlock('---').type).toBe('divider')
  })

  test('renders mermaid fence as mermaid block when ascii conversion succeeds', () => {
    const block = getSingleBlock('```mermaid\ngraph TD\nA --> B\n```')
    expect(block.type).toBe('mermaid')
    if (block.type !== 'mermaid') return
    expect(block.ascii.length).toBeGreaterThan(0)
    expect(/[AB│─┌┐└┘]/.test(block.ascii)).toBe(true)
  })

  test('falls back to code block for invalid mermaid', () => {
    const block = getSingleBlock('```mermaid\n@@ invalid\n```')
    expect(['code', 'mermaid']).toContain(block.type)
    if (block.type === 'code') {
      expect(block.language).toBe('mermaid')
      expect(block.rawCode).toContain('@@ invalid')
    } else {
      expect(block.ascii.length).toBeGreaterThan(0)
    }
  })
})

describe('render-blocks layer - Suite E: list item structure', () => {
  test('renders bullet list markers and nested content', () => {
    const block = getSingleBlock('- one\n- two')
    expect(block.type).toBe('list')
    if (block.type !== 'list') return
    expect(block.style).toBe('bullet')
    expect(block.items.map((item) => item.marker)).toEqual(['- ', '- '])
    expect(collectText(block.items[0]!.content)).toContain('one')
    expect(collectText(block.items[1]!.content)).toContain('two')
  })

  test('renders ordered list markers with actual numbers', () => {
    const block = getSingleBlock('9. nine\n10. ten')
    expect(block.type).toBe('list')
    if (block.type !== 'list') return
    expect(block.items.map((item) => item.marker)).toEqual(['9. ', '10. '])
  })

  test('renders task list checked state and markers', () => {
    const block = getSingleBlock('- [ ] todo\n- [x] done')
    expect(block.type).toBe('list')
    if (block.type !== 'list') return
    expect(block.style).toBe('task')
    expect(block.items[0]).toMatchObject({ checked: false, marker: '- [ ] ' })
    expect(block.items[1]).toMatchObject({ checked: true, marker: '- [x] ' })
  })

  test('renders nested list as nested child blocks within item content', () => {
    const block = getSingleBlock('- parent\n  - child')
    expect(block.type).toBe('list')
    if (block.type !== 'list') return
    expect(block.items[0]?.content.map((child) => child.type)).toEqual(['paragraph', 'list'])
  })

  test('preserves blank lines inside list item as spacer blocks', () => {
    const block = getSingleBlock('- first paragraph\n\n  second paragraph')
    expect(block.type).toBe('list')
    if (block.type !== 'list') return
    expect(block.items[0]?.content.map((child) => child.type)).toEqual(['paragraph', 'spacer', 'paragraph'])
  })
})

describe('render-blocks layer - Suite F: blockquote structure', () => {
  test('renders blockquote with nested paragraph content', () => {
    const block = getSingleBlock('> quote')
    expect(block.type).toBe('blockquote')
    if (block.type !== 'blockquote') return
    expect(block.content[0]?.type).toBe('paragraph')
  })

  test('preserves blank lines inside blockquote as spacers', () => {
    const block = getSingleBlock('> first\n>\n> second')
    expect(block.type).toBe('blockquote')
    if (block.type !== 'blockquote') return
    expect(block.content.map((child) => child.type)).toEqual(['paragraph', 'spacer', 'paragraph'])
  })

  test('renders nested list inside blockquote', () => {
    const block = getSingleBlock('> intro\n>\n> - item')
    expect(block.type).toBe('blockquote')
    if (block.type !== 'blockquote') return
    expect(block.content.map((child) => child.type)).toEqual(['paragraph', 'spacer', 'list'])
  })

  test('renders nested blockquote structure recursively', () => {
    const block = getSingleBlock('> outer\n> > inner')
    expect(block.type).toBe('blockquote')
    if (block.type !== 'blockquote') return
    expect(block.content.map((child) => child.type)).toEqual(['paragraph', 'blockquote'])
  })
})

describe('render-blocks layer - Suite G: code block structure', () => {
  test('renders code block language raw code and syntax lines', () => {
    const block = getSingleBlock('```ts\nconst x = 1\n```')
    expect(block.type).toBe('code')
    if (block.type !== 'code') return
    expect(block.language).toBe('ts')
    expect(block.rawCode).toBe('const x = 1')
    expect(block.lines).toHaveLength(1)
  })

  test('preserves empty line in code block', () => {
    const block = getSingleBlock('```txt\nline1\n\nline3\n```')
    expect(block.type).toBe('code')
    if (block.type !== 'code') return
    expect(block.lines).toHaveLength(3)
    expect(spansToText(block.lines[1] ?? [])).toBe(' ')
  })

  test('uses fallback code text color when syntax highlight unavailable', () => {
    const block = getSingleBlock('```unknownlang\nabc\n```')
    expect(block.type).toBe('code')
    if (block.type !== 'code') return
    expect(block.lines.flat().every((span) => span.fg === palette.codeTextFg)).toBe(true)
  })

  test('applies source highlight backgrounds inside code lines', () => {
    const source = '```js\nconst value = 42\n```'
    const start = source.indexOf('value')
    const block = renderBlocks(source, {
      ...baseOptions,
      highlights: [{ start, end: start + 'value'.length, backgroundColor: 'magenta' }],
    })[0]

    expect(block?.type).toBe('code')
    if (block?.type !== 'code') return
    expect(block.lines).toHaveLength(1)
    expect(block.lines[0]?.map((span) => span.text)).toEqual(['const ', 'value', ' = 42'])
    expect(block.lines[0]?.[0]?.bg).toBeUndefined()
    expect(block.lines[0]?.[1]?.bg).toBe('magenta')
    expect(block.lines[0]?.[2]?.bg).toBeUndefined()
  })
})

describe('render-blocks layer - Suite H: highlight range splitting correctness', () => {
  test('splits paragraph text into pre-highlight highlight post-highlight spans', () => {
    const source = 'abcdef'
    const block = renderBlocks(source, {
      highlights: [{ start: 2, end: 4, backgroundColor: 'yellow' }],
    })[0]

    expect(block?.type).toBe('paragraph')
    if (block?.type !== 'paragraph') return
    expect(block.content.map((span) => span.text)).toEqual(['ab', 'cd', 'ef'])
    expect(block.content[0]?.bg).toBeUndefined()
    expect(block.content[1]?.bg).toBe('yellow')
    expect(block.content[2]?.bg).toBeUndefined()
  })

  test('splits through formatted content without losing styles', () => {
    const source = '**abcd**'
    const start = source.indexOf('b')
    const block = renderBlocks(source, {
      highlights: [{ start, end: start + 2, backgroundColor: 'cyan' }],
    })[0]

    expect(block?.type).toBe('paragraph')
    if (block?.type !== 'paragraph') return
    expect(spansToText(block.content)).toBe('abcd')
    expect(block.content.every((span) => span.bold)).toBe(true)
    expect(block.content.some((span) => span.text === 'bc' && span.bg === 'cyan')).toBe(true)
  })

  test('splits artifact ref text while preserving ref metadata', () => {
    const source = 'See [[file.ts#part|label]] now'
    // Highlight the entire wiki link node source range
    const start = source.indexOf('[[')
    const end = source.indexOf(']]') + 2
    const block = renderBlocks(source, {
      highlights: [{ start, end, backgroundColor: 'green' }],
    })[0]

    expect(block?.type).toBe('paragraph')
    if (block?.type !== 'paragraph') return
    const refSpans = block.content.filter((span) => span.ref?.name === 'file.ts')
    expect(refSpans.length).toBeGreaterThan(0)
    expect(refSpans.every((span) => span.ref?.section === 'part')).toBe(true)
    expect(refSpans.every((span) => span.ref?.label === 'label')).toBe(true)
    expect(refSpans.some((span) => span.text.includes('label') && span.bg === 'green')).toBe(true)
  })
})

describe('render-blocks layer - Suite I: edge cases', () => {
  test('renders empty document to no blocks', () => {
    expect(renderBlocks('')).toEqual([])
  })

  test('handles deeply nested mixed structures', () => {
    const blocks = renderBlocks('> - one\n>   > quote\n>   > - two')
    expect(blocks).toHaveLength(1)
    expect(blocks[0]?.type).toBe('blockquote')
    expect(() => collectText(blocks)).not.toThrow()
    expect(collectText(blocks)).toContain('one')
    expect(collectText(blocks)).toContain('quote')
    expect(collectText(blocks)).toContain('two')
  })

  test('renders mixed block types in stable order', () => {
    const blocks = renderBlocks(
      '# Title\n\nBody\n\n- item\n\n---\n\n| A | B |\n| - | - |\n| 1 | 2 |\n\n```ts\nconst x = 1\n```',
    )
    expect(blockTypes(blocks)).toEqual([
      'heading',
      'spacer',
      'paragraph',
      'spacer',
      'list',
      'spacer',
      'divider',
      'spacer',
      'table',
      'spacer',
      'code',
    ])
  })
})

describe('render-blocks layer - Suite J: wiki-link / artifact ref parsing', () => {
  test('renders [[artifact-id]] as span with ref metadata', () => {
    const block = getSingleBlock('[[artifact-id]]')
    expect(block.type).toBe('paragraph')
    if (block.type !== 'paragraph') return
    expect(block.content).toHaveLength(1)
    expect(block.content[0]?.text).toBe('[[artifact-id]]')
    expect(block.content[0]?.ref).toEqual({ name: 'artifact-id', section: undefined, label: undefined })
  })

  test('renders [[artifact-id#section]] as span with ref name and section', () => {
    const block = getSingleBlock('[[artifact-id#section]]')
    expect(block.type).toBe('paragraph')
    if (block.type !== 'paragraph') return
    expect(block.content[0]?.ref).toEqual({ name: 'artifact-id', section: 'section', label: undefined })
  })

  test('renders [[artifact-id#section|label]] as span with ref name, section, and label text', () => {
    const block = getSingleBlock('[[artifact-id#section|label]]')
    expect(block.type).toBe('paragraph')
    if (block.type !== 'paragraph') return
    expect(block.content[0]?.text).toBe('label')
    expect(block.content[0]?.ref).toEqual({ name: 'artifact-id', section: 'section', label: 'label' })
  })

  test('preserves wiki-link refs inside formatted text', () => {
    const block = getSingleBlock('**See [[artifact-id#section|label]]**')
    expect(block.type).toBe('paragraph')
    if (block.type !== 'paragraph') return
    const refSpan = block.content.find((span) => span.ref?.name === 'artifact-id')
    expect(refSpan?.bold).toBe(true)
    expect(refSpan?.ref).toEqual({ name: 'artifact-id', section: 'section', label: 'label' })
  })

  test('does not parse wiki-links inside code spans or code blocks', () => {
    const inline = getSingleBlock('`[[artifact-id]]`')
    expect(inline.type).toBe('paragraph')
    if (inline.type === 'paragraph') {
      expect(inline.content.some((span) => span.ref)).toBe(false)
    }

    const code = getSingleBlock('```txt\n[[artifact-id]]\n```')
    expect(code.type).toBe('code')
    if (code.type === 'code') {
      expect(code.lines.flat().some((span) => span.ref)).toBe(false)
    }
  })
})

describe('render-blocks layer - extras', () => {
  test('slugify normalizes heading text', () => {
    expect(slugify('Hello, World!')).toBe('hello-world')
  })

  test('spansToText joins span text', () => {
    const spans: Span[] = [{ text: 'a' }, { text: 'b', bold: true }]
    expect(spansToText(spans)).toBe('ab')
  })

  test('highlight range type remains usable in test suite', () => {
    const range: HighlightRange = { start: 1, end: 2, backgroundColor: 'x' }
    expect(range.end - range.start).toBe(1)
  })

  test('collectText returns text for block arrays', () => {
    expect(collectText(renderBlocks('Alpha\n\nBeta'))).toContain('Alpha')
  })
})