import { describe, expect, test } from 'bun:test'

import { parseMarkdown } from './index'
import type {
  BlockquoteContentNode,
  BlockquoteNode,
  BulletItemNode,
  BulletListNode,
  CodeBlockNode,
  DocumentContentNode,
  DocumentItemNode,
  DocumentNode,
  HeadingNode,
  HtmlBlockNode,
  InlineNode,
  OrderedItemNode,
  OrderedListNode,
  ParagraphNode,
  RootBlockNode,
  TableCellNode,
  TableNode,
  TableRowNode,
  TaskItemNode,
  TaskListNode,
} from './schema'
import { findDivergence, findStablePrefixCount } from './incremental'

function contentTypes(source: string, previous?: ReturnType<typeof parseMarkdown>) {
  return parseMarkdown(source, previous ? { previous } : undefined).content.map(item => item.content.type)
}

function inlineText(nodes?: readonly InlineNode[]): string {
  return (nodes ?? []).map((node) => {
    switch (node.type) {
      case 'text':
      case 'inlineCode':
        return node.text
      case 'image':
        return node.attrs.alt ?? ''
      case 'emphasis':
      case 'strong':
      case 'strikethrough':
      case 'link':
        return inlineText(node.content)
      case 'hardBreak':
      case 'softBreak':
        return '\n'
      default:
        return ''
    }
  }).join('')
}

function normalizeParagraphLike(block: ParagraphNode | HeadingNode) {
  return {
    type: block.type,
    position: [block.position.start.offset, block.position.end.offset],
    ...(block.type === 'heading' ? { level: block.attrs.level } : {}),
    text: inlineText(block.content),
  }
}

function normalizeCodeBlock(block: CodeBlockNode) {
  return {
    type: block.type,
    position: [block.position.start.offset, block.position.end.offset],
    language: block.attrs.language ?? null,
    value: (block.content ?? []).map((node) => node.text).join(''),
  }
}

function normalizeTableCell(cell: TableCellNode) {
  return {
    position: [cell.position.start.offset, cell.position.end.offset],
    text: inlineText(cell.content[0]?.content),
  }
}

function normalizeTableRow(row: TableRowNode) {
  return {
    position: [row.position.start.offset, row.position.end.offset],
    cells: row.content.map(normalizeTableCell),
  }
}

function normalizeTable(block: TableNode) {
  return {
    type: block.type,
    position: [block.position.start.offset, block.position.end.offset],
    rows: block.content.map(normalizeTableRow),
  }
}

function normalizeListItem(item: BulletItemNode | OrderedItemNode | TaskItemNode) {
  return {
    type: item.type,
    position: [item.position.start.offset, item.position.end.offset],
    children: item.content.map((child) => ({
      position: [child.position.start.offset, child.position.end.offset],
      content: normalizeBlock(child.content),
    })),
  }
}

function normalizeList(block: BulletListNode | OrderedListNode | TaskListNode) {
  return {
    type: block.type,
    position: [block.position.start.offset, block.position.end.offset],
    content: block.content.map((item) => (
      item.type === 'listItemBreak'
        ? {
            type: item.type,
            position: [item.position.start.offset, item.position.end.offset],
            blankLines: [...item.meta.blankLines],
            continuation: item.meta.continuation,
          }
        : normalizeListItem(item)
    )),
  }
}

function normalizeBlockquote(block: BlockquoteNode) {
  return {
    type: block.type,
    position: [block.position.start.offset, block.position.end.offset],
    content: block.content.map((item) => (
      item.type === 'blockquoteItemBreak'
        ? {
            type: item.type,
            position: [item.position.start.offset, item.position.end.offset],
            blankLines: [...item.meta.blankLines],
            continuation: item.meta.continuation,
          }
        : {
            type: item.type,
            position: [item.position.start.offset, item.position.end.offset],
            content: normalizeBlock(item.content),
          }
    )),
  }
}

function normalizeBlock(block: RootBlockNode | BlockquoteContentNode | DocumentContentNode): unknown {
  switch (block.type) {
    case 'paragraph':
    case 'heading':
      return normalizeParagraphLike(block)
    case 'codeBlock':
      return normalizeCodeBlock(block)
    case 'table':
      return normalizeTable(block)
    case 'bulletList':
    case 'orderedList':
    case 'taskList':
      return normalizeList(block)
    case 'blockquote':
      return normalizeBlockquote(block)
    case 'horizontalRule':
      return {
        type: block.type,
        position: [block.position.start.offset, block.position.end.offset],
      }
    case 'htmlBlock':
      return {
        type: block.type,
        position: [block.position.start.offset, block.position.end.offset],
        content: block.content,
      }
    case 'blankLines':
      return {
        type: block.type,
        position: [block.position.start.offset, block.position.end.offset],
        count: block.count,
      }
    case 'image':
      return {
        type: block.type,
        position: [block.position.start.offset, block.position.end.offset],
        src: block.attrs.src,
        alt: block.attrs.alt,
      }
    case 'definition':
      return {
        type: block.type,
        position: [block.position.start.offset, block.position.end.offset],
        label: block.label,
        url: block.url,
        title: block.title,
      }
    default:
      return {
        type: block.type,
        position: [block.position.start.offset, block.position.end.offset],
      }
  }
}

function normalizeItem(item: DocumentItemNode) {
  return {
    type: item.content.type,
    position: [item.content.position.start.offset, item.content.position.end.offset],
    content: normalizeBlock(item.content),
  }
}

function stripIdentity(doc: DocumentNode) {
  const content = doc.content.map(normalizeItem).reduce<Array<ReturnType<typeof normalizeItem>>>((items, item) => {
    const last = items[items.length - 1]
    if (last?.type === 'blankLines' && item.type === 'blankLines') {
      const lastContent = last.content as { type: 'blankLines'; count: number; position: [number, number] }
      const itemContent = item.content as { type: 'blankLines'; count: number; position: [number, number] }
      lastContent.count += itemContent.count
      lastContent.position = [lastContent.position[0], itemContent.position[1]]
      last.position = [last.position[0], item.position[1]]
      return items
    }
    items.push(item)
    return items
  }, [])

  return {
    source: doc.source,
    content,
  }
}

function expectStructurallyEqual(actual: DocumentNode, expected: DocumentNode) {
  expect(stripIdentity(actual)).toEqual(stripIdentity(expected))
}

function samplePrefixes(source: string, count = 10) {
  if (source.length === 0) return ['']
  const points = new Set<number>([1, source.length])
  const target = Math.min(count, source.length)
  for (let index = 1; index <= target; index += 1) {
    points.add(Math.max(1, Math.min(source.length, Math.ceil((index * source.length) / target))))
  }
  return [...points].sort((a, b) => a - b).map((end) => source.slice(0, end))
}

describe('incremental markdown parsing', () => {
  test('streaming paragraph append keeps a single paragraph', () => {
    const prefixes = [
      'Hello',
      'Hello world',
      'Hello world, this is',
      'Hello world, this is a test.',
    ]

    let previous = parseMarkdown(prefixes[0])

    expect(previous.content).toHaveLength(1)
    expect(previous.content[0]?.content.type).toBe('paragraph')

    for (const prefix of prefixes.slice(1)) {
      previous = parseMarkdown(prefix, { previous })
      expect(previous.content).toHaveLength(1)
      expect(previous.content[0]?.content.type).toBe('paragraph')
    }
  })

  test('streaming multiple paragraphs produces paragraph, blankLines, paragraph and reuses first paragraph once second starts', () => {
    const beforeSecond = parseMarkdown('First paragraph.\n\n')
    const firstParagraphRef = beforeSecond.content[0]

    const duringSecond = parseMarkdown('First paragraph.\n\nSecond', { previous: beforeSecond })
    expect(duringSecond.content[0]).toBe(firstParagraphRef)

    const finalDoc = parseMarkdown('First paragraph.\n\nSecond paragraph.', { previous: duringSecond })
    expect(finalDoc.content.map(item => item.content.type)).toEqual(['paragraph', 'blankLines', 'paragraph'])
    expect(finalDoc.content[0]).toBe(firstParagraphRef)
  })

  test('streaming heading then paragraph keeps heading stable and forms paragraph', () => {
    const prefixes = ['# Ti', '# Title', '# Title\n\n', '# Title\n\nSome', '# Title\n\nSome text here']

    let previous = parseMarkdown(prefixes[0])

    for (const prefix of prefixes.slice(1)) {
      previous = parseMarkdown(prefix, { previous })
    }

    expect(previous.content.map(item => item.content.type)).toEqual(['heading', 'blankLines', 'paragraph'])
  })

  test('streaming fenced code block forms one code block item', () => {
    const prefixes = ['```', '```js\n', '```js\ncon', '```js\nconsole.log(1)\n', '```js\nconsole.log(1)\n```']

    let previous = parseMarkdown(prefixes[0])

    for (const prefix of prefixes.slice(1)) {
      previous = parseMarkdown(prefix, { previous })
      expect(previous.content).toHaveLength(1)
    }

    expect(previous.content[0]?.content.type).toBe('codeBlock')
  })

  test('append does not break existing stable items', () => {
    const sourceBeforeAppend = '# Title\n\nParagraph one.\n\nParagraph'
    const sourceAfterAppend = '# Title\n\nParagraph one.\n\nParagraph two.'

    const previous = parseMarkdown(sourceBeforeAppend)
    const headingRef = previous.content[0]
    const blankAfterHeadingRef = previous.content[1]
    const firstParagraphRef = previous.content[2]
    const blankAfterFirstParagraphRef = previous.content[3]

    const next = parseMarkdown(sourceAfterAppend, { previous })

    expect(next.content.map(item => item.content.type)).toEqual([
      'heading',
      'blankLines',
      'paragraph',
      'blankLines',
      'paragraph',
    ])
    expect(next.content[0]).toBe(headingRef)
    expect(next.content[1]).toBe(blankAfterHeadingRef)
    expect(next.content[2]).toBe(firstParagraphRef)
    expect(next.content[3]).toBe(blankAfterFirstParagraphRef)
  })

  test('non-append middle edit keeps stable prefix before the change', () => {
    const previous = parseMarkdown('Alpha\n\nBravo\n\nCharlie')
    const nextSource = 'Alpha\n\nBravo changed\n\nCharlie'
    const divergeAt = findDivergence(previous.source, nextSource)

    expect(divergeAt).not.toBeNull()

    const stable = findStablePrefixCount(previous, divergeAt!)

    // stableCount=2: Alpha paragraph + blankLines after it.
    // We cut after the last blankLines node before divergeAt,
    // so the Bravo paragraph is reparsed (correct, since it changed).
    expect(stable.stableCount).toBe(2)
    expect(stable.cutPoint).toBe(previous.content[1]?.content.position.end.offset)

    const next = parseMarkdown(nextSource, { previous })
    expect(next.content[0]).toBe(previous.content[0])
    expect(next.content[1]).toBe(previous.content[1])
  })

  test('identical source returns previous document and no divergence', () => {
    const previous = parseMarkdown('Same text')

    expect(findDivergence(previous.source, previous.source)).toBeNull()
    expect(parseMarkdown(previous.source, { previous })).toBe(previous)
  })

  test('incremental table streaming matches fresh parse across paragraph to table transition', () => {
    const prefixes = [
      '| a | b |',
      '| a | b |\n| --- | --- |',
      '| a | b |\n| --- | --- |\n| c | d |',
    ]

    let previous: DocumentNode | undefined
    for (const prefix of prefixes) {
      const incremental = parseMarkdown(prefix, previous ? { previous } : undefined)
      const fresh = parseMarkdown(prefix)

      expectStructurallyEqual(incremental, fresh)
      previous = incremental
    }

    expect(contentTypes(prefixes[0]!)).toEqual(['paragraph'])
    expect(contentTypes(prefixes[1]!)).toEqual(['table'])
    expect(contentTypes(prefixes[2]!)).toEqual(['table'])

    const finalTable = parseMarkdown(prefixes[2]!).content[0]?.content
    expect(finalTable?.type).toBe('table')
    if (finalTable?.type === 'table') {
      expect(finalTable.content).toHaveLength(2)
    }
  })

  test('mid-edit paragraph to heading transition reparses correctly', () => {
    const previous = parseMarkdown('some text')
    const next = parseMarkdown('# some text', { previous })
    const fresh = parseMarkdown('# some text')

    expect(next.content[0]?.content.type).toBe('heading')
    expectStructurallyEqual(next, fresh)
  })

  test('mid-document edit preserves stable title identity and reparses changed paragraph', () => {
    const previous = parseMarkdown('# Title\n\nOld text\n\nFooter')
    const titleRef = previous.content[0]
    const nextSource = '# Title\n\nNew text\n\nFooter'

    const next = parseMarkdown(nextSource, { previous })
    const fresh = parseMarkdown(nextSource)

    expect(next.content[0]).toBe(titleRef)
    expect(next.content[2]).not.toBe(previous.content[2])
    expectStructurallyEqual(next, fresh)
  })

  test('incremental parse matches fresh parse across sampled streaming prefixes', () => {
    const documents = [
      'Plain paragraph text that grows over time.',
      '# Heading\n\nParagraph below it.',
      '| a | b |\n| --- | --- |\n| c | d |',
      '```ts\nconsole.log("hi")\n```',
      '- one\n- two\n- three',
      '# Mixed\n\nParagraph\n\n- item\n- item 2\n\n| a | b |\n| --- | --- |\n| c | d |',
    ]

    for (const source of documents) {
      let previous: DocumentNode | undefined

      for (const prefix of samplePrefixes(source, 10)) {
        const incremental = parseMarkdown(prefix, previous ? { previous } : undefined)
        const fresh = parseMarkdown(prefix)

        expectStructurallyEqual(incremental, fresh)
        previous = incremental
      }
    }
  })

  test('mid-document replacements match fresh parse', () => {
    const cases = [
      {
        before: 'Alpha bravo charlie',
        after: 'Alpha delta charlie',
      },
      {
        before: '| a | b |\n| --- | --- |\n| c | d |',
        after: '| a | b |\n| --- | --- |\n| c | z |',
      },
      {
        before: '# Old title\n\nBody',
        after: '# New title\n\nBody',
      },
      {
        before: '# Title\n\nMiddle paragraph\n\nFooter',
        after: '# Title\n\n\n\nFooter',
      },
      {
        before: '# Title\n\nFooter',
        after: '# Title\n\nInserted paragraph\n\nFooter',
      },
    ]

    for (const { before, after } of cases) {
      const previous = parseMarkdown(before)
      const incremental = parseMarkdown(after, { previous })
      const fresh = parseMarkdown(after)

      expectStructurallyEqual(incremental, fresh)
    }
  })
})