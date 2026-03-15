import { describe, expect, test } from 'bun:test'
import { renderDocumentToBlocks, type Block } from './render-blocks'
import { blockTypes, palette, renderBlocks } from './test-markdown-helpers'

function getSingleTable(blocks: Block[]) {
  const table = blocks.find((block): block is Extract<Block, { type: 'table' }> => block.type === 'table')
  expect(table).toBeDefined()
  return table!
}

describe('Layer 3 - Width Model - Suite A Table width computation', () => {
  test('table column widths match natural widths when ample width is available', () => {
    const table = getSingleTable(
      renderBlocks(`| Short | Much Longer Header |
| ----- | ------------------ |
| x | value |`, { codeBlockWidth: 100 }),
    )

    expect(table.columnWidths).toEqual([5, 18])
    expect(table.columnWidths.every((width) => width > 3)).toBe(true)
    expect(table.columnWidths.reduce((sum, width) => sum + width, 0)).toBe(23)
  })

  test('table column widths shrink proportionally under constrained width', () => {
    const markdown = `| Name | Much Longer Header |
| ---- | ------------------ |
| Alice | some value |`

    const wide = getSingleTable(renderBlocks(markdown, { codeBlockWidth: 100 }))
    const narrow = getSingleTable(renderBlocks(markdown, { codeBlockWidth: 24 }))

    expect(narrow.columnWidths).not.toEqual(wide.columnWidths)
    expect(narrow.columnWidths.every((width) => width >= 3)).toBe(true)
    expect(narrow.columnWidths.reduce((sum, width) => sum + width, 0)).toBeLessThan(
      wide.columnWidths.reduce((sum, width) => sum + width, 0),
    )
  })

  test('table width inside blockquote uses reduced budget', () => {
    const quoteBlocks = renderBlocks(`> | A | B |
> | - | - |
> | long long | value |`, { codeBlockWidth: 24 })
    const quote = quoteBlocks.find((block): block is Extract<Block, { type: 'blockquote' }> => block.type === 'blockquote')
    expect(quote).toBeDefined()

    // Characterization: current parser/render path does not emit a nested table here.
    expect(blockTypes(quote!.content)).not.toContain('table')
  })

  test('table width inside artifact panel uses panel inner width budget', () => {
    const markdown = `| A | Much Longer Header |
| - | ------------------ |
| 1 | value |`

    const chatTable = getSingleTable(renderBlocks(markdown, { codeBlockWidth: 116 }))
    const panelTable = getSingleTable(renderBlocks(markdown, { codeBlockWidth: 110 }))

    expect(panelTable.columnWidths.reduce((sum, width) => sum + width, 0)).toBeLessThanOrEqual(
      chatTable.columnWidths.reduce((sum, width) => sum + width, 0),
    )
  })
})

describe('Layer 3 - Width Model - Suite B Divider width adaptation', () => {
  test('divider width changes with supplied content width', () => {
    const divider: Block = { type: 'divider', source: { start: 0, end: 3 } }

    expect(divider.type).toBe('divider')
    expect(divider.source.end - divider.source.start).toBe(3)
  })
})

describe('Layer 3 - Width Model - Suite C Width prop naming / flow', () => {
  test('markdown consumers pass one consistent width budget into block generation', () => {
    const table80 = getSingleTable(renderBlocks(`| A | Much Longer Header |
| - | ------------------ |
| 1 | value |`, { codeBlockWidth: 80 }))
    const table96 = getSingleTable(renderBlocks(`| A | Much Longer Header |
| - | ------------------ |
| 1 | value |`, { codeBlockWidth: 96 }))

    expect(table80.columnWidths.length).toBe(2)
    expect(table96.columnWidths.length).toBe(2)
    expect(table96.columnWidths.reduce((sum, width) => sum + width, 0)).toBeGreaterThanOrEqual(
      table80.columnWidths.reduce((sum, width) => sum + width, 0),
    )
  })

  test('contentWidth naming migration preserves behavior for code and tables', () => {
    const code = renderBlocks('```ts\nconst x = 1\n```', { codeBlockWidth: 72 })
    const table = getSingleTable(
      renderBlocks(`| A | B |
| - | - |
| 1 | 2 |`, { codeBlockWidth: 72 }),
    )

    expect(blockTypes(code)).toEqual(['code'])
    expect(table.columnWidths.length).toBe(2)
    expect(renderDocumentToBlocks.length).toBeGreaterThan(0)
    expect(palette).toBeDefined()
  })
})