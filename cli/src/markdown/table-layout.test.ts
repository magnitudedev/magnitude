import { describe, expect, it } from 'bun:test'
import { computeTableLayoutPlan } from './table-layout'
import type { Span } from './blocks'

const s = (text: string): Span[] => [{ text }]

describe('markdown/table-layout', () => {
  it('basic 2x2 table fits available width', () => {
    const plan = computeTableLayoutPlan({
      headers: [s('h1'), s('h2')],
      rows: [[s('a'), s('b')]],
      availableWidth: 40,
    })
    expect(plan.columnWidths).toHaveLength(2)
    expect(plan.tableWidth).toBeLessThanOrEqual(40)
  })

  it('uses natural widths when space allows', () => {
    const plan = computeTableLayoutPlan({
      headers: [s('short'), s('much longer text')],
      rows: [],
      availableWidth: 80,
      widthMode: 'content',
    })
    expect(plan.columnWidths).toEqual(plan.naturalColumnWidths)
  })

  it('shrinks proportionally when over budget', () => {
    const plan = computeTableLayoutPlan({
      headers: [s('aaaaaaaaaa'), s('bbbbbbbbbbbbbbbbbbbb')],
      rows: [],
      availableWidth: 20,
      fitter: 'proportional',
      wrapMode: 'none',
    })
    expect(plan.columnWidths[0] + plan.columnWidths[1]).toBe(plan.contentBudget)
  })

  it('shrinks with balanced strategy', () => {
    const plan = computeTableLayoutPlan({
      headers: [s('aaaaaaaaaa'), s('bbbbbbbbbbbbbbbbbbbb')],
      rows: [],
      availableWidth: 20,
      fitter: 'balanced',
      wrapMode: 'none',
    })
    expect(plan.columnWidths[0] + plan.columnWidths[1]).toBe(plan.contentBudget)
  })

  it('respects fixed column widths', () => {
    const plan = computeTableLayoutPlan({
      headers: [s('h1'), s('h2')],
      rows: [[s('x'), s('y')]],
      availableWidth: 80,
      fixedColumnWidths: [5, 9],
    })
    expect(plan.columnWidths).toEqual([5, 9])
  })

  it('clamps to min column width', () => {
    const plan = computeTableLayoutPlan({
      headers: [s('a'), s('b')],
      rows: [],
      availableWidth: 4,
      minColumnWidth: 4,
    })
    expect(plan.columnWidths.every((w) => w >= 4)).toBeTrue()
  })

  it('applies alignment to cell lines', () => {
    const plan = computeTableLayoutPlan({
      headers: [s('x'), s('x'), s('x')],
      rows: [[s('a'), s('a'), s('a')]],
      alignments: ['left', 'center', 'right'],
      availableWidth: 40,
      fixedColumnWidths: [6, 6, 6],
      wrapMode: 'none',
    })
    const [l, c, r] = plan.rows[0]!.cells.map((cell) => cell.lines[0]!.spans.map((sp) => sp.text).join(''))
    expect(l).toBe('a     ')
    expect(c).toBe('  a   ')
    expect(r).toBe('     a')
  })

  it('supports wrap modes none/word/char', () => {
    const base = {
      headers: [s('h')],
      rows: [[s('lorem ipsum')]],
      availableWidth: 12,
      fixedColumnWidths: [5],
    }
    const none = computeTableLayoutPlan({ ...base, wrapMode: 'none' })
    const word = computeTableLayoutPlan({ ...base, wrapMode: 'word' })
    const char = computeTableLayoutPlan({ ...base, wrapMode: 'char' })

    expect(none.rows[0]!.height).toBe(1)
    expect(word.rows[0]!.height).toBeGreaterThan(1)
    expect(char.rows[0]!.height).toBeGreaterThan(1)
  })

  it('handles single-column table', () => {
    const plan = computeTableLayoutPlan({
      headers: [s('only')],
      rows: [[s('row')]],
      availableWidth: 20,
    })
    expect(plan.columnWidths).toHaveLength(1)
    expect(plan.rows).toHaveLength(1)
  })

  it('handles empty cells', () => {
    const plan = computeTableLayoutPlan({
      headers: [[], s('h2')],
      rows: [[[{ text: '' }], []]],
      availableWidth: 20,
    })
    expect(plan.headers.cells[0]!.lines[0]!.spans).toBeArray()
    expect(plan.rows[0]!.cells[1]!.lines[0]!.spans).toBeArray()
  })

  it('handles very wide content', () => {
    const plan = computeTableLayoutPlan({
      headers: [s('h')],
      rows: [[s('x'.repeat(200))]],
      availableWidth: 20,
      wrapMode: 'none',
    })
    const line = plan.rows[0]!.cells[0]!.lines[0]!.spans.map((sp) => sp.text).join('')
    expect(line.endsWith('…')).toBeTrue()
  })

  it("widthMode 'full' expands columns", () => {
    const plan = computeTableLayoutPlan({
      headers: [s('a'), s('b')],
      rows: [],
      availableWidth: 40,
      widthMode: 'full',
    })
    expect(plan.columnWidths.reduce((a, b) => a + b, 0)).toBe(plan.contentBudget)
  })

  it('border options change width calculation', () => {
    const withBorders = computeTableLayoutPlan({
      headers: [s('a')],
      rows: [],
      availableWidth: 20,
      borders: { outer: true, inner: true },
    })
    const withoutBorders = computeTableLayoutPlan({
      headers: [s('a')],
      rows: [],
      availableWidth: 20,
      borders: { outer: false, inner: false },
    })
    expect(withBorders.tableWidth).toBeGreaterThan(withoutBorders.tableWidth)
  })
})
