import { describe, test, expect } from 'bun:test'
import { formatNumberedLines, parseEditOps, applyOps } from '../line-edit'

describe('formatNumberedLines', () => {
  test('formats lines with plain numbers', () => {
    const result = formatNumberedLines('hello\nworld')
    const lines = result.split('\n')
    expect(lines).toHaveLength(2)
    expect(lines[0]).toBe('1|hello')
    expect(lines[1]).toBe('2|world')
  })

  test('handles empty lines', () => {
    const result = formatNumberedLines('hello\n\nworld')
    const lines = result.split('\n')
    expect(lines).toHaveLength(3)
    expect(lines[1]).toBe('2|')
  })

  test('handles single line', () => {
    expect(formatNumberedLines('only')).toBe('1|only')
  })
})

describe('parseEditOps', () => {
  test('parses replace op', () => {
    const ops = parseEditOps('<replace from=2 to=4>\nnew content\n</replace>')
    expect(ops).toHaveLength(1)
    expect(ops[0]).toEqual({ type: 'replace', from: 2, to: 4, content: 'new content' })
  })

  test('parses remove op', () => {
    const ops = parseEditOps('<remove from=3 to=5 />')
    expect(ops).toHaveLength(1)
    expect(ops[0]).toEqual({ type: 'remove', from: 3, to: 5 })
  })

  test('parses remove op without space before slash', () => {
    const ops = parseEditOps('<remove from=3 to=5/>')
    expect(ops).toHaveLength(1)
    expect(ops[0]).toEqual({ type: 'remove', from: 3, to: 5 })
  })

  test('parses insert op', () => {
    const ops = parseEditOps('<insert after=2>\nnew line here\n</insert>')
    expect(ops).toHaveLength(1)
    expect(ops[0]).toEqual({ type: 'insert', after: 2, content: 'new line here' })
  })

  test('parses insert after=0 (top of file)', () => {
    const ops = parseEditOps('<insert after=0>\nfirst line\n</insert>')
    expect(ops).toHaveLength(1)
    expect(ops[0]).toEqual({ type: 'insert', after: 0, content: 'first line' })
  })

  test('parses multiple ops', () => {
    const response = '<remove from=1 to=1 />\n<replace from=5 to=6>\nnew stuff\n</replace>\n<insert after=10>\nextra\n</insert>'
    const ops = parseEditOps(response)
    expect(ops).toHaveLength(3)
  })

  test('parses multi-line replace content', () => {
    const ops = parseEditOps('<replace from=1 to=2>\nline a\nline b\nline c\n</replace>')
    expect(ops[0]).toEqual({ type: 'replace', from: 1, to: 2, content: 'line a\nline b\nline c' })
  })
})

describe('applyOps', () => {
  const sample = 'line one\nline two\nline three\nline four\nline five'

  test('replace range', () => {
    const result = applyOps(sample, [{ type: 'replace', from: 2, to: 4, content: 'new middle' }])
    const lines = result.content.split('\n')
    expect(lines).toHaveLength(3)
    expect(lines[0]).toBe('line one')
    expect(lines[1]).toBe('new middle')
    expect(lines[2]).toBe('line five')
  })

  test('replace single line', () => {
    const result = applyOps(sample, [{ type: 'replace', from: 2, to: 2, content: 'replaced two' }])
    const lines = result.content.split('\n')
    expect(lines).toHaveLength(5)
    expect(lines[1]).toBe('replaced two')
  })

  test('remove range', () => {
    const result = applyOps(sample, [{ type: 'remove', from: 2, to: 4 }])
    const lines = result.content.split('\n')
    expect(lines).toHaveLength(2)
    expect(lines[0]).toBe('line one')
    expect(lines[1]).toBe('line five')
  })

  test('insert after line', () => {
    const result = applyOps(sample, [{ type: 'insert', after: 2, content: 'inserted line' }])
    const lines = result.content.split('\n')
    expect(lines).toHaveLength(6)
    expect(lines[1]).toBe('line two')
    expect(lines[2]).toBe('inserted line')
    expect(lines[3]).toBe('line three')
  })

  test('insert at top of file (after=0)', () => {
    const result = applyOps(sample, [{ type: 'insert', after: 0, content: 'header' }])
    const lines = result.content.split('\n')
    expect(lines).toHaveLength(6)
    expect(lines[0]).toBe('header')
    expect(lines[1]).toBe('line one')
  })

  test('multiple ops applied bottom-up', () => {
    const result = applyOps(sample, [
      { type: 'replace', from: 1, to: 1, content: 'new first' },
      { type: 'replace', from: 5, to: 5, content: 'new fifth' }
    ])
    const lines = result.content.split('\n')
    expect(lines[0]).toBe('new first')
    expect(lines[4]).toBe('new fifth')
    expect(lines).toHaveLength(5)
  })

  test('mixed ops bottom-up', () => {
    const result = applyOps(sample, [
      { type: 'remove', from: 4, to: 5 },
      { type: 'insert', after: 1, content: 'inserted' }
    ])
    const lines = result.content.split('\n')
    expect(lines).toHaveLength(4)
    expect(lines[0]).toBe('line one')
    expect(lines[1]).toBe('inserted')
    expect(lines[2]).toBe('line two')
    expect(lines[3]).toBe('line three')
  })

  test('out of range throws', () => {
    expect(() => applyOps(sample, [{ type: 'replace', from: 0, to: 1, content: 'x' }])).toThrow(/out of range/)
    expect(() => applyOps(sample, [{ type: 'replace', from: 1, to: 10, content: 'x' }])).toThrow(/out of range/)
  })

  test('diffs are returned', () => {
    const result = applyOps(sample, [{ type: 'replace', from: 2, to: 3, content: 'a\nb' }])
    expect(result.diffs).toHaveLength(1)
    expect(result.diffs[0].startLine).toBe(2)
    expect(result.diffs[0].removedLines).toEqual(['line two', 'line three'])
    expect(result.diffs[0].addedLines).toEqual(['a', 'b'])
  })
})
