import { describe, it, expect } from 'bun:test'
import { parseEditOps } from '../line-edit'

describe('parseEditOps', () => {
  it('basic remove', () => {
    const ops = parseEditOps('<remove from=5 to=10 />')
    expect(ops).toEqual([{ type: 'remove', from: 5, to: 10 }])
  })

  it('quoted attributes on remove', () => {
    const ops = parseEditOps('<remove from="5" to="10" />')
    expect(ops).toEqual([{ type: 'remove', from: 5, to: 10 }])
  })

  it('basic replace', () => {
    const ops = parseEditOps('<replace from=5 to=10>\nnew content\n</replace>')
    expect(ops).toEqual([{ type: 'replace', from: 5, to: 10, content: 'new content' }])
  })

  it('basic insert', () => {
    const ops = parseEditOps('<insert after=5>\nnew content\n</insert>')
    expect(ops).toEqual([{ type: 'insert', after: 5, content: 'new content' }])
  })

  it('quoted attributes on replace', () => {
    const ops = parseEditOps('<replace from="5" to="10">\nstuff\n</replace>')
    expect(ops).toEqual([{ type: 'replace', from: 5, to: 10, content: 'stuff' }])
  })

  it('quoted attributes on insert', () => {
    const ops = parseEditOps('<insert after="5">\nstuff\n</insert>')
    expect(ops).toEqual([{ type: 'insert', after: 5, content: 'stuff' }])
  })

  it('content containing XML-like strings', () => {
    const ops = parseEditOps('<replace from=1 to=2>\nconst x = `<agent id="foo" />`\n</replace>')
    expect(ops).toEqual([{ type: 'replace', from: 1, to: 2, content: 'const x = `<agent id="foo" />`' }])
  })

  it('nesting: content containing same tag name', () => {
    const ops = parseEditOps('<replace from=1 to=5>\n<replace from=3 to=4>inner</replace>\n</replace>')
    expect(ops).toEqual([{ type: 'replace', from: 1, to: 5, content: '<replace from=3 to=4>inner</replace>' }])
  })

  it('multiple operations', () => {
    const input = '<remove from=1 to=2 />\n<insert after=5>\nhello\n</insert>\n<replace from=10 to=12>\nworld\n</replace>'
    const ops = parseEditOps(input)
    expect(ops).toHaveLength(3)
    expect(ops[0]).toEqual({ type: 'remove', from: 1, to: 2 })
    expect(ops[1]).toEqual({ type: 'insert', after: 5, content: 'hello' })
    expect(ops[2]).toEqual({ type: 'replace', from: 10, to: 12, content: 'world' })
  })

  it('mixed quoted and unquoted attributes', () => {
    const ops = parseEditOps('<replace from="5" to=10>\ncontent\n</replace>')
    expect(ops).toEqual([{ type: 'replace', from: 5, to: 10, content: 'content' }])
  })

  it('self-closing with no space before />', () => {
    const ops = parseEditOps('<remove from=5 to=10/>')
    expect(ops).toEqual([{ type: 'remove', from: 5, to: 10 }])
  })

  it('whitespace variations in tags', () => {
    const ops = parseEditOps('<remove   from=5   to=10   />')
    expect(ops).toEqual([{ type: 'remove', from: 5, to: 10 }])
  })

  it('empty response returns empty array', () => {
    expect(parseEditOps('')).toEqual([])
    expect(parseEditOps('no ops here')).toEqual([])
  })
})