/**
 * Category 3: First-close-wins for parameter
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator, parse, getToolInput, hasEvent, YIELD_USER,
} from './helpers'

const v = () => grammarValidator()
const Y = YIELD_USER
const IO = '<magnitude:invoke tool="shell">\n'
const IC = '</magnitude:invoke>'

describe('Category 3: first-close-wins for parameter', () => {
  // =========================================================================
  // Basic parameter close → ACCEPT
  // =========================================================================

  it('01: param + ws + invoke close', () => {
    const input = `${IO}<magnitude:parameter name="command">ls</magnitude:parameter>\n${IC}\n${Y}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('ls')
  })

  it('02: param + immediate invoke close (zero ws)', () => {
    const input = `${IO}<magnitude:parameter name="command">ls</magnitude:parameter>${IC}\n${Y}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('ls')
  })

  it('03: empty param body', () => {
    const input = `${IO}<magnitude:parameter name="command"></magnitude:parameter>\n${IC}\n${Y}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('')
  })

  it('04: param body with < that is not a close', () => {
    const input = `${IO}<magnitude:parameter name="command">echo "x < y"</magnitude:parameter>\n${IC}\n${Y}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toContain('x < y')
  })

  it('05: param body with HTML tags', () => {
    const input = `${IO}<magnitude:parameter name="command">echo "<div>hi</div>"</magnitude:parameter>\n${IC}\n${Y}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toContain('<div>hi</div>')
  })

  // =========================================================================
  // Multi-param tool
  // =========================================================================

  it('06: edit tool with three params', () => {
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f.ts</magnitude:parameter>\n<magnitude:parameter name="old">foo</magnitude:parameter>\n<magnitude:parameter name="new">bar</magnitude:parameter>\n</magnitude:invoke>\n${Y}`
    v().passes(input)
    const tool = getToolInput(parse(input))
    expect(tool?.path).toBe('f.ts')
    expect(tool?.old).toBe('foo')
    expect(tool?.new).toBe('bar')
  })

  // =========================================================================
  // FALSE CLOSE → REJECT
  // =========================================================================

  it('07: false close in param body → REJECT', () => {
    const input = `${IO}<magnitude:parameter name="command">echo </magnitude:parameter>more</magnitude:parameter>\n${IC}\n${Y}`
    v().rejects(input)
    expect(getToolInput(parse(input))?.command).toBe('echo ')
  })

  it('08: false close in non-last param of edit → REJECT', () => {
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:parameter>extra</magnitude:parameter>\n<magnitude:parameter name="old">x</magnitude:parameter>\n<magnitude:parameter name="new">y</magnitude:parameter>\n</magnitude:invoke>\n${Y}`
    v().rejects(input)
  })

  it('09: false close in last param of edit → REJECT', () => {
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:parameter>\n<magnitude:parameter name="old">x</magnitude:parameter>\n<magnitude:parameter name="new">y</magnitude:parameter>z</magnitude:parameter>\n</magnitude:invoke>\n${Y}`
    v().rejects(input)
  })
})
