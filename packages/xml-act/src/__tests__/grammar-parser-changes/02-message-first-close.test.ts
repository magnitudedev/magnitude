/**
 * Category 2: First-close-wins for message
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator, parse, hasEvent, collectMessageChunks, YIELD_USER,
} from './helpers'

const v = () => grammarValidator()
const Y = YIELD_USER

describe('Category 2: first-close-wins for message', () => {
  // =========================================================================
  // Close + whitespace variants → all ACCEPT
  // =========================================================================

  it('01: close + immediate yield (zero ws)', () => {
    const input = `<magnitude:message to="u">hello</magnitude:message>${Y}`
    v().passes(input)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
    expect(collectMessageChunks(parse(input))).toBe('hello')
  })

  it('02: close + space + yield', () => {
    v().passes(`<magnitude:message to="u">hi</magnitude:message> ${Y}`)
  })

  it('03: close + newline + yield', () => {
    v().passes(`<magnitude:message to="u">hi</magnitude:message>\n${Y}`)
  })

  it('04: close + multiple newlines + yield', () => {
    v().passes(`<magnitude:message to="u">hi</magnitude:message>\n\n\n${Y}`)
  })

  it('05: close + mixed ws + yield', () => {
    v().passes(`<magnitude:message to="u">hi</magnitude:message> \t\n${Y}`)
  })

  // =========================================================================
  // Close + chaining → ACCEPT
  // =========================================================================

  it('06: close + immediate invoke', () => {
    const input = `<magnitude:message to="u">hi</magnitude:message><magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>\n${Y}`
    v().passes(input)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
    expect(hasEvent(parse(input), 'ToolInputReady')).toBe(true)
  })

  it('07: close + newline + invoke', () => {
    const input = `<magnitude:message to="u">hi</magnitude:message>\n<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>\n${Y}`
    v().passes(input)
  })

  it('08: think then message then yield', () => {
    const input = `<magnitude:think about="t">think</magnitude:think>\n<magnitude:message to="u">hi</magnitude:message>\n${Y}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
  })

  it('09: think then message then invoke then yield', () => {
    const input = `<magnitude:think about="t">think</magnitude:think>\n<magnitude:message to="u">hi</magnitude:message>\n<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>\n${Y}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
    expect(hasEvent(parse(input), 'ToolInputReady')).toBe(true)
  })

  // =========================================================================
  // Body content edge cases → ACCEPT
  // =========================================================================

  it('10: body with < that is not a close tag', () => {
    const input = `<magnitude:message to="u">use < for less-than</magnitude:message>\n${Y}`
    v().passes(input)
    expect(collectMessageChunks(parse(input))).toContain('use < for less-than')
  })

  it('11: empty body', () => {
    const input = `<magnitude:message to="u"></magnitude:message>\n${Y}`
    v().passes(input)
    expect(collectMessageChunks(parse(input))).toBe('')
  })

  // =========================================================================
  // FALSE CLOSE → REJECT
  // =========================================================================

  it('12: false close + non-ws char → REJECT', () => {
    const input = `<magnitude:message to="u">text</magnitude:message>more</magnitude:message>\n${Y}`
    v().rejects(input)
    expect(collectMessageChunks(parse(input))).toBe('text')
  })

  it('13: false close in message after think → REJECT', () => {
    const input = `<magnitude:think about="t">r</magnitude:think>\n<magnitude:message to="u">text</magnitude:message>extra</magnitude:message>\n${Y}`
    v().rejects(input)
  })
})
