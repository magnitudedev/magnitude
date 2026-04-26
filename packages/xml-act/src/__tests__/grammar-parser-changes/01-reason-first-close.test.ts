/**
 * Category 1: First-close-wins for think
 *
 * The first </magnitude:think> closes the block immediately.
 * No greedy last-match confirmation. After the close, only ws is allowed
 * before the next structural element or yield.
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator, parse, hasEvent, countEvents,
  collectLensChunks, YIELD_USER,
} from './helpers'

const v = () => grammarValidator()
const Y = YIELD_USER

describe('Category 1: first-close-wins for think', () => {
  // =========================================================================
  // Close + whitespace variants → all ACCEPT
  // =========================================================================

  it('01: close + immediate yield (zero ws)', () => {
    const input = `<magnitude:think about="t">x</magnitude:think>${Y}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
    expect(collectLensChunks(parse(input))).toBe('x')
  })

  it('02: close + space + yield', () => {
    const input = `<magnitude:think about="t">x</magnitude:think> ${Y}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  it('03: close + tab + yield', () => {
    v().passes(`<magnitude:think about="t">x</magnitude:think>\t${Y}`)
  })

  it('04: close + newline + yield', () => {
    v().passes(`<magnitude:think about="t">x</magnitude:think>\n${Y}`)
  })

  it('05: close + multiple newlines + yield', () => {
    v().passes(`<magnitude:think about="t">x</magnitude:think>\n\n\n${Y}`)
  })

  it('06: close + mixed ws + yield', () => {
    v().passes(`<magnitude:think about="t">x</magnitude:think> \t\n \t\n${Y}`)
  })

  it('07: close + 10 spaces + yield', () => {
    v().passes(`<magnitude:think about="t">x</magnitude:think>          ${Y}`)
  })

  // =========================================================================
  // Close + chaining to next structural element → all ACCEPT
  // =========================================================================

  it('08: close + immediate message (zero ws)', () => {
    const input = `<magnitude:think about="t">think</magnitude:think><magnitude:message to="u">hi</magnitude:message>\n${Y}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
  })

  it('09: close + newline + message', () => {
    const input = `<magnitude:think about="t">think</magnitude:think>\n<magnitude:message to="u">hi</magnitude:message>\n${Y}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
  })

  it('10: close + immediate invoke', () => {
    const input = `<magnitude:think about="t">think</magnitude:think><magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>\n${Y}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
    expect(hasEvent(parse(input), 'ToolInputReady')).toBe(true)
  })

  it('11: close + newline + another think', () => {
    const input = `<magnitude:think about="a">first</magnitude:think>\n<magnitude:think about="b">second</magnitude:think>\n${Y}`
    v().passes(input)
    expect(countEvents(parse(input), 'LensEnd')).toBe(2)
  })

  it('12: three consecutive thinks', () => {
    const input = `<magnitude:think about="a">1</magnitude:think>\n<magnitude:think about="b">2</magnitude:think>\n<magnitude:think about="c">3</magnitude:think>\n${Y}`
    v().passes(input)
    expect(countEvents(parse(input), 'LensEnd')).toBe(3)
  })

  // =========================================================================
  // Incomplete turns → REJECT
  // =========================================================================

  it('13: close alone (no yield) is incomplete', () => {
    v().rejects(`<magnitude:think about="t">text</magnitude:think>`)
  })

  it('14: close + whitespace only (no yield) is incomplete', () => {
    v().rejects(`<magnitude:think about="t">text</magnitude:think>\n`)
  })

  // =========================================================================
  // Body content edge cases → ACCEPT
  // =========================================================================

  it('15: body with < that is not a close tag', () => {
    const input = `<magnitude:think about="t">foo < bar</magnitude:think>\n${Y}`
    v().passes(input)
    expect(collectLensChunks(parse(input))).toContain('foo < bar')
  })

  it('16: body with HTML-like tags', () => {
    const input = `<magnitude:think about="t"><div>hello</div></magnitude:think>\n${Y}`
    v().passes(input)
    expect(collectLensChunks(parse(input))).toContain('<div>hello</div>')
  })

  it('17: body with partial close prefix </rea', () => {
    const input = `<magnitude:think about="t">text </rea partial</magnitude:think>\n${Y}`
    v().passes(input)
    expect(collectLensChunks(parse(input))).toContain('</rea partial')
  })

  it('18: body with wrong close tag name </reasonX', () => {
    const input = `<magnitude:think about="t">text </reasonX> stuff</magnitude:think>\n${Y}`
    v().passes(input)
    expect(collectLensChunks(parse(input))).toContain('</reasonX>')
  })

  it('19: empty body', () => {
    const input = `<magnitude:think about="t"></magnitude:think>\n${Y}`
    v().passes(input)
    expect(collectLensChunks(parse(input))).toBe('')
  })

  it('20: body with only whitespace', () => {
    const input = `<magnitude:think about="t">  \n  </magnitude:think>\n${Y}`
    v().passes(input)
  })

  // =========================================================================
  // FALSE CLOSE — greedy currently accepts, first-close-wins rejects
  // =========================================================================

  it('21: false close + non-ws char → REJECT', () => {
    // Currently: greedy treats first close as false, body = "x</magnitude:think>more"
    // After: first close ends body, "more" is not ws → reject
    const input = `<magnitude:think about="t">x</magnitude:think>more</magnitude:think>\n${Y}`
    v().rejects(input)
    // Parser still extracts correct body from first close
    expect(collectLensChunks(parse(input))).toBe('x')
  })

  it('22: false close + letter then real close → REJECT', () => {
    const input = `<magnitude:think about="t">The tag </magnitude:think>x is used</magnitude:think>\n${Y}`
    v().rejects(input)
    expect(collectLensChunks(parse(input))).toBe('The tag ')
  })

  it('23: multiple false closes → REJECT', () => {
    const input = `<magnitude:think about="t">a</magnitude:think>b</magnitude:think>c</magnitude:think>\n${Y}`
    v().rejects(input)
    expect(collectLensChunks(parse(input))).toBe('a')
  })

  it('24: user repro — two thinks then prose → REJECT', () => {
    const input =
      `<magnitude:think about="a">first</magnitude:think>\n` +
      `<magnitude:think about="b">second</magnitude:think>\n` +
      `Let me look into this.\n` + Y
    // "Let me look into this." is not whitespace → reject
    v().rejects(input)
  })
})
