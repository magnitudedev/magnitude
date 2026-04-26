/**
 * Category 1: First-close-wins for reason
 *
 * The first </magnitude:reason> closes the block immediately.
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

describe('Category 1: first-close-wins for reason', () => {
  // =========================================================================
  // Close + whitespace variants → all ACCEPT
  // =========================================================================

  it('01: close + immediate yield (zero ws)', () => {
    const input = `<magnitude:reason about="t">x</magnitude:reason>${Y}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
    expect(collectLensChunks(parse(input))).toBe('x')
  })

  it('02: close + space + yield', () => {
    const input = `<magnitude:reason about="t">x</magnitude:reason> ${Y}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  it('03: close + tab + yield', () => {
    v().passes(`<magnitude:reason about="t">x</magnitude:reason>\t${Y}`)
  })

  it('04: close + newline + yield', () => {
    v().passes(`<magnitude:reason about="t">x</magnitude:reason>\n${Y}`)
  })

  it('05: close + multiple newlines + yield', () => {
    v().passes(`<magnitude:reason about="t">x</magnitude:reason>\n\n\n${Y}`)
  })

  it('06: close + mixed ws + yield', () => {
    v().passes(`<magnitude:reason about="t">x</magnitude:reason> \t\n \t\n${Y}`)
  })

  it('07: close + 10 spaces + yield', () => {
    v().passes(`<magnitude:reason about="t">x</magnitude:reason>          ${Y}`)
  })

  // =========================================================================
  // Close + chaining to next structural element → all ACCEPT
  // =========================================================================

  it('08: close + immediate message (zero ws)', () => {
    const input = `<magnitude:reason about="t">think</magnitude:reason><magnitude:message to="u">hi</magnitude:message>\n${Y}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
  })

  it('09: close + newline + message', () => {
    const input = `<magnitude:reason about="t">think</magnitude:reason>\n<magnitude:message to="u">hi</magnitude:message>\n${Y}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
  })

  it('10: close + immediate invoke', () => {
    const input = `<magnitude:reason about="t">think</magnitude:reason><magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>\n${Y}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
    expect(hasEvent(parse(input), 'ToolInputReady')).toBe(true)
  })

  it('11: close + newline + another reason', () => {
    const input = `<magnitude:reason about="a">first</magnitude:reason>\n<magnitude:reason about="b">second</magnitude:reason>\n${Y}`
    v().passes(input)
    expect(countEvents(parse(input), 'LensEnd')).toBe(2)
  })

  it('12: three consecutive reasons', () => {
    const input = `<magnitude:reason about="a">1</magnitude:reason>\n<magnitude:reason about="b">2</magnitude:reason>\n<magnitude:reason about="c">3</magnitude:reason>\n${Y}`
    v().passes(input)
    expect(countEvents(parse(input), 'LensEnd')).toBe(3)
  })

  // =========================================================================
  // Incomplete turns → REJECT
  // =========================================================================

  it('13: close alone (no yield) is incomplete', () => {
    v().rejects(`<magnitude:reason about="t">text</magnitude:reason>`)
  })

  it('14: close + whitespace only (no yield) is incomplete', () => {
    v().rejects(`<magnitude:reason about="t">text</magnitude:reason>\n`)
  })

  // =========================================================================
  // Body content edge cases → ACCEPT
  // =========================================================================

  it('15: body with < that is not a close tag', () => {
    const input = `<magnitude:reason about="t">foo < bar</magnitude:reason>\n${Y}`
    v().passes(input)
    expect(collectLensChunks(parse(input))).toContain('foo < bar')
  })

  it('16: body with HTML-like tags', () => {
    const input = `<magnitude:reason about="t"><div>hello</div></magnitude:reason>\n${Y}`
    v().passes(input)
    expect(collectLensChunks(parse(input))).toContain('<div>hello</div>')
  })

  it('17: body with partial close prefix </rea', () => {
    const input = `<magnitude:reason about="t">text </rea partial</magnitude:reason>\n${Y}`
    v().passes(input)
    expect(collectLensChunks(parse(input))).toContain('</rea partial')
  })

  it('18: body with wrong close tag name </reasonX', () => {
    const input = `<magnitude:reason about="t">text </reasonX> stuff</magnitude:reason>\n${Y}`
    v().passes(input)
    expect(collectLensChunks(parse(input))).toContain('</reasonX>')
  })

  it('19: empty body', () => {
    const input = `<magnitude:reason about="t"></magnitude:reason>\n${Y}`
    v().passes(input)
    expect(collectLensChunks(parse(input))).toBe('')
  })

  it('20: body with only whitespace', () => {
    const input = `<magnitude:reason about="t">  \n  </magnitude:reason>\n${Y}`
    v().passes(input)
  })

  // =========================================================================
  // FALSE CLOSE — greedy currently accepts, first-close-wins rejects
  // =========================================================================

  it('21: false close + non-ws char → REJECT', () => {
    // Currently: greedy treats first close as false, body = "x</magnitude:reason>more"
    // After: first close ends body, "more" is not ws → reject
    const input = `<magnitude:reason about="t">x</magnitude:reason>more</magnitude:reason>\n${Y}`
    v().rejects(input)
    // Parser still extracts correct body from first close
    expect(collectLensChunks(parse(input))).toBe('x')
  })

  it('22: false close + letter then real close → REJECT', () => {
    const input = `<magnitude:reason about="t">The tag </magnitude:reason>x is used</magnitude:reason>\n${Y}`
    v().rejects(input)
    expect(collectLensChunks(parse(input))).toBe('The tag ')
  })

  it('23: multiple false closes → REJECT', () => {
    const input = `<magnitude:reason about="t">a</magnitude:reason>b</magnitude:reason>c</magnitude:reason>\n${Y}`
    v().rejects(input)
    expect(collectLensChunks(parse(input))).toBe('a')
  })

  it('24: user repro — two reasons then prose → REJECT', () => {
    const input =
      `<magnitude:reason about="a">first</magnitude:reason>\n` +
      `<magnitude:reason about="b">second</magnitude:reason>\n` +
      `Let me look into this.\n` + Y
    // "Let me look into this." is not whitespace → reject
    v().rejects(input)
  })
})
