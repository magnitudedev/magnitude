/**
 * Category 6: Edge cases and interactions
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator, parse, hasEvent, getToolInput,
  collectLensChunks, collectMessageChunks, countEvents, YIELD_USER,
} from './helpers'

const v = () => grammarValidator()
const Y = YIELD_USER

describe('Category 6: edge cases', () => {
  // =========================================================================
  // Wrong-case and non-magnitude close tags in bodies
  // =========================================================================

  it('01: wrong-case </Magnitude:think> is body content', () => {
    const input = `<magnitude:think about="t">text </Magnitude:think> more</magnitude:think>\n${Y}`
    v().passes(input)
    const chunks = collectLensChunks(parse(input))
    expect(chunks).toContain('</Magnitude:think>')
    expect(chunks).toContain('more')
  })

  it('02: </magnitude:REASON> (wrong case) in think body → REJECT (pre-existing)', () => {
    // The BUC stops at </magnitude: prefix, then expects "think>" but sees "REASON>"
    v().rejects(`<magnitude:think about="t">text </magnitude:REASON> more</magnitude:think>\n${Y}`)
  })

  it('03: non-magnitude close tag in body', () => {
    const input = `<magnitude:think about="t">text </div> more</magnitude:think>\n${Y}`
    v().passes(input)
    expect(collectLensChunks(parse(input))).toContain('</div>')
  })

  // =========================================================================
  // Whitespace-only between elements (no prose allowed)
  // =========================================================================

  it('04: prose text between think and yield → REJECT', () => {
    v().rejects(`<magnitude:think about="t">x</magnitude:think>\nSome text\n${Y}`)
  })

  it('05: prose text between think and message → REJECT', () => {
    v().rejects(`<magnitude:think about="t">x</magnitude:think>\nSome text\n<magnitude:message to="u">hi</magnitude:message>\n${Y}`)
  })

  it('06: prose text between message and yield → REJECT', () => {
    v().rejects(`<magnitude:message to="u">hi</magnitude:message>\nSome text\n${Y}`)
  })

  it('07: prose text at start of turn → REJECT', () => {
    v().rejects(`Hello world\n${Y}`)
  })

  it('08: prose text between two invokes → REJECT', () => {
    v().rejects(
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>\n` +
      `Some text\n` +
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">pwd</magnitude:parameter>\n</magnitude:invoke>\n${Y}`
    )
  })

  // =========================================================================
  // Multiple structural elements
  // =========================================================================

  it('09: think + message + invoke + yield', () => {
    const input =
      `<magnitude:think about="t">think</magnitude:think>\n` +
      `<magnitude:message to="u">hi</magnitude:message>\n` +
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>\n${Y}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
    expect(hasEvent(parse(input), 'ToolInputReady')).toBe(true)
  })

  it('10: two invokes in sequence', () => {
    const input =
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>\n` +
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">pwd</magnitude:parameter>\n</magnitude:invoke>\n${Y}`
    v().passes(input)
    expect(countEvents(parse(input), 'ToolInputReady')).toBe(2)
  })

  it('11: invoke with alias close then another invoke', () => {
    const input =
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:shell>\n` +
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">pwd</magnitude:parameter>\n</magnitude:invoke>\n${Y}`
    v().passes(input)
    expect(countEvents(parse(input), 'ToolInputReady')).toBe(2)
  })

  // =========================================================================
  // Yield-only turns
  // =========================================================================

  it('12: yield only (no structural elements)', () => {
    v().passes(Y)
  })

  it('13: whitespace + yield', () => {
    v().passes(`\n${Y}`)
  })

  it('14: lots of whitespace + yield', () => {
    v().passes(`\n\n\n   \t\n${Y}`)
  })
})
