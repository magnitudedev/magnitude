/**
 * Category 1: Top-Level Close Tag Confirmation
 *
 * All bodies use recursive greedy last-match.
 * Top-level confirmation: </tagname> + any whitespace + < (next structural tag).
 * Non-whitespace after close tag rejects it to content.
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator, parse, hasEvent, countEvents, collectLensChunks,
  collectMessageChunks, YIELD,
} from './helpers'

const v = () => grammarValidator()

describe('top-level close tag confirmation', () => {
  // ---- Think: confirmed ----

  it('01: </magnitude:think>< confirms (immediate)', () => {
    const input = `<magnitude:think about="t">\nx\n</magnitude:think><${YIELD.slice(1)}`
    v().passes(input)
    const events = parse(input)
    expect(hasEvent(events, 'LensEnd')).toBe(true)
  })

  it('02: </magnitude:think> + 1 space + < confirms', () => {
    const input = `<magnitude:think about="t">\nx\n</magnitude:think> <${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  it('03: </magnitude:think> + 10 spaces + < confirms (no bound)', () => {
    const input = `<magnitude:think about="t">\nx\n</magnitude:think>          <${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  it('04: </magnitude:think> + tab + < confirms', () => {
    const input = `<magnitude:think about="t">\nx\n</magnitude:think>\t<${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  it('05: </magnitude:think> + \\n + < confirms', () => {
    const input = `<magnitude:think about="t">\nx\n</magnitude:think>\n<${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  it('06: </magnitude:think> + multiple \\n + < confirms', () => {
    const input = `<magnitude:think about="t">\nx\n</magnitude:think>\n\n\n<${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  it('07: </magnitude:think> + mixed ws + < confirms', () => {
    const input = `<magnitude:think about="t">\nx\n</magnitude:think> \t\n \t\n<${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  // ---- Think: rejected (false close tags) ----

  it('08: </magnitude:think> + non-ws char rejects under first-close-wins', () => {
    const input = `<magnitude:think about="t">\n</magnitude:think>x more\n</magnitude:think><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('09: </magnitude:think> + \\n without ever seeing < — incomplete', () => {
    const input = `<magnitude:think about="t">\nx\n</magnitude:think>\n`
    v().rejects(input)
  })

  it('10: false </magnitude:think> then real close rejects', () => {
    const input = `<magnitude:think about="t">\nThe tag </magnitude:think>x is used\n</magnitude:think><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('11: multiple false </magnitude:think> before real one rejects', () => {
    const input = `<magnitude:think about="t">\n</magnitude:think>a\n</magnitude:think>b\n</magnitude:think><${YIELD.slice(1)}`
    v().rejects(input)
  })

  // ---- Think: chaining ----

  it('12: </magnitude:think> confirmed by <magnitude:think (another lens)', () => {
    const input = `<magnitude:think about="a">\nx\n</magnitude:think><magnitude:think about="b">\ny\n</magnitude:think><${YIELD.slice(1)}`
    v().passes(input)
    expect(countEvents(parse(input), 'LensEnd')).toBe(2)
  })

  it('13: </magnitude:think> confirmed by <invoke', () => {
    const input = `<magnitude:think about="a">\nx\n</magnitude:think><magnitude:invoke tool="tree">\n</magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  it('14: </magnitude:think> confirmed by <magnitude:message', () => {
    const input = `<magnitude:think about="a">\nx\n</magnitude:think><magnitude:message to="u">\nh\n</magnitude:message><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
  })

  // ---- Think: content edge cases ----

  it('15: content with < that is not a close tag', () => {
    const input = `<magnitude:think about="t">\nfoo < bar\n</magnitude:think><${YIELD.slice(1)}`
    v().passes(input)
    const text = collectLensChunks(parse(input))
    expect(text).toContain('foo < bar')
  })

  it('16: content with HTML-like tags', () => {
    const input = `<magnitude:think about="t">\n<div>hello</div>\n</magnitude:think><${YIELD.slice(1)}`
    v().passes(input)
    const text = collectLensChunks(parse(input))
    expect(text).toContain('<div>hello</div>')
  })

  it('17: content with partial close tag </rea', () => {
    const input = `<magnitude:think about="t">\n</rea partial\n</magnitude:think><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  it('18: content with </reasonX (wrong close tag name)', () => {
    const input = `<magnitude:think about="t">\n</reasonX> stuff\n</magnitude:think><${YIELD.slice(1)}`
    v().passes(input)
    const text = collectLensChunks(parse(input))
    expect(text).toContain('</reasonX>')
  })

  // ---- Message: confirmed ----

  it('19: </magnitude:message>< confirms (yield)', () => {
    const input = `<magnitude:message to="u">\nh\n</magnitude:message><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
  })

  it('20: </magnitude:message> + \\n + < confirms', () => {
    const input = `<magnitude:message to="u">\nh\n</magnitude:message>\n<${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
  })

  it('21: </magnitude:message> + spaces + < confirms', () => {
    const input = `<magnitude:message to="u">\nh\n</magnitude:message>  <${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
  })

  it('22: </magnitude:message> + 20 spaces + < confirms (no bound)', () => {
    const input = `<magnitude:message to="u">\nh\n</magnitude:message>                    <${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
  })

  // ---- Message: rejected ----

  it('23: false </magnitude:message> in content rejects', () => {
    const input = `<magnitude:message to="u">\nuse </magnitude:message>x to close\n</magnitude:message><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('24: </magnitude:message> + non-ws rejects', () => {
    const input = `<magnitude:message to="u">\n</magnitude:message>foo\n</magnitude:message><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('25: </magnitude:message> + \\n alone (no <) does not terminate', () => {
    const input = `<magnitude:message to="u">\nh\n</magnitude:message>\n`
    v().rejects(input)
  })

  // ---- Message: chaining ----

  it('26: </magnitude:message> confirmed by <invoke', () => {
    const input = `<magnitude:message to="u">\nh\n</magnitude:message><magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
    expect(hasEvent(parse(input), 'ToolInputReady')).toBe(true)
  })

  it('27: </magnitude:message> confirmed by <magnitude:message (another message)', () => {
    const input = `<magnitude:message to="u">\na\n</magnitude:message><magnitude:message to="v">\nb\n</magnitude:message><${YIELD.slice(1)}`
    v().passes(input)
    expect(countEvents(parse(input), 'MessageEnd')).toBe(2)
  })
})
