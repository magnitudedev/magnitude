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
  // ---- Reason: confirmed ----

  it('01: </magnitude:reason>< confirms (immediate)', () => {
    const input = `<magnitude:reason about="t">\nx\n</magnitude:reason><${YIELD.slice(1)}`
    v().passes(input)
    const events = parse(input)
    expect(hasEvent(events, 'LensEnd')).toBe(true)
  })

  it('02: </magnitude:reason> + 1 space + < confirms', () => {
    const input = `<magnitude:reason about="t">\nx\n</magnitude:reason> <${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  it('03: </magnitude:reason> + 10 spaces + < confirms (no bound)', () => {
    const input = `<magnitude:reason about="t">\nx\n</magnitude:reason>          <${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  it('04: </magnitude:reason> + tab + < confirms', () => {
    const input = `<magnitude:reason about="t">\nx\n</magnitude:reason>\t<${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  it('05: </magnitude:reason> + \\n + < confirms', () => {
    const input = `<magnitude:reason about="t">\nx\n</magnitude:reason>\n<${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  it('06: </magnitude:reason> + multiple \\n + < confirms', () => {
    const input = `<magnitude:reason about="t">\nx\n</magnitude:reason>\n\n\n<${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  it('07: </magnitude:reason> + mixed ws + < confirms', () => {
    const input = `<magnitude:reason about="t">\nx\n</magnitude:reason> \t\n \t\n<${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  // ---- Reason: rejected (false close tags) ----

  it('08: </magnitude:reason> + non-ws char rejects — false close becomes content', () => {
    const input = `<magnitude:reason about="t">\n</magnitude:reason>x more\n</magnitude:reason><${YIELD.slice(1)}`
    v().passes(input)
    const events = parse(input)
    expect(hasEvent(events, 'LensEnd')).toBe(true)
    const text = collectLensChunks(events)
    expect(text).toContain('</magnitude:reason>x more')
  })

  it('09: </magnitude:reason> + \\n without ever seeing < — incomplete', () => {
    const input = `<magnitude:reason about="t">\nx\n</magnitude:reason>\n`
    v().rejects(input)
  })

  it('10: false </magnitude:reason> then real close', () => {
    const input = `<magnitude:reason about="t">\nThe tag </magnitude:reason>x is used\n</magnitude:reason><${YIELD.slice(1)}`
    v().passes(input)
    const events = parse(input)
    expect(countEvents(events, 'LensEnd')).toBe(1)
    const text = collectLensChunks(events)
    expect(text).toContain('</magnitude:reason>x is used')
  })

  it('11: multiple false </magnitude:reason> before real one', () => {
    const input = `<magnitude:reason about="t">\n</magnitude:reason>a\n</magnitude:reason>b\n</magnitude:reason><${YIELD.slice(1)}`
    v().passes(input)
    const events = parse(input)
    expect(countEvents(events, 'LensEnd')).toBe(1)
    const text = collectLensChunks(events)
    expect(text).toContain('</magnitude:reason>a')
    expect(text).toContain('</magnitude:reason>b')
  })

  // ---- Reason: chaining ----

  it('12: </magnitude:reason> confirmed by <magnitude:reason (another lens)', () => {
    const input = `<magnitude:reason about="a">\nx\n</magnitude:reason><magnitude:reason about="b">\ny\n</magnitude:reason><${YIELD.slice(1)}`
    v().passes(input)
    expect(countEvents(parse(input), 'LensEnd')).toBe(2)
  })

  it('13: </magnitude:reason> confirmed by <invoke', () => {
    const input = `<magnitude:reason about="a">\nx\n</magnitude:reason><magnitude:invoke tool="tree">\n</magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  it('14: </magnitude:reason> confirmed by <magnitude:message', () => {
    const input = `<magnitude:reason about="a">\nx\n</magnitude:reason><magnitude:message to="u">\nh\n</magnitude:message><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
  })

  // ---- Reason: content edge cases ----

  it('15: content with < that is not a close tag', () => {
    const input = `<magnitude:reason about="t">\nfoo < bar\n</magnitude:reason><${YIELD.slice(1)}`
    v().passes(input)
    const text = collectLensChunks(parse(input))
    expect(text).toContain('foo < bar')
  })

  it('16: content with HTML-like tags', () => {
    const input = `<magnitude:reason about="t">\n<div>hello</div>\n</magnitude:reason><${YIELD.slice(1)}`
    v().passes(input)
    const text = collectLensChunks(parse(input))
    expect(text).toContain('<div>hello</div>')
  })

  it('17: content with partial close tag </rea', () => {
    const input = `<magnitude:reason about="t">\n</rea partial\n</magnitude:reason><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  it('18: content with </reasonX (wrong close tag name)', () => {
    const input = `<magnitude:reason about="t">\n</reasonX> stuff\n</magnitude:reason><${YIELD.slice(1)}`
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

  it('23: false </magnitude:message> in content', () => {
    const input = `<magnitude:message to="u">\nuse </magnitude:message>x to close\n</magnitude:message><${YIELD.slice(1)}`
    v().passes(input)
    const events = parse(input)
    expect(countEvents(events, 'MessageEnd')).toBe(1)
    const text = collectMessageChunks(events)
    expect(text).toContain('</magnitude:message>x to close')
  })

  it('24: </magnitude:message> + non-ws rejects', () => {
    const input = `<magnitude:message to="u">\n</magnitude:message>foo\n</magnitude:message><${YIELD.slice(1)}`
    v().passes(input)
    const text = collectMessageChunks(parse(input))
    expect(text).toContain('</magnitude:message>foo')
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
