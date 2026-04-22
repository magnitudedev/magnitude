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

  it('01: </reason>< confirms (immediate)', () => {
    const input = `<reason about="t">\nx\n</reason><${YIELD.slice(1)}`
    v().passes(input)
    const events = parse(input)
    expect(hasEvent(events, 'LensEnd')).toBe(true)
  })

  it('02: </reason> + 1 space + < confirms', () => {
    const input = `<reason about="t">\nx\n</reason> <${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  it('03: </reason> + 10 spaces + < confirms (no bound)', () => {
    const input = `<reason about="t">\nx\n</reason>          <${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  it('04: </reason> + tab + < confirms', () => {
    const input = `<reason about="t">\nx\n</reason>\t<${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  it('05: </reason> + \\n + < confirms', () => {
    const input = `<reason about="t">\nx\n</reason>\n<${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  it('06: </reason> + multiple \\n + < confirms', () => {
    const input = `<reason about="t">\nx\n</reason>\n\n\n<${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  it('07: </reason> + mixed ws + < confirms', () => {
    const input = `<reason about="t">\nx\n</reason> \t\n \t\n<${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  // ---- Reason: rejected (false close tags) ----

  it('08: </reason> + non-ws char rejects — false close becomes content', () => {
    const input = `<reason about="t">\n</reason>x more\n</reason><${YIELD.slice(1)}`
    v().passes(input)
    const events = parse(input)
    expect(hasEvent(events, 'LensEnd')).toBe(true)
    const text = collectLensChunks(events)
    expect(text).toContain('</reason>x more')
  })

  it('09: </reason> + \\n without ever seeing < — incomplete', () => {
    const input = `<reason about="t">\nx\n</reason>\n`
    v().rejects(input)
  })

  it('10: false </reason> then real close', () => {
    const input = `<reason about="t">\nThe tag </reason>x is used\n</reason><${YIELD.slice(1)}`
    v().passes(input)
    const events = parse(input)
    expect(countEvents(events, 'LensEnd')).toBe(1)
    const text = collectLensChunks(events)
    expect(text).toContain('</reason>x is used')
  })

  it('11: multiple false </reason> before real one', () => {
    const input = `<reason about="t">\n</reason>a\n</reason>b\n</reason><${YIELD.slice(1)}`
    v().passes(input)
    const events = parse(input)
    expect(countEvents(events, 'LensEnd')).toBe(1)
    const text = collectLensChunks(events)
    expect(text).toContain('</reason>a')
    expect(text).toContain('</reason>b')
  })

  // ---- Reason: chaining ----

  it('12: </reason> confirmed by <reason (another lens)', () => {
    const input = `<reason about="a">\nx\n</reason><reason about="b">\ny\n</reason><${YIELD.slice(1)}`
    v().passes(input)
    expect(countEvents(parse(input), 'LensEnd')).toBe(2)
  })

  it('13: </reason> confirmed by <invoke', () => {
    const input = `<reason about="a">\nx\n</reason><invoke tool="tree">\n</invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  it('14: </reason> confirmed by <message', () => {
    const input = `<reason about="a">\nx\n</reason><message to="u">\nh\n</message><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
  })

  // ---- Reason: content edge cases ----

  it('15: content with < that is not a close tag', () => {
    const input = `<reason about="t">\nfoo < bar\n</reason><${YIELD.slice(1)}`
    v().passes(input)
    const text = collectLensChunks(parse(input))
    expect(text).toContain('foo < bar')
  })

  it('16: content with HTML-like tags', () => {
    const input = `<reason about="t">\n<div>hello</div>\n</reason><${YIELD.slice(1)}`
    v().passes(input)
    const text = collectLensChunks(parse(input))
    expect(text).toContain('<div>hello</div>')
  })

  it('17: content with partial close tag </rea', () => {
    const input = `<reason about="t">\n</rea partial\n</reason><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  it('18: content with </reasonX (wrong close tag name)', () => {
    const input = `<reason about="t">\n</reasonX> stuff\n</reason><${YIELD.slice(1)}`
    v().passes(input)
    const text = collectLensChunks(parse(input))
    expect(text).toContain('</reasonX>')
  })

  // ---- Message: confirmed ----

  it('19: </message>< confirms (yield)', () => {
    const input = `<message to="u">\nh\n</message><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
  })

  it('20: </message> + \\n + < confirms', () => {
    const input = `<message to="u">\nh\n</message>\n<${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
  })

  it('21: </message> + spaces + < confirms', () => {
    const input = `<message to="u">\nh\n</message>  <${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
  })

  it('22: </message> + 20 spaces + < confirms (no bound)', () => {
    const input = `<message to="u">\nh\n</message>                    <${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
  })

  // ---- Message: rejected ----

  it('23: false </message> in content', () => {
    const input = `<message to="u">\nuse </message>x to close\n</message><${YIELD.slice(1)}`
    v().passes(input)
    const events = parse(input)
    expect(countEvents(events, 'MessageEnd')).toBe(1)
    const text = collectMessageChunks(events)
    expect(text).toContain('</message>x to close')
  })

  it('24: </message> + non-ws rejects', () => {
    const input = `<message to="u">\n</message>foo\n</message><${YIELD.slice(1)}`
    v().passes(input)
    const text = collectMessageChunks(parse(input))
    expect(text).toContain('</message>foo')
  })

  it('25: </message> + \\n alone (no <) does not terminate', () => {
    const input = `<message to="u">\nh\n</message>\n`
    v().rejects(input)
  })

  // ---- Message: chaining ----

  it('26: </message> confirmed by <invoke', () => {
    const input = `<message to="u">\nh\n</message><invoke tool="shell">\n<parameter name="command">ls</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
    expect(hasEvent(parse(input), 'ToolInputReady')).toBe(true)
  })

  it('27: </message> confirmed by <message (another message)', () => {
    const input = `<message to="u">\na\n</message><message to="v">\nb\n</message><${YIELD.slice(1)}`
    v().passes(input)
    expect(countEvents(parse(input), 'MessageEnd')).toBe(2)
  })
})
