/**
 * Category 11: Edge Cases
 *
 * Empty bodies, single characters, partial close tags, unusual content.
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator, parse, hasEvent, getToolInput, collectLensChunks, YIELD,
} from './helpers'

const v = () => grammarValidator()

describe('edge cases', () => {
  it('01: empty parameter body', () => {
    const input = `<invoke tool="shell">\n<parameter name="command"></parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('')
  })

  it('02: parameter body is single character', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">x</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('x')
  })

  it('03: parameter body is a single <', () => {
    const input = `<invoke tool="shell">\n<parameter name="command"><</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('<')
  })

  it('04: parameter body ends with <', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">foo <</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('foo <')
  })

  it('05: parameter body is just newlines', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">\n\n\n</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
  })

  it('06: parameter body with < comparison operators', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">if [ $a -lt $b ] && [ $c < $d ]; then echo yes; fi</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toContain('$c < $d')
  })

  it('07: parameter body with partial close tag </param', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">text </param more</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('text </param more')
  })

  it('08: parameter body with </parameterX (wrong close tag)', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">text </parameterX> more</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toContain('</parameterX>')
  })

  it('09: reason with < in content', () => {
    const input = `<reason about="t">\nfoo < bar\n</reason><${YIELD.slice(1)}`
    v().passes(input)
    expect(collectLensChunks(parse(input))).toContain('foo < bar')
  })

  it('10: reason body empty', () => {
    const input = `<reason about="t">\n\n</reason><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  it('11: message body empty', () => {
    const input = `<message to="u">\n\n</message><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
  })

  it('12: very long parameter content', () => {
    const longContent = 'x'.repeat(200)
    const input = `<invoke tool="shell">\n<parameter name="command">${longContent}</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe(longContent)
  })

  it('13: parameter body with many < characters', () => {
    const input = `<invoke tool="shell">\n<parameter name="command"><<<<<<<<<</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('<<<<<<<<<')
  })

  it('14: parameter containing heredoc-style content', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">cat << 'EOF'\nline1\nline2\nEOF</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toContain('EOF')
  })

  it('15: invoke with no close (just param then yield) — rejected', () => {
    v().rejects(`<invoke tool="shell">\n<parameter name="command">ls</parameter>${YIELD}`)
  })

  it('16: param outside invoke — rejected', () => {
    v().rejects(`<parameter name="command">ls</parameter>${YIELD}`)
  })
})
