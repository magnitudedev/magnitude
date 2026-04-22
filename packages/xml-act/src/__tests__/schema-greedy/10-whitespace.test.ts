/**
 * Category 10: Whitespace Variations
 *
 * Various whitespace patterns inside invokes and between tags.
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator, parse, getToolInput, getToolInputs, YIELD,
} from './helpers'

const v = () => grammarValidator()

describe('whitespace variations', () => {
  it('01: no whitespace between params', () => {
    const input = `<invoke tool="edit">\n<parameter name="path">f</parameter><parameter name="old">x</parameter><parameter name="new">y</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.path).toBe('f')
  })

  it('02: newlines between params', () => {
    const input = `<invoke tool="edit">\n<parameter name="path">f</parameter>\n<parameter name="old">x</parameter><parameter name="new">y</parameter>\n</invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.path).toBe('f')
  })

  it('03: multiple newlines between params', () => {
    const input = `<invoke tool="edit">\n<parameter name="path">f</parameter>\n\n\n<parameter name="old">x</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
  })

  it('04: tabs between params', () => {
    const input = `<invoke tool="edit">\n<parameter name="path">f</parameter>\t\t<parameter name="old">x</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
  })

  it('05: whitespace before first param', () => {
    const input = `<invoke tool="shell">\n  <parameter name="command">ls</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
  })

  it('06: no newline after invoke open', () => {
    const input = `<invoke tool="shell"><parameter name="command">ls</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
  })

  it('07: lots of whitespace after invoke open', () => {
    const input = `<invoke tool="shell">\n\n\n<parameter name="command">ls</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
  })

  it('08: whitespace between invoke close and next invoke', () => {
    const input =
      `<invoke tool="shell">\n<parameter name="command">ls</parameter></invoke>` +
      `\n\n` +
      `<invoke tool="shell">\n<parameter name="command">pwd</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
  })

  it('09: whitespace between reason close and message open', () => {
    const input = `<reason about="t">\nx\n</reason>  \t\n  <message to="u">\nh\n</message><${YIELD.slice(1)}`
    v().passes(input)
  })

  it('10: no whitespace anywhere (maximally compact)', () => {
    const input = `<invoke tool="shell"><parameter name="command">ls</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('ls')
  })
})
