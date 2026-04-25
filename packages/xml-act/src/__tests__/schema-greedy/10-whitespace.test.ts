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
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:parameter><magnitude:parameter name="old">x</magnitude:parameter><magnitude:parameter name="new">y</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.path).toBe('f')
  })

  it('02: newlines between params', () => {
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:parameter>\n<magnitude:parameter name="old">x</magnitude:parameter><magnitude:parameter name="new">y</magnitude:parameter>\n</magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.path).toBe('f')
  })

  it('03: multiple newlines between params', () => {
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:parameter>\n\n\n<magnitude:parameter name="old">x</magnitude:parameter>\n<magnitude:parameter name="new">y</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
  })

  it('04: tabs between params', () => {
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:parameter>\t\t<magnitude:parameter name="old">x</magnitude:parameter>\t<magnitude:parameter name="new">y</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
  })

  it('05: whitespace before first param', () => {
    const input = `<magnitude:invoke tool="shell">\n  <magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
  })

  it('06: no newline after invoke open', () => {
    const input = `<magnitude:invoke tool="shell"><magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
  })

  it('07: lots of whitespace after invoke open', () => {
    const input = `<magnitude:invoke tool="shell">\n\n\n<magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
  })

  it('08: whitespace between invoke close and next invoke', () => {
    const input =
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke>` +
      `\n\n` +
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">pwd</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
  })

  it('09: whitespace between reason close and message open', () => {
    const input = `<magnitude:reason about="t">\nx\n</magnitude:reason>  \t\n  <magnitude:message to="u">\nh\n</magnitude:message><${YIELD.slice(1)}`
    v().passes(input)
  })

  it('10: no whitespace anywhere (maximally compact)', () => {
    const input = `<magnitude:invoke tool="shell"><magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('ls')
  })
})
