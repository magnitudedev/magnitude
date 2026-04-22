/**
 * Category 9: Mixed Full Turn Sequences
 *
 * Complete turns with various combinations of reason, message, invoke, yield.
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator, parse, hasEvent, countEvents, getToolInputs, YIELD,
} from './helpers'

const v = () => grammarValidator()

describe('mixed full turn sequences', () => {
  it('01: yield only', () => {
    const input = YIELD
    v().passes(input)
    expect(hasEvent(parse(input), 'TurnEnd')).toBe(true)
  })

  it('02: reason + yield', () => {
    const input = `<reason about="t">\nthinking\n</reason><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  it('03: message + yield', () => {
    const input = `<message to="u">\nhello\n</message><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
  })

  it('04: reason + message + yield', () => {
    const input = `<reason about="t">\nthinking\n</reason><message to="u">\nhello\n</message><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
  })

  it('05: reason + invoke + yield', () => {
    const input = `<reason about="t">\nplan\n</reason><invoke tool="shell">\n<parameter name="command">ls</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
    expect(hasEvent(parse(input), 'ToolInputReady')).toBe(true)
  })

  it('06: reason + message + invoke + yield', () => {
    const input =
      `<reason about="t">\nthinking\n</reason>` +
      `<message to="u">\nhello\n</message>` +
      `<invoke tool="shell">\n<parameter name="command">ls</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
    expect(hasEvent(parse(input), 'ToolInputReady')).toBe(true)
  })

  it('07: multiple invokes', () => {
    const input =
      `<invoke tool="shell">\n<parameter name="command">ls</parameter></invoke>` +
      `<invoke tool="tree">\n</invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInputs(parse(input)).length).toBe(2)
  })

  it('08: three invokes then yield', () => {
    const input =
      `<invoke tool="shell">\n<parameter name="command">a</parameter></invoke>` +
      `<invoke tool="shell">\n<parameter name="command">b</parameter></invoke>` +
      `<invoke tool="shell">\n<parameter name="command">c</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInputs(parse(input)).length).toBe(3)
  })

  it('09: invoke + message + yield', () => {
    const input =
      `<invoke tool="shell">\n<parameter name="command">ls</parameter></invoke>` +
      `<message to="u">\nresults\n</message><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'ToolInputReady')).toBe(true)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
  })

  it('10: invoke with all edit params + message + yield', () => {
    const input =
      `<invoke tool="edit">\n` +
      `<parameter name="path">f</parameter>` +
      `<parameter name="old">x</parameter>` +
      `<parameter name="new">y</parameter>` +
      `</invoke>` +
      `<message to="u">\ndone\n</message><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'ToolInputReady')).toBe(true)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
  })

  it('11: multiple reasons + message + invoke + yield', () => {
    const input =
      `<reason about="a">\nfirst\n</reason>` +
      `<reason about="b">\nsecond\n</reason>` +
      `<message to="u">\nhello\n</message>` +
      `<invoke tool="shell">\n<parameter name="command">ls</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(countEvents(parse(input), 'LensEnd')).toBe(2)
  })

  // ---- Forbidden ----

  it('12: yield before invoke rejected', () => {
    v().rejects(`${YIELD}<invoke tool="shell">\n</invoke><${YIELD.slice(1)}`)
  })

  it('13: reason after message rejected', () => {
    v().rejects(`<message to="u">\nh\n</message><reason about="t">\nx\n</reason><${YIELD.slice(1)}`)
  })

  it('14: two yields rejected', () => {
    v().rejects(`${YIELD}${YIELD}`)
  })

  it('15: content after yield rejected', () => {
    v().rejects(`${YIELD}extra`)
  })
})
