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
    const input = `<magnitude:reason about="t">\nthinking\n</magnitude:reason><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  it('03: message + yield', () => {
    const input = `<magnitude:message to="u">\nhello\n</magnitude:message><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
  })

  it('04: reason + message + yield', () => {
    const input = `<magnitude:reason about="t">\nthinking\n</magnitude:reason><magnitude:message to="u">\nhello\n</magnitude:message><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
  })

  it('05: reason + invoke + yield', () => {
    const input = `<magnitude:reason about="t">\nplan\n</magnitude:reason><magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
    expect(hasEvent(parse(input), 'ToolInputReady')).toBe(true)
  })

  it('06: reason + message + invoke + yield', () => {
    const input =
      `<magnitude:reason about="t">\nthinking\n</magnitude:reason>` +
      `<magnitude:message to="u">\nhello\n</magnitude:message>` +
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
    expect(hasEvent(parse(input), 'ToolInputReady')).toBe(true)
  })

  it('07: multiple invokes', () => {
    const input =
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke>` +
      `<magnitude:invoke tool="tree">\n</magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInputs(parse(input)).length).toBe(2)
  })

  it('08: three invokes then yield', () => {
    const input =
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">a</magnitude:parameter></magnitude:invoke>` +
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">b</magnitude:parameter></magnitude:invoke>` +
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">c</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInputs(parse(input)).length).toBe(3)
  })

  it('09: invoke + message + yield', () => {
    const input =
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke>` +
      `<magnitude:message to="u">\nresults\n</magnitude:message><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'ToolInputReady')).toBe(true)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
  })

  it('10: invoke with all edit params + message + yield', () => {
    const input =
      `<magnitude:invoke tool="edit">\n` +
      `<magnitude:parameter name="path">f</magnitude:parameter>` +
      `<magnitude:parameter name="old">x</magnitude:parameter>` +
      `<magnitude:parameter name="new">y</magnitude:parameter>` +
      `</magnitude:invoke>` +
      `<magnitude:message to="u">\ndone\n</magnitude:message><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'ToolInputReady')).toBe(true)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
  })

  it('11: multiple reasons + message + invoke + yield', () => {
    const input =
      `<magnitude:reason about="a">\nfirst\n</magnitude:reason>` +
      `<magnitude:reason about="b">\nsecond\n</magnitude:reason>` +
      `<magnitude:message to="u">\nhello\n</magnitude:message>` +
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(countEvents(parse(input), 'LensEnd')).toBe(2)
  })

  // ---- Forbidden ----

  it('12: yield before invoke rejected', () => {
    v().rejects(`${YIELD}<magnitude:invoke tool="shell">\n</magnitude:invoke><${YIELD.slice(1)}`)
  })

  it('13: reason after message rejected', () => {
    v().rejects(`<magnitude:message to="u">\nh\n</magnitude:message><magnitude:reason about="t">\nx\n</magnitude:reason><${YIELD.slice(1)}`)
  })

  it('14: two yields rejected', () => {
    v().rejects(`${YIELD}${YIELD}`)
  })

  it('15: content after yield rejected', () => {
    v().rejects(`${YIELD}extra`)
  })
})
