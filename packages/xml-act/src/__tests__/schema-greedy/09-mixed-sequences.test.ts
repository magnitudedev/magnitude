/**
 * Category 9: Mixed Full Turn Sequences
 *
 * Complete turns with various combinations of think, message, invoke, yield.
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

  it('02: think + yield', () => {
    const input = `<magnitude:think about="t">\nthinking\n</magnitude:think><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
  })

  it('03: message + yield', () => {
    const input = `<magnitude:message to="u">\nhello\n</magnitude:message><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
  })

  it('04: think + message + yield', () => {
    const input = `<magnitude:think about="t">\nthinking\n</magnitude:think><magnitude:message to="u">\nhello\n</magnitude:message><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
  })

  it('05: think + invoke + yield', () => {
    const input = `<magnitude:think about="t">\nplan\n</magnitude:think><magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
    expect(hasEvent(parse(input), 'ToolInputReady')).toBe(true)
  })

  it('06: think + message + invoke + yield', () => {
    const input =
      `<magnitude:think about="t">\nthinking\n</magnitude:think>` +
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

  it('11: multiple thinks + message + invoke + yield', () => {
    const input =
      `<magnitude:think about="a">\nfirst\n</magnitude:think>` +
      `<magnitude:think about="b">\nsecond\n</magnitude:think>` +
      `<magnitude:message to="u">\nhello\n</magnitude:message>` +
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(countEvents(parse(input), 'LensEnd')).toBe(2)
  })

  // ---- Forbidden ----

  it('12: yield before invoke rejected', () => {
    v().rejects(`${YIELD}<magnitude:invoke tool="shell">\n</magnitude:invoke><${YIELD.slice(1)}`)
  })

  it('13: think after message rejected', () => {
    v().rejects(`<magnitude:message to="u">\nh\n</magnitude:message><magnitude:think about="t">\nx\n</magnitude:think><${YIELD.slice(1)}`)
  })

  it('14: two yields rejected', () => {
    v().rejects(`${YIELD}${YIELD}`)
  })

  it('15: content after yield rejected', () => {
    v().rejects(`${YIELD}extra`)
  })
})
