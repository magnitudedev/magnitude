/**
 * Category 9: Multi-tool interactions
 *
 * Tests involving multiple tools in the same turn, mixed alias/canonical closes.
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator, parse, getToolInput, countEvents, hasEvent, YIELD_USER,
} from './helpers'

const v = () => grammarValidator()
const Y = YIELD_USER

describe('Category 9: multi-tool interactions', () => {
  // =========================================================================
  // Multiple invokes in same turn
  // =========================================================================

  it('01: two shell invokes, both canonical close', () => {
    const input =
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>\n` +
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">pwd</magnitude:parameter>\n</magnitude:invoke>\n${Y}`
    v().passes(input)
    expect(countEvents(parse(input), 'ToolInputReady')).toBe(2)
  })

  it('02: shell then edit, both canonical close', () => {
    const input =
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>\n` +
      `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:parameter>\n<magnitude:parameter name="old">x</magnitude:parameter>\n<magnitude:parameter name="new">y</magnitude:parameter>\n</magnitude:invoke>\n${Y}`
    v().passes(input)
    expect(countEvents(parse(input), 'ToolInputReady')).toBe(2)
  })

  it('03: shell with alias close then edit with alias close', () => {
    const input =
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:shell>\n` +
      `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:parameter>\n<magnitude:parameter name="old">x</magnitude:parameter>\n<magnitude:parameter name="new">y</magnitude:parameter>\n</magnitude:edit>\n${Y}`
    v().passes(input)
    expect(countEvents(parse(input), 'ToolInputReady')).toBe(2)
  })

  it('04: shell alias close then edit canonical close', () => {
    const input =
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:shell>\n` +
      `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:parameter>\n<magnitude:parameter name="old">x</magnitude:parameter>\n<magnitude:parameter name="new">y</magnitude:parameter>\n</magnitude:invoke>\n${Y}`
    v().passes(input)
  })

  it('05: think + message + two invokes with alias closes', () => {
    const input =
      `<magnitude:think about="t">plan</magnitude:think>\n` +
      `<magnitude:message to="u">doing</magnitude:message>\n` +
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:command>\n</magnitude:shell>\n` +
      `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:path>\n<magnitude:parameter name="old">x</magnitude:old>\n<magnitude:parameter name="new">y</magnitude:new>\n</magnitude:edit>\n${Y}`
    v().passes(input)
    expect(hasEvent(parse(input), 'LensEnd')).toBe(true)
    expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
    expect(countEvents(parse(input), 'ToolInputReady')).toBe(2)
  })

  // =========================================================================
  // Cross-tool alias rejection
  // =========================================================================

  it('06: </magnitude:shell> does not close edit invoke', () => {
    v().rejects(`<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:parameter>\n<magnitude:parameter name="old">x</magnitude:parameter>\n<magnitude:parameter name="new">y</magnitude:parameter>\n</magnitude:shell>\n${Y}`)
  })

  it('07: </magnitude:command> does not close path param in edit', () => {
    // edit has params: path, old, new — not command
    v().rejects(`<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:command>\n<magnitude:parameter name="old">x</magnitude:parameter>\n<magnitude:parameter name="new">y</magnitude:parameter>\n</magnitude:invoke>\n${Y}`)
  })

  it('08: </magnitude:path> does not close command param in shell', () => {
    // shell has param: command — not path
    v().rejects(`<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:path>\n</magnitude:invoke>\n${Y}`)
  })
})
