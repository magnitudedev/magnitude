/**
 * Category 4: Bounded Parameter Count
 *
 * Grammar bounds the number of parameters per tool to the schema count.
 * Fewer params allowed (early close). More than N rejected.
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator, parse, parseWithGrep, hasEvent, getToolInputs, getToolInput, YIELD,
  GREP_TOOL_DEF,
} from './helpers'
import { buildValidator } from '../../grammar/__tests__/helpers'

const v = () => grammarValidator()

describe('bounded parameter count', () => {
  it('01: 0 params for 0-param tool accepted', () => {
    const input = `<invoke tool="tree">\n</invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'ToolInputReady')).toBe(true)
  })

  it('02: 1 param for 1-param tool accepted', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">ls</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))).toEqual({ command: 'ls' })
  })

  it('03: 0 params for 1-param tool accepted (early close)', () => {
    const input = `<invoke tool="shell">\n</invoke><${YIELD.slice(1)}`
    v().passes(input)
    // Parser may emit MissingRequiredField error but tool still closes
    expect(hasEvent(parse(input), 'ToolParseError') || hasEvent(parse(input), 'ToolInputReady')).toBe(true)
  })

  it('04: 3 params for 3-param tool accepted', () => {
    const input = `<invoke tool="edit">\n<parameter name="path">f</parameter><parameter name="old">x</parameter><parameter name="new">y</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))).toEqual({ path: 'f', old: 'x', new: 'y' })
  })

  it('05: 2 params for 3-param tool accepted (early close, ToolParseError for missing new)', () => {
    const input = `<invoke tool="edit">\n<parameter name="path">f</parameter><parameter name="old">x</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    const events = parse(input)
    // Missing required 'new' param → ToolParseError, no ToolInputReady
    expect(hasEvent(events, 'ToolParseError')).toBe(true)
    expect(getToolInputs(events).length).toBe(0)
  })

  it('06: 1 param for 3-param tool accepted (early close)', () => {
    const input = `<invoke tool="edit">\n<parameter name="path">f</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
  })

  it('07: 4th param (duplicate) absorbed as content of 3rd (greedy)', () => {
    const input = `<invoke tool="edit">\n<parameter name="path">f</parameter><parameter name="old">x</parameter><parameter name="new">y</parameter><parameter name="path">z</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    // Duplicate "path" absorbed as content of "new" param
    const ti = getToolInput(parse(input))
    expect(ti?.new).toBe('y')
  })

  it('08: 2nd param (duplicate) absorbed as content of 1st (greedy)', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">ls</parameter><parameter name="command">pwd</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    // Duplicate "command" absorbed as content of first "command"
    const ti = getToolInput(parse(input))
    expect(ti?.command).toBe('ls')
  })

  // 4-param tool (grep)
  it('09: all 4 params for 4-param tool accepted', () => {
    const gv = buildValidator([GREP_TOOL_DEF])
    const input = `<invoke tool="grep">\n<parameter name="pattern">TODO</parameter><parameter name="glob">*.ts</parameter><parameter name="path">src</parameter><parameter name="limit">10</parameter></invoke><${YIELD.slice(1)}`
    gv.passes(input)
  })

  it('10: 3 of 4 params accepted (early close)', () => {
    const gv = buildValidator([GREP_TOOL_DEF])
    const input = `<invoke tool="grep">\n<parameter name="pattern">TODO</parameter><parameter name="glob">*.ts</parameter><parameter name="path">src</parameter></invoke><${YIELD.slice(1)}`
    gv.passes(input)
  })

  it('11: 1 of 4 params accepted (early close)', () => {
    const gv = buildValidator([GREP_TOOL_DEF])
    const input = `<invoke tool="grep">\n<parameter name="pattern">TODO</parameter></invoke><${YIELD.slice(1)}`
    gv.passes(input)
  })

  it('12: 5th param (duplicate) absorbed as content of 4th (greedy)', () => {
    const gv = buildValidator([GREP_TOOL_DEF])
    const input = `<invoke tool="grep">\n<parameter name="pattern">a</parameter><parameter name="glob">b</parameter><parameter name="path">c</parameter><parameter name="limit">d</parameter><parameter name="pattern">e</parameter></invoke><${YIELD.slice(1)}`
    gv.passes(input)
    const events = parseWithGrep(input)
    const ti = getToolInput(events)
    expect(ti?.limit).toBe('d')
  })
})
