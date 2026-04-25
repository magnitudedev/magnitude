/**
 * Category 4: Bounded Parameter Count
 *
 * Grammar bounds the number of parameters per tool to the schema count
 * and now requires all required params before close. Extra params are rejected.
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
    const input = `<magnitude:invoke tool="tree">\n</magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'ToolInputReady')).toBe(true)
  })

  it('02: 1 param for 1-param tool accepted', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))).toEqual({ command: 'ls' })
  })

  it('03: 0 params for 1-param tool rejected', () => {
    const input = `<magnitude:invoke tool="shell">\n</magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('04: 3 params for 3-param tool accepted', () => {
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:parameter><magnitude:parameter name="old">x</magnitude:parameter><magnitude:parameter name="new">y</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))).toEqual({ path: 'f', old: 'x', new: 'y' })
  })

  it('05: 2 params for 3-param tool rejected', () => {
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:parameter><magnitude:parameter name="old">x</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('06: 1 param for 3-param tool rejected', () => {
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('07: 4th param (duplicate) rejected', () => {
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:parameter><magnitude:parameter name="old">x</magnitude:parameter><magnitude:parameter name="new">y</magnitude:parameter><magnitude:parameter name="path">z</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('08: 2nd param (duplicate) rejected by grammar before parser duplicate handling', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter><magnitude:parameter name="command">pwd</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  // 4-param tool (grep)
  it('09: all 4 params for 4-param tool accepted', () => {
    const gv = buildValidator([GREP_TOOL_DEF])
    const input = `<magnitude:invoke tool="grep">\n<magnitude:parameter name="pattern">TODO</magnitude:parameter><magnitude:parameter name="glob">*.ts</magnitude:parameter><magnitude:parameter name="path">src</magnitude:parameter><magnitude:parameter name="limit">10</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    gv.passes(input)
  })

  it('10: 3 of 4 params rejected', () => {
    const gv = buildValidator([GREP_TOOL_DEF])
    const input = `<magnitude:invoke tool="grep">\n<magnitude:parameter name="pattern">TODO</magnitude:parameter><magnitude:parameter name="glob">*.ts</magnitude:parameter><magnitude:parameter name="path">src</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    gv.rejects(input)
  })

  it('11: 1 of 4 params rejected', () => {
    const gv = buildValidator([GREP_TOOL_DEF])
    const input = `<magnitude:invoke tool="grep">\n<magnitude:parameter name="pattern">TODO</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    gv.rejects(input)
  })

  it('12: 5th param (duplicate) rejected', () => {
    const gv = buildValidator([GREP_TOOL_DEF])
    const input = `<magnitude:invoke tool="grep">\n<magnitude:parameter name="pattern">a</magnitude:parameter><magnitude:parameter name="glob">b</magnitude:parameter><magnitude:parameter name="path">c</magnitude:parameter><magnitude:parameter name="limit">d</magnitude:parameter><magnitude:parameter name="pattern">e</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    gv.rejects(input)
  })
})
