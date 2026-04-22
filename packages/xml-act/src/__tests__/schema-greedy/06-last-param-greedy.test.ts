/**
 * Category 6: Last-Parameter Deep Greedy Matching
 *
 * When we know with certainty this is the last parameter (all N slots consumed),
 * confirmation deepens: </parameter> + ws + </invoke> + ws + < (next top-level tag).
 * This is the strongest possible signal.
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator, parse, hasEvent, getToolInput, getToolInputs,
  YIELD, GREP_TOOL_DEF,
} from './helpers'
import { buildValidator } from '../../grammar/__tests__/helpers'

const v = () => grammarValidator()

describe('last-parameter deep greedy matching', () => {
  // ---- Confirmed ----

  it('01: last param (1/1 shell) confirmed by </parameter></invoke><', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">ls -la</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('ls -la')
  })

  it('02: last param (3/3 edit) confirmed by </parameter></invoke><', () => {
    const input = `<invoke tool="edit">\n<parameter name="path">f</parameter><parameter name="old">x</parameter><parameter name="new">y</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))).toEqual({ path: 'f', old: 'x', new: 'y' })
  })

  it('03: last param with ws between </parameter> and </invoke>', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">ls</parameter> </invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('ls')
  })

  it('04: last param with \\n between </parameter> and </invoke>', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">ls</parameter>\n</invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('ls')
  })

  it('05: last param with ws between </invoke> and next tag', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">ls</parameter></invoke> <${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('ls')
  })

  it('06: last param with \\n between </invoke> and next tag', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">ls</parameter></invoke>\n<${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('ls')
  })

  it('07: last param empty body', () => {
    const input = `<invoke tool="shell">\n<parameter name="command"></parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('')
  })

  it('08: last param body is only whitespace', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">   </parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
  })

  // ---- False closes absorbed ----

  it('09: false </parameter> in last param content', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">echo </parameter>; ls</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('echo </parameter>; ls')
  })

  it('10: false </parameter></invoke> in content (no < after)', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">text</parameter></invoke>MORE text</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('text</parameter></invoke>MORE text')
  })

  it('11: false </parameter></invoke> followed by \\n (not <)', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">text</parameter></invoke>\nmore</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toContain('text')
  })

  it('12: false </parameter></invoke> followed by space (not <)', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">text</parameter></invoke> still content</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toContain('text</parameter></invoke> still content')
  })

  it('13: content is exactly </parameter></invoke> (needs real close after)', () => {
    const input = `<invoke tool="shell">\n<parameter name="command"></parameter></invoke>!</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('</parameter></invoke>!')
  })

  it('14: multiple false </parameter></invoke> sequences', () => {
    const input = `<invoke tool="shell">\n<parameter name="command"></parameter></invoke>a</parameter></invoke>b</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('</parameter></invoke>a</parameter></invoke>b')
  })

  // ---- Last param still allows filter ----

  it('15: last param slot but filter follows instead', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">ls</parameter><filter>$.stdout</filter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('ls')
  })

  // ---- Multiple invokes ----

  it('16: two invokes, each with last-param deep confirmation', () => {
    const input =
      `<invoke tool="shell">\n<parameter name="command">ls</parameter></invoke>` +
      `<invoke tool="shell">\n<parameter name="command">pwd</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    const inputs = getToolInputs(parse(input))
    expect(inputs.length).toBe(2)
    expect(inputs[0].input).toEqual({ command: 'ls' })
    expect(inputs[1].input).toEqual({ command: 'pwd' })
  })

  // ---- 4-param tool last param ----

  it('17: 4/4 grep params, last gets deep confirmation', () => {
    const gv = buildValidator([GREP_TOOL_DEF])
    const input = `<invoke tool="grep">\n<parameter name="pattern">TODO</parameter><parameter name="glob">*.ts</parameter><parameter name="path">src</parameter><parameter name="limit">10</parameter></invoke><${YIELD.slice(1)}`
    gv.passes(input)
  })

  it('18: false </parameter></invoke> in 4th param of grep', () => {
    const gv = buildValidator([GREP_TOOL_DEF])
    const input = `<invoke tool="grep">\n<parameter name="pattern">a</parameter><parameter name="glob">b</parameter><parameter name="path">c</parameter><parameter name="limit">text</parameter></invoke>NOT_LT more</parameter></invoke><${YIELD.slice(1)}`
    gv.passes(input)
  })
})
