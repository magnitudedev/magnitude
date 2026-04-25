/**
 * Category 6: Last-Parameter Deep Greedy Matching
 *
 * When we know with certainty this is the last parameter (all N slots consumed),
 * confirmation deepens: </magnitude:parameter> + ws + </magnitude:invoke> + ws + < (next top-level tag).
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

  it('01: last param (1/1 shell) confirmed by </magnitude:parameter></magnitude:invoke><', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls -la</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('ls -la')
  })

  it('02: last param (3/3 edit) confirmed by </magnitude:parameter></magnitude:invoke><', () => {
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:parameter><magnitude:parameter name="old">x</magnitude:parameter><magnitude:parameter name="new">y</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))).toEqual({ path: 'f', old: 'x', new: 'y' })
  })

  it('03: last param with ws between </magnitude:parameter> and </magnitude:invoke>', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter> </magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('ls')
  })

  it('04: last param with \\n between </magnitude:parameter> and </magnitude:invoke>', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('ls')
  })

  it('05: last param with ws between </magnitude:invoke> and next tag', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke> <${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('ls')
  })

  it('06: last param with \\n between </magnitude:invoke> and next tag', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke>\n<${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('ls')
  })

  it('07: last param empty body', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command"></magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('')
  })

  it('08: last param body is only whitespace', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">   </magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
  })

  // ---- False closes absorbed ----

  it('09: false </magnitude:parameter> in last param content', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo </magnitude:parameter>; ls</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('echo </magnitude:parameter>; ls')
  })

  it('10: false </magnitude:parameter></magnitude:invoke> in content (no < after)', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">text</magnitude:parameter></magnitude:invoke>MORE text</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('text</magnitude:parameter></magnitude:invoke>MORE text')
  })

  it('11: false </magnitude:parameter></magnitude:invoke> followed by \\n (not <)', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">text</magnitude:parameter></magnitude:invoke>\nmore</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toContain('text')
  })

  it('12: false </magnitude:parameter></magnitude:invoke> followed by space (not <)', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">text</magnitude:parameter></magnitude:invoke> still content</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toContain('text</magnitude:parameter></magnitude:invoke> still content')
  })

  it('13: content is exactly </magnitude:parameter></magnitude:invoke> (needs real close after)', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command"></magnitude:parameter></magnitude:invoke>!</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('</magnitude:parameter></magnitude:invoke>!')
  })

  it('14: multiple false </magnitude:parameter></magnitude:invoke> sequences', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command"></magnitude:parameter></magnitude:invoke>a</magnitude:parameter></magnitude:invoke>b</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('</magnitude:parameter></magnitude:invoke>a</magnitude:parameter></magnitude:invoke>b')
  })

  // ---- Filter path tightened ----

  it('15: last param slot but filter follows instead — rejected in current grammar', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter><magnitude:filter>$.stdout</magnitude:filter></magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  // ---- Multiple invokes ----

  it('16: two invokes, each with last-param deep confirmation', () => {
    const input =
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke>` +
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">pwd</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    const inputs = getToolInputs(parse(input))
    expect(inputs.length).toBe(2)
    expect(inputs[0].input).toEqual({ command: 'ls' })
    expect(inputs[1].input).toEqual({ command: 'pwd' })
  })

  // ---- 4-param tool last param ----

  it('17: 4/4 grep params, last gets deep confirmation', () => {
    const gv = buildValidator([GREP_TOOL_DEF])
    const input = `<magnitude:invoke tool="grep">\n<magnitude:parameter name="pattern">TODO</magnitude:parameter><magnitude:parameter name="glob">*.ts</magnitude:parameter><magnitude:parameter name="path">src</magnitude:parameter><magnitude:parameter name="limit">10</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    gv.passes(input)
  })

  it('18: false </magnitude:parameter></magnitude:invoke> in 4th param of grep', () => {
    const gv = buildValidator([GREP_TOOL_DEF])
    const input = `<magnitude:invoke tool="grep">\n<magnitude:parameter name="pattern">a</magnitude:parameter><magnitude:parameter name="glob">b</magnitude:parameter><magnitude:parameter name="path">c</magnitude:parameter><magnitude:parameter name="limit">text</magnitude:parameter></magnitude:invoke>NOT_LT more</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    gv.passes(input)
  })
})
