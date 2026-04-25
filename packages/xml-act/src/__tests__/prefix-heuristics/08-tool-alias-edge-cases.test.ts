/**
 * Rule: tool aliases are schema-driven and narrow.
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator,
  grepGrammarValidator,
  parse,
  parseWithGrep,
  expectStructuralError,
  expectNoStructuralError,
  expectPreservedInMessage,
  getToolInput,
} from './helpers'

const v = () => grammarValidator()
const vg = () => grepGrammarValidator()

describe('prefix heuristics: tool alias edge cases', () => {
  it('01: self-closing shell alias rejected', () => {
    const input = '<magnitude:shell/><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectStructuralError(events, {
      variant: 'InvalidMagnitudeOpen',
      tagName: 'magnitude:shell',
      detailIncludes: ['magnitude:shell'],
    })
  })

  it('02: self-closing tree alias rejected', () => {
    const input = '<magnitude:tree/><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectStructuralError(events, {
      variant: 'InvalidMagnitudeOpen',
      tagName: 'magnitude:tree',
      detailIncludes: ['magnitude:tree'],
    })
  })

  it('03: shell alias with extra attrs', () => {
    const input = '<magnitude:shell mode="x"><magnitude:command>pwd</magnitude:command></magnitude:shell><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expect(getToolInput(events)).toEqual({ command: 'pwd' })
  })

  it('04: edit alias with extra attrs', () => {
    const input = '<magnitude:edit data-x="1"><magnitude:path>a.ts</magnitude:path><magnitude:old>x</magnitude:old><magnitude:new>y</magnitude:new></magnitude:edit><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expect(getToolInput(events)).toEqual({ path: 'a.ts', old: 'x', new: 'y' })
  })

  it('05: unknown alias-looking foo with attrs invalid', () => {
    const input = '<magnitude:foo mode="x">bar</magnitude:foo><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectStructuralError(events, {
      variant: 'InvalidMagnitudeOpen',
      tagName: 'magnitude:foo',
      detailIncludes: ['magnitude:foo'],
    })
  })

  it('06: shell alias mixed with canonical invoke', () => {
    const input = '<magnitude:shell><magnitude:command>pwd</magnitude:command></magnitude:shell><magnitude:invoke tool="tree"></magnitude:invoke><magnitude:yield_user/>'
    v().passes(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expect(getToolInput(events, 0)).toEqual({ command: 'pwd' })
    expect(getToolInput(events, 1)).toEqual({})
  })

  it('07: canonical invoke mixed with shell alias', () => {
    const input = '<magnitude:invoke tool="tree"></magnitude:invoke><magnitude:shell><magnitude:command>pwd</magnitude:command></magnitude:shell><magnitude:yield_user/>'
    v().passes(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expect(getToolInput(events, 0)).toEqual({})
    expect(getToolInput(events, 1)).toEqual({ command: 'pwd' })
  })

  it('08: grep alias as known tool under grep schema', () => {
    const input = '<magnitude:grep><magnitude:pattern>TODO</magnitude:pattern></magnitude:grep><magnitude:yield_user/>'
    vg().passes(input)
    const events = parseWithGrep(input)
    expectNoStructuralError(events)
    expect(getToolInput(events)).toEqual({ pattern: 'TODO' })
  })

  it('09: grep alias unavailable under default schema', () => {
    const input = '<magnitude:grep><magnitude:pattern>TODO</magnitude:pattern></magnitude:grep><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectStructuralError(events, {
      variant: 'InvalidMagnitudeOpen',
      tagName: 'magnitude:grep',
      detailIncludes: ['magnitude:grep'],
    })
  })

  it('10: tree alias followed by message', () => {
    const input = '<magnitude:tree></magnitude:tree><magnitude:message>done</magnitude:message><magnitude:yield_user/>'
    v().passes(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expect(getToolInput(events)).toEqual({})
  })

  it('11: shell alias nested in message still invalid', () => {
    const input = '<magnitude:message>pre <magnitude:shell><magnitude:command>pwd</magnitude:command></magnitude:shell> post</magnitude:message><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectStructuralError(events, {
      variant: 'InvalidMagnitudeOpen',
      tagName: 'magnitude:shell',
      parentTagName: 'magnitude:message',
      detailIncludes: ['<magnitude:shell>', 'magnitude:message'],
    })
    expectPreservedInMessage(events, '<magnitude:shell>')
  })

  it('12: tree alias nested in invoke still invalid', () => {
    const input = '<magnitude:invoke tool="shell"><magnitude:tree></magnitude:tree></magnitude:invoke><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectStructuralError(events, {
      variant: 'InvalidMagnitudeOpen',
      tagName: 'magnitude:tree',
      parentTagName: 'magnitude:invoke',
      detailIncludes: ['<magnitude:tree>', 'magnitude:invoke'],
    })
  })
})
