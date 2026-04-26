/**
 * Rule: inside invoke, only parameter/filter and known parameter aliases are valid.
 */
import { describe, it } from 'vitest'
import {
  grammarValidator,
  grepGrammarValidator,
  parse,
  parseWithGrep,
  expectStructuralError,
  expectNoStructuralError,
} from './helpers'

const v = () => grammarValidator()

describe('prefix heuristics: invalid invoke body', () => {
  const invalidDefaultCases = [
    {
      label: '01: message open inside invoke',
      input: '<magnitude:invoke tool="shell"><magnitude:message>x</magnitude:message></magnitude:invoke><magnitude:yield_user/>',
      raw: '<magnitude:message>',
    },
    {
      label: '02: think open inside invoke',
      input: '<magnitude:invoke tool="shell"><magnitude:think>x</magnitude:think></magnitude:invoke><magnitude:yield_user/>',
      raw: '<magnitude:think>',
    },
    {
      label: '03: invoke open inside invoke',
      input: '<magnitude:invoke tool="shell"><magnitude:invoke tool="shell"></magnitude:invoke></magnitude:invoke><magnitude:yield_user/>',
      raw: '<magnitude:invoke tool="shell">',
    },
    {
      label: '04: shell alias open inside invoke',
      input: '<magnitude:invoke tool="shell"><magnitude:shell></magnitude:shell></magnitude:invoke><magnitude:yield_user/>',
      raw: '<magnitude:shell>',
    },
    {
      label: '05: unknown prefixed open inside invoke',
      input: '<magnitude:invoke tool="shell"><magnitude:foo>bar</magnitude:foo></magnitude:invoke><magnitude:yield_user/>',
      raw: '<magnitude:foo>',
    },
    {
      label: '08: unknown parameter-like alias for shell',
      input: '<magnitude:invoke tool="shell"><magnitude:path>a.ts</magnitude:path></magnitude:invoke><magnitude:yield_user/>',
      raw: '<magnitude:path>',
    },
    {
      label: '12: escape is not a valid invoke child',
      input: '<magnitude:invoke tool="shell"><magnitude:escape><magnitude:message>x</magnitude:message></magnitude:escape></magnitude:invoke><magnitude:yield_user/>',
      raw: '<magnitude:escape>',
    },
  ] as const

  for (const testCase of invalidDefaultCases) {
    it(testCase.label, () => {
      v().rejects(testCase.input)
      const events = parse(testCase.input)
      expectStructuralError(events, {
        variant: 'InvalidMagnitudeOpen',
        tagName: testCase.raw.slice(1, -1).split(' ')[0],
        parentTagName: 'magnitude:invoke',
        detailIncludes: [testCase.raw, 'magnitude:invoke'],
      })
    })
  }

  it('06: canonical parameter remains valid', () => {
    const input = '<magnitude:invoke tool="shell"><magnitude:parameter name="command">pwd</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>'
    v().passes(input)
    const events = parse(input)
    expectNoStructuralError(events)
  })

  it('07: command alias is rejected by grammar but still parsed heuristically', () => {
    const input = '<magnitude:invoke tool="shell"><magnitude:command>pwd</magnitude:command></magnitude:invoke><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectNoStructuralError(events)
  })

  it('09: grep pattern alias valid under parser heuristics but rejected by grammar', () => {
    const input = '<magnitude:invoke tool="grep"><magnitude:pattern>TODO</magnitude:pattern></magnitude:invoke><magnitude:yield_user/>'
    grepGrammarValidator().rejects(input)
    const events = parseWithGrep(input)
    expectNoStructuralError(events)
  })

  it('10: grep command alias invalid under grep schema', () => {
    const input = '<magnitude:invoke tool="grep"><magnitude:command>pwd</magnitude:command></magnitude:invoke><magnitude:yield_user/>'
    grepGrammarValidator().rejects(input)
    const events = parseWithGrep(input)
    expectStructuralError(events, {
      variant: 'InvalidMagnitudeOpen',
      tagName: 'magnitude:command',
      parentTagName: 'magnitude:invoke',
      detailIncludes: ['<magnitude:command>', 'magnitude:invoke'],
    })
  })

  it('11: filter + canonical parameter remains valid', () => {
    const input = '<magnitude:invoke tool="shell"><magnitude:filter>*.ts</magnitude:filter><magnitude:parameter name="command">pwd</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>'
    v().passes(input)
    const events = parse(input)
    expectNoStructuralError(events)
  })
})
