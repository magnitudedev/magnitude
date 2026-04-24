/**
 * Rule: unknown magnitude:* opens are always errors at top level.
 */
import { describe, it } from 'vitest'
import {
  grammarValidator,
  parse,
  expectStructuralError,
  expectNoStructuralError,
} from './helpers'

const v = () => grammarValidator()

describe('prefix heuristics: unknown top-level', () => {
  const invalidCases = [
    {
      label: '01: unknown foo at top level',
      input: '<magnitude:foo>bar</magnitude:foo><magnitude:yield_user/>',
      raw: '<magnitude:foo>',
    },
    {
      label: '02: unknown capitalized name at top level',
      input: '<magnitude:Foo>bar</magnitude:Foo><magnitude:yield_user/>',
      raw: '<magnitude:Foo>',
    },
    {
      label: '03: unknown hyphenated name at top level',
      input: '<magnitude:foo-bar>bar</magnitude:foo-bar><magnitude:yield_user/>',
      raw: '<magnitude:foo-bar>',
    },
    {
      label: '04: unknown alias-looking tag with attrs',
      input: '<magnitude:foo x="1">bar</magnitude:foo><magnitude:yield_user/>',
      raw: '<magnitude:foo x="1">',
    },
    {
      label: '05: unknown empty top-level tag',
      input: '<magnitude:foo></magnitude:foo><magnitude:yield_user/>',
      raw: '<magnitude:foo>',
    },
    {
      label: '08: unknown top-level between valid message and yield',
      input: '<magnitude:message>hello</magnitude:message><magnitude:foo>bar</magnitude:foo><magnitude:yield_user/>',
      raw: '<magnitude:foo>',
    },
    {
      label: '09: unknown top-level before valid shell alias',
      input: '<magnitude:foo>bar</magnitude:foo><magnitude:shell><magnitude:command>pwd</magnitude:command></magnitude:shell><magnitude:yield_user/>',
      raw: '<magnitude:foo>',
    },
    {
      label: '10: unknown top-level before canonical invoke',
      input: '<magnitude:foo>bar</magnitude:foo><magnitude:invoke tool="shell"><magnitude:parameter name="command">pwd</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>',
      raw: '<magnitude:foo>',
    },
  ] as const

  for (const testCase of invalidCases) {
    it(testCase.label, () => {
      v().rejects(testCase.input)
      const events = parse(testCase.input)
      expectStructuralError(events, {
        variant: 'InvalidMagnitudeOpen',
        tagName: testCase.raw.slice(1, -1).split(' ')[0],
        detailIncludes: [testCase.raw, 'top level', 'magnitude:escape'],
      })
    })
  }

  it('06: known tool alias shell remains valid', () => {
    const input = '<magnitude:shell><magnitude:command>pwd</magnitude:command></magnitude:shell><magnitude:yield_user/>'
    v().passes(input)
    const events = parse(input)
    expectNoStructuralError(events)
  })

  it('07: known structural message remains valid', () => {
    const input = '<magnitude:message>hello</magnitude:message><magnitude:yield_user/>'
    v().passes(input)
    const events = parse(input)
    expectNoStructuralError(events)
  })
})
