/**
 * Rule: inside parameter/filter bodies, nested magnitude: opens are invalid.
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator,
  parse,
  expectStructuralError,
  expectNoStructuralError,
  getStructuralErrors,
} from './helpers'

const v = () => grammarValidator()

describe('prefix heuristics: invalid parameter/filter body', () => {
  const invalidCases = [
    {
      label: '01: invoke open inside parameter body',
      input: '<magnitude:invoke tool="shell"><magnitude:parameter name="command">pre <magnitude:invoke tool="shell"></magnitude:invoke> post</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>',
      raw: '<magnitude:invoke tool="shell">',
      parentTagName: 'magnitude:parameter',
    },
    {
      label: '02: message open inside parameter body',
      input: '<magnitude:invoke tool="shell"><magnitude:parameter name="command">pre <magnitude:message>x</magnitude:message> post</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>',
      raw: '<magnitude:message>',
      parentTagName: 'magnitude:parameter',
    },
    {
      label: '03: shell alias inside parameter body',
      input: '<magnitude:invoke tool="shell"><magnitude:parameter name="command">pre <magnitude:shell></magnitude:shell> post</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>',
      raw: '<magnitude:shell>',
      parentTagName: 'magnitude:parameter',
    },
    {
      label: '04: unknown prefixed open inside parameter body',
      input: '<magnitude:invoke tool="shell"><magnitude:parameter name="command">pre <magnitude:foo>bar</magnitude:foo> post</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>',
      raw: '<magnitude:foo>',
      parentTagName: 'magnitude:parameter',
    },
    {
      label: '06: parameter open inside filter body',
      input: '<magnitude:invoke tool="shell"><magnitude:filter>pre <magnitude:parameter name="command">x</magnitude:parameter> post</magnitude:filter></magnitude:invoke><magnitude:yield_user/>',
      raw: '<magnitude:parameter name="command">',
      parentTagName: 'magnitude:filter',
    },
    {
      label: '07: think open inside filter body',
      input: '<magnitude:invoke tool="shell"><magnitude:filter>pre <magnitude:think>x</magnitude:think> post</magnitude:filter></magnitude:invoke><magnitude:yield_user/>',
      raw: '<magnitude:think>',
      parentTagName: 'magnitude:filter',
    },
    {
      label: '08: shell alias inside filter body',
      input: '<magnitude:invoke tool="shell"><magnitude:filter>pre <magnitude:shell></magnitude:shell> post</magnitude:filter></magnitude:invoke><magnitude:yield_user/>',
      raw: '<magnitude:shell>',
      parentTagName: 'magnitude:filter',
    },
    {
      label: '09: command alias inside filter body',
      input: '<magnitude:invoke tool="shell"><magnitude:filter>pre <magnitude:command>pwd</magnitude:command> post</magnitude:filter></magnitude:invoke><magnitude:yield_user/>',
      raw: '<magnitude:command>',
      parentTagName: 'magnitude:filter',
    },
    {
      label: '10: unknown prefixed open inside filter body',
      input: '<magnitude:invoke tool="shell"><magnitude:filter>pre <magnitude:foo>bar</magnitude:foo> post</magnitude:filter></magnitude:invoke><magnitude:yield_user/>',
      raw: '<magnitude:foo>',
      parentTagName: 'magnitude:filter',
    },
  ] as const

  for (const testCase of invalidCases) {
    it(testCase.label, () => {
      v().rejects(testCase.input)
      const events = parse(testCase.input)
      expectStructuralError(events, {
        variant: 'InvalidMagnitudeOpen',
        tagName: testCase.raw.slice(1, -1).split(' ')[0],
        parentTagName: testCase.parentTagName,
        detailIncludes: [testCase.raw, testCase.parentTagName],
      })
    })
  }

  it('05: escape open inside parameter body is invalid', () => {
    const input = '<magnitude:invoke tool="shell"><magnitude:parameter name="command">pre <magnitude:escape><magnitude:shell><magnitude:command>pwd</magnitude:command></magnitude:shell></magnitude:escape> post</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    // Multiple structural errors: escape open + nested tags inside parameter
    const errors = getStructuralErrors(events)
    expect(errors.length).toBeGreaterThanOrEqual(1)
    const escapeError = errors.find((e: any) => e.error._tag === 'InvalidMagnitudeOpen' && e.error.tagName === 'magnitude:escape')
    expect(escapeError).toBeDefined()
  })

  it('11: escape open inside filter body is invalid', () => {
    const input = '<magnitude:invoke tool="shell"><magnitude:filter><magnitude:escape><magnitude:message>literal</magnitude:message></magnitude:escape></magnitude:filter></magnitude:invoke><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectStructuralError(events, {
      variant: 'InvalidMagnitudeOpen',
      tagName: 'magnitude:escape',
      parentTagName: 'magnitude:filter',
      detailIncludes: ['<magnitude:escape>', 'magnitude:filter'],
    })
  })

  it('12: escape-wrapped unknown prefixed markup inside filter is invalid', () => {
    const input = '<magnitude:invoke tool="shell"><magnitude:filter><magnitude:escape><magnitude:foo>bar</magnitude:foo></magnitude:escape></magnitude:filter></magnitude:invoke><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectStructuralError(events, {
      variant: 'InvalidMagnitudeOpen',
      tagName: 'magnitude:escape',
      parentTagName: 'magnitude:filter',
      detailIncludes: ['<magnitude:escape>', 'magnitude:filter'],
    })
  })
})
