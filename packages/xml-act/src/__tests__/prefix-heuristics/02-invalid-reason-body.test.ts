/**
 * Rule: inside <magnitude:think>, any nested magnitude: open tag is invalid.
 */
import { describe, expect, it } from 'vitest'
import {
  grammarValidator,
  grepGrammarValidator,
  parse,
  parseWithGrep,
  expectStructuralError,
  getStructuralErrors,
  expectPreservedInLens,
} from './helpers'

const v = () => grammarValidator()

const invalidCases = [
  {
    label: '01: invoke open inside think',
    input: '<magnitude:think>before <magnitude:invoke tool="shell"></magnitude:invoke> after</magnitude:think><magnitude:yield_user/>',
    raw: '<magnitude:invoke tool="shell">',
  },
  {
    label: '02: message open inside think',
    input: '<magnitude:think>before <magnitude:message>nested</magnitude:message> after</magnitude:think><magnitude:yield_user/>',
    raw: '<magnitude:message>',
  },
  {
    label: '03: think open inside think',
    input: '<magnitude:think>before <magnitude:think>nested</magnitude:think> after</magnitude:think><magnitude:yield_user/>',
    raw: '<magnitude:think>',
    minStructuralErrors: 1,
  },
  {
    label: '04: parameter open inside think',
    input: '<magnitude:think>before <magnitude:parameter name="command">pwd</magnitude:parameter> after</magnitude:think><magnitude:yield_user/>',
    raw: '<magnitude:parameter name="command">',
  },
  {
    label: '05: filter open inside think',
    input: '<magnitude:think>before <magnitude:filter>*.ts</magnitude:filter> after</magnitude:think><magnitude:yield_user/>',
    raw: '<magnitude:filter>',
  },
  {
    label: '06: shell alias inside think',
    input: '<magnitude:think>before <magnitude:shell><magnitude:command>pwd</magnitude:command></magnitude:shell> after</magnitude:think><magnitude:yield_user/>',
    raw: '<magnitude:shell>',
  },
  {
    label: '07: edit alias inside think',
    input: '<magnitude:think>before <magnitude:edit><magnitude:path>a.ts</magnitude:path><magnitude:old>x</magnitude:old><magnitude:new>y</magnitude:new></magnitude:edit> after</magnitude:think><magnitude:yield_user/>',
    raw: '<magnitude:edit>',
  },
  {
    label: '08: tree alias inside think',
    input: '<magnitude:think>before <magnitude:tree></magnitude:tree> after</magnitude:think><magnitude:yield_user/>',
    raw: '<magnitude:tree>',
  },
  {
    label: '09: command alias inside think',
    input: '<magnitude:think>before <magnitude:command>pwd</magnitude:command> after</magnitude:think><magnitude:yield_user/>',
    raw: '<magnitude:command>',
  },
  {
    label: '10: unknown prefixed open inside think',
    input: '<magnitude:think>before <magnitude:foo>bar</magnitude:foo> after</magnitude:think><magnitude:yield_user/>',
    raw: '<magnitude:foo>',
  },
] as const

describe('prefix heuristics: invalid think body', () => {
  for (const testCase of invalidCases) {
    it(testCase.label, () => {
      v().rejects(testCase.input)
      const events = parse(testCase.input)
      if ('minStructuralErrors' in testCase) {
        const structuralErrors = getStructuralErrors(events)
        expect(structuralErrors.length).toBeGreaterThanOrEqual(
          testCase.minStructuralErrors
        )
        const error = structuralErrors[0].error as any
        expect(error._tag).toBe('InvalidMagnitudeOpen')
        expect(error.tagName).toBe(testCase.raw.slice(1, -1).split(' ')[0])
        expect(error.parentTagName).toBe('magnitude:think')
        for (const snippet of [testCase.raw, 'magnitude:think']) {
          expect(String(error.detail ?? '')).toContain(snippet)
        }
      } else {
        expectStructuralError(events, {
          variant: 'InvalidMagnitudeOpen',
          tagName: testCase.raw.slice(1, -1).split(' ')[0],
          parentTagName: 'magnitude:think',
          detailIncludes: [testCase.raw, 'magnitude:think'],
        })
      }
      expectPreservedInLens(events, testCase.raw)
    })
  }

  it('11: escape open inside think is invalid', () => {
    const input = '<magnitude:think><magnitude:escape><magnitude:message>literal</magnitude:message></magnitude:escape></magnitude:think><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectStructuralError(events, {
      variant: 'InvalidMagnitudeOpen',
      tagName: 'magnitude:escape',
      parentTagName: 'magnitude:think',
      detailIncludes: ['<magnitude:escape>', 'magnitude:think'],
    })
    expectPreservedInLens(events, '<magnitude:escape>')
  })

  it('12: escape open inside think is invalid with grep toolset too', () => {
    const input = '<magnitude:think><magnitude:escape><magnitude:grep><magnitude:pattern>TODO</magnitude:pattern></magnitude:grep></magnitude:escape></magnitude:think><magnitude:yield_user/>'
    grepGrammarValidator().rejects(input)
    const events = parseWithGrep(input)
    expectStructuralError(events, {
      variant: 'InvalidMagnitudeOpen',
      tagName: 'magnitude:escape',
      parentTagName: 'magnitude:think',
      detailIncludes: ['<magnitude:escape>', 'magnitude:think'],
    })
    expectPreservedInLens(events, '<magnitude:escape>')
  })
})
