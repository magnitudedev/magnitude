/**
 * Rule: inside <magnitude:reason>, any nested magnitude: open tag is invalid.
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
    label: '01: invoke open inside reason',
    input: '<magnitude:reason>before <magnitude:invoke tool="shell"></magnitude:invoke> after</magnitude:reason><magnitude:yield_user/>',
    raw: '<magnitude:invoke tool="shell">',
  },
  {
    label: '02: message open inside reason',
    input: '<magnitude:reason>before <magnitude:message>nested</magnitude:message> after</magnitude:reason><magnitude:yield_user/>',
    raw: '<magnitude:message>',
  },
  {
    label: '03: reason open inside reason',
    input: '<magnitude:reason>before <magnitude:reason>nested</magnitude:reason> after</magnitude:reason><magnitude:yield_user/>',
    raw: '<magnitude:reason>',
    minStructuralErrors: 1,
  },
  {
    label: '04: parameter open inside reason',
    input: '<magnitude:reason>before <magnitude:parameter name="command">pwd</magnitude:parameter> after</magnitude:reason><magnitude:yield_user/>',
    raw: '<magnitude:parameter name="command">',
  },
  {
    label: '05: filter open inside reason',
    input: '<magnitude:reason>before <magnitude:filter>*.ts</magnitude:filter> after</magnitude:reason><magnitude:yield_user/>',
    raw: '<magnitude:filter>',
  },
  {
    label: '06: shell alias inside reason',
    input: '<magnitude:reason>before <magnitude:shell><magnitude:command>pwd</magnitude:command></magnitude:shell> after</magnitude:reason><magnitude:yield_user/>',
    raw: '<magnitude:shell>',
  },
  {
    label: '07: edit alias inside reason',
    input: '<magnitude:reason>before <magnitude:edit><magnitude:path>a.ts</magnitude:path><magnitude:old>x</magnitude:old><magnitude:new>y</magnitude:new></magnitude:edit> after</magnitude:reason><magnitude:yield_user/>',
    raw: '<magnitude:edit>',
  },
  {
    label: '08: tree alias inside reason',
    input: '<magnitude:reason>before <magnitude:tree></magnitude:tree> after</magnitude:reason><magnitude:yield_user/>',
    raw: '<magnitude:tree>',
  },
  {
    label: '09: command alias inside reason',
    input: '<magnitude:reason>before <magnitude:command>pwd</magnitude:command> after</magnitude:reason><magnitude:yield_user/>',
    raw: '<magnitude:command>',
  },
  {
    label: '10: unknown prefixed open inside reason',
    input: '<magnitude:reason>before <magnitude:foo>bar</magnitude:foo> after</magnitude:reason><magnitude:yield_user/>',
    raw: '<magnitude:foo>',
  },
] as const

describe('prefix heuristics: invalid reason body', () => {
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
        expect(error.parentTagName).toBe('magnitude:reason')
        for (const snippet of [testCase.raw, 'magnitude:reason']) {
          expect(String(error.detail ?? '')).toContain(snippet)
        }
      } else {
        expectStructuralError(events, {
          variant: 'InvalidMagnitudeOpen',
          tagName: testCase.raw.slice(1, -1).split(' ')[0],
          parentTagName: 'magnitude:reason',
          detailIncludes: [testCase.raw, 'magnitude:reason'],
        })
      }
      expectPreservedInLens(events, testCase.raw)
    })
  }

  it('11: escape open inside reason is invalid', () => {
    const input = '<magnitude:reason><magnitude:escape><magnitude:message>literal</magnitude:message></magnitude:escape></magnitude:reason><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectStructuralError(events, {
      variant: 'InvalidMagnitudeOpen',
      tagName: 'magnitude:escape',
      parentTagName: 'magnitude:reason',
      detailIncludes: ['<magnitude:escape>', 'magnitude:reason'],
    })
    expectPreservedInLens(events, '<magnitude:escape>')
  })

  it('12: escape open inside reason is invalid with grep toolset too', () => {
    const input = '<magnitude:reason><magnitude:escape><magnitude:grep><magnitude:pattern>TODO</magnitude:pattern></magnitude:grep></magnitude:escape></magnitude:reason><magnitude:yield_user/>'
    grepGrammarValidator().rejects(input)
    const events = parseWithGrep(input)
    expectStructuralError(events, {
      variant: 'InvalidMagnitudeOpen',
      tagName: 'magnitude:escape',
      parentTagName: 'magnitude:reason',
      detailIncludes: ['<magnitude:escape>', 'magnitude:reason'],
    })
    expectPreservedInLens(events, '<magnitude:escape>')
  })
})
