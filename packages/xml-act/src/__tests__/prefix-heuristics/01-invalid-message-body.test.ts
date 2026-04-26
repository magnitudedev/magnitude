/**
 * Rule: inside <magnitude:message>, any nested magnitude: open tag is invalid.
 */
import { describe, expect, it } from 'vitest'
import {
  grammarValidator,
  parse,
  expectStructuralError,
  getStructuralErrors,
  expectPreservedInMessage,
} from './helpers'

const v = () => grammarValidator()

const invalidCases = [
  {
    label: '01: invoke open inside message',
    input: '<magnitude:message>before <magnitude:invoke tool="shell"></magnitude:invoke> after</magnitude:message><magnitude:yield_user/>',
    raw: '<magnitude:invoke tool="shell">',
  },
  {
    label: '02: message open inside message',
    input: '<magnitude:message>before <magnitude:message>nested</magnitude:message> after</magnitude:message><magnitude:yield_user/>',
    raw: '<magnitude:message>',
    minStructuralErrors: 1,
  },
  {
    label: '03: think open inside message',
    input: '<magnitude:message>before <magnitude:think>nested</magnitude:think> after</magnitude:message><magnitude:yield_user/>',
    raw: '<magnitude:think>',
  },
  {
    label: '04: parameter open inside message',
    input: '<magnitude:message>before <magnitude:parameter name="command">pwd</magnitude:parameter> after</magnitude:message><magnitude:yield_user/>',
    raw: '<magnitude:parameter name="command">',
  },
  {
    label: '05: filter open inside message',
    input: '<magnitude:message>before <magnitude:filter>*.ts</magnitude:filter> after</magnitude:message><magnitude:yield_user/>',
    raw: '<magnitude:filter>',
  },
  {
    label: '06: shell alias inside message',
    input: '<magnitude:message>before <magnitude:shell><magnitude:command>pwd</magnitude:command></magnitude:shell> after</magnitude:message><magnitude:yield_user/>',
    raw: '<magnitude:shell>',
  },
  {
    label: '07: edit alias inside message',
    input: '<magnitude:message>before <magnitude:edit><magnitude:path>a.ts</magnitude:path><magnitude:old>x</magnitude:old><magnitude:new>y</magnitude:new></magnitude:edit> after</magnitude:message><magnitude:yield_user/>',
    raw: '<magnitude:edit>',
  },
  {
    label: '08: tree alias inside message',
    input: '<magnitude:message>before <magnitude:tree></magnitude:tree> after</magnitude:message><magnitude:yield_user/>',
    raw: '<magnitude:tree>',
  },
  {
    label: '09: command alias inside message',
    input: '<magnitude:message>before <magnitude:command>pwd</magnitude:command> after</magnitude:message><magnitude:yield_user/>',
    raw: '<magnitude:command>',
  },
  {
    label: '10: unknown prefixed open inside message',
    input: '<magnitude:message>before <magnitude:foo>bar</magnitude:foo> after</magnitude:message><magnitude:yield_user/>',
    raw: '<magnitude:foo>',
  },
] as const

describe('prefix heuristics: invalid message body', () => {
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
        expect(error.parentTagName).toBe('magnitude:message')
        for (const snippet of [testCase.raw, 'magnitude:message']) {
          expect(String(error.detail ?? '')).toContain(snippet)
        }
      } else {
        expectStructuralError(events, {
          variant: 'InvalidMagnitudeOpen',
          tagName: testCase.raw.slice(1, -1).split(' ')[0],
          parentTagName: 'magnitude:message',
          detailIncludes: [testCase.raw, 'magnitude:message'],
        })
      }
      expectPreservedInMessage(events, testCase.raw)
    })
  }

  it('11: escape open inside message is invalid', () => {
    const input = '<magnitude:message>before <magnitude:escape><magnitude:shell><magnitude:command>pwd</magnitude:command></magnitude:shell></magnitude:escape> after</magnitude:message><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectStructuralError(events, {
      variant: 'InvalidMagnitudeOpen',
      tagName: 'magnitude:escape',
      parentTagName: 'magnitude:message',
      detailIncludes: ['<magnitude:escape>', 'magnitude:message'],
    })
    expectPreservedInMessage(events, '<magnitude:escape>')
  })

  it('12: escape-wrapped yield-looking syntax is invalid', () => {
    const input = '<magnitude:message><magnitude:escape><magnitude:yield_user/></magnitude:escape></magnitude:message><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectStructuralError(events, {
      variant: 'InvalidMagnitudeOpen',
      tagName: 'magnitude:escape',
      parentTagName: 'magnitude:message',
      detailIncludes: ['<magnitude:escape>', 'magnitude:message'],
    })
    expectPreservedInMessage(events, '<magnitude:escape>')
  })
})
