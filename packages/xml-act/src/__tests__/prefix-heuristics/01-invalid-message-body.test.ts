/**
 * Rule: inside <magnitude:message>, any nested magnitude: open tag other than
 * <magnitude:escape> is invalid.
 */
import { describe, it } from 'vitest'
import {
  grammarValidator,
  parse,
  expectStructuralError,
  expectNoStructuralError,
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
  },
  {
    label: '03: reason open inside message',
    input: '<magnitude:message>before <magnitude:reason>nested</magnitude:reason> after</magnitude:message><magnitude:yield_user/>',
    raw: '<magnitude:reason>',
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
      expectStructuralError(events, {
        variant: 'InvalidMagnitudeOpen',
        tagName: testCase.raw.slice(1, -1).split(' ')[0],
        parentTagName: 'magnitude:message',
        detailIncludes: [testCase.raw, 'magnitude:message', 'magnitude:escape'],
      })
      expectPreservedInMessage(events, testCase.raw)
    })
  }

  it('11: escape is allowed', () => {
    const input = '<magnitude:message>before <magnitude:escape><magnitude:shell><magnitude:command>pwd</magnitude:command></magnitude:shell></magnitude:escape> after</magnitude:message><magnitude:yield_user/>'
    v().passes(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expectPreservedInMessage(events, '<magnitude:shell>')
  })

  it('12: escaped yield-looking syntax remains literal', () => {
    const input = '<magnitude:message><magnitude:escape><magnitude:yield_user/></magnitude:escape></magnitude:message><magnitude:yield_user/>'
    v().passes(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expectPreservedInMessage(events, '<magnitude:yield_user/>')
  })
})
