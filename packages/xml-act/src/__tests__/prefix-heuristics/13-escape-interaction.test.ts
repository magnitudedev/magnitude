/**
 * Rule: escape remains the literal opt-out.
 */
import { describe, it } from 'vitest'
import {
  grammarValidator,
  parse,
  expectStructuralError,
  expectNoStructuralError,
  expectPreservedInMessage,
  expectPreservedInLens,
} from './helpers'

const v = () => grammarValidator()

describe('prefix heuristics: escape interaction', () => {
  it('01: literal shell alias in message escape', () => {
    const input = '<magnitude:message><magnitude:escape><magnitude:shell><magnitude:command>pwd</magnitude:command></magnitude:shell></magnitude:escape></magnitude:message><magnitude:yield_user/>'
    v().passes(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expectPreservedInMessage(events, '<magnitude:shell>')
  })

  it('02: literal invoke in reason escape', () => {
    const input = '<magnitude:reason><magnitude:escape><magnitude:invoke tool="shell"></magnitude:invoke></magnitude:escape></magnitude:reason><magnitude:yield_user/>'
    v().passes(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expectPreservedInLens(events, '<magnitude:invoke tool="shell">')
  })

  it('03: literal unknown prefixed tag in message escape', () => {
    const input = '<magnitude:message><magnitude:escape><magnitude:foo>bar</magnitude:foo></magnitude:escape></magnitude:message><magnitude:yield_user/>'
    v().passes(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expectPreservedInMessage(events, '<magnitude:foo>')
  })

  it('04: literal mismatched close in message escape', () => {
    const input = '<magnitude:message><magnitude:escape></magnitude:reason></magnitude:escape></magnitude:message><magnitude:yield_user/>'
    v().passes(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expectPreservedInMessage(events, '</magnitude:reason>')
  })

  it('05: literal param alias in parameter escape', () => {
    const input = '<magnitude:invoke tool="shell"><magnitude:parameter name="command"><magnitude:escape><magnitude:path>a.ts</magnitude:path></magnitude:escape></magnitude:parameter></magnitude:invoke><magnitude:yield_user/>'
    v().passes(input)
    const events = parse(input)
    expectNoStructuralError(events)
  })

  it('06: escaped unknown top-level-like text then valid canonical tree invoke', () => {
    const input = '<magnitude:message><magnitude:escape><magnitude:foo>bar</magnitude:foo></magnitude:escape></magnitude:message><magnitude:invoke tool="tree"></magnitude:invoke><magnitude:yield_user/>'
    v().passes(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expectPreservedInMessage(events, '<magnitude:foo>')
  })

  it('07: escaped yield-looking syntax', () => {
    const input = '<magnitude:reason><magnitude:escape><magnitude:yield_user/></magnitude:escape></magnitude:reason><magnitude:yield_user/>'
    v().passes(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expectPreservedInLens(events, '<magnitude:yield_user/>')
  })

  it('08: escaped canonical parameter inside message', () => {
    const input = '<magnitude:message><magnitude:escape><magnitude:parameter name="command">pwd</magnitude:parameter></magnitude:escape></magnitude:message><magnitude:yield_user/>'
    v().passes(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expectPreservedInMessage(events, '<magnitude:parameter name="command">')
  })

  it('09: escaped mismatch-close-looking tool end', () => {
    const input = '<magnitude:message><magnitude:escape></magnitude:tree></magnitude:escape></magnitude:message><magnitude:yield_user/>'
    v().passes(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expectPreservedInMessage(events, '</magnitude:tree>')
  })

  it('10: control case without escape still errors', () => {
    const input = '<magnitude:message><magnitude:foo>bar</magnitude:foo></magnitude:message><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectStructuralError(events, {
      variant: 'InvalidMagnitudeOpen',
      tagName: 'magnitude:foo',
      parentTagName: 'magnitude:message',
      detailIncludes: ['<magnitude:foo>', 'magnitude:message'],
    })
    expectPreservedInMessage(events, '<magnitude:foo>')
  })
})
