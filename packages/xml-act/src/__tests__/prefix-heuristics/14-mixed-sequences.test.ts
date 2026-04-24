/**
 * Rule: aliases, fail-fast, recovery, escape, and canonical forms compose in full turns.
 */
import { describe, expect, it } from 'vitest'
import {
  grammarValidator,
  grepGrammarValidator,
  parse,
  parseWithGrep,
  expectStructuralError,
  expectNoStructuralError,
  expectPreservedInMessage,
  collectLensChunks,
  collectMessageChunks,
  getToolInputs,
  getToolInput,
} from './helpers'

const v = () => grammarValidator()

describe('prefix heuristics: mixed sequences', () => {
  it('01: reason then shell alias then message', () => {
    const input = '<magnitude:reason>prep</magnitude:reason><magnitude:shell><magnitude:command>pwd</magnitude:command></magnitude:shell><magnitude:message>done</magnitude:message><magnitude:yield_user/>'
    v().passes(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expect(collectLensChunks(events)).toContain('prep')
    expect(getToolInput(events)).toEqual({ command: 'pwd' })
    expect(collectMessageChunks(events)).toContain('done')
  })

  it('02: message then canonical invoke then tree alias', () => {
    const input = '<magnitude:message>prep</magnitude:message><magnitude:invoke tool="shell"><magnitude:command>pwd</magnitude:command></magnitude:invoke><magnitude:tree></magnitude:tree><magnitude:yield_user/>'
    v().passes(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expect(collectMessageChunks(events)).toContain('prep')
    expect(getToolInputs(events).map(x => x.input)).toEqual([{ command: 'pwd' }, {}])
  })

  it('03: top-level unknown between valid items', () => {
    const input = '<magnitude:reason>prep</magnitude:reason><magnitude:foo>bar</magnitude:foo><magnitude:shell><magnitude:command>pwd</magnitude:command></magnitude:shell><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectStructuralError(events, {
      variant: 'InvalidMagnitudeOpen',
      tagName: 'magnitude:foo',
      detailIncludes: ['<magnitude:foo>'],
    })
  })

  it('04: message body invalid shell alias preserves text', () => {
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

  it('05: recovered message close followed by shell alias', () => {
    const input = `<magnitude:message>hello
</magnitude:reason>
<magnitude:shell><magnitude:command>pwd</magnitude:command></magnitude:shell><magnitude:yield_user/>`
    v().rejects(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expect(collectMessageChunks(events)).toContain('hello')
    expect(getToolInput(events)).toEqual({ command: 'pwd' })
  })

  it('06: recovered command alias close inside shell alias invoke', () => {
    const input = `<magnitude:shell><magnitude:command>pwd
</magnitude:message></magnitude:shell><magnitude:message>done</magnitude:message><magnitude:yield_user/>`
    v().rejects(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expect(getToolInput(events)).toEqual({ command: 'pwd\n' })
    expect(collectMessageChunks(events)).toContain('done')
  })

  it('07: escaped literal unknown plus canonical invoke', () => {
    const input = '<magnitude:message><magnitude:escape><magnitude:foo>bar</magnitude:foo></magnitude:escape></magnitude:message><magnitude:invoke tool="tree"></magnitude:invoke><magnitude:yield_user/>'
    v().passes(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expectPreservedInMessage(events, '<magnitude:foo>')
    expect(getToolInput(events)).toEqual({})
  })

  it('08: mixed grep alias then edit canonical', () => {
    const input = '<magnitude:grep><magnitude:pattern>TODO</magnitude:pattern></magnitude:grep><magnitude:invoke tool="edit"><magnitude:path>a.ts</magnitude:path><magnitude:old>x</magnitude:old><magnitude:new>y</magnitude:new></magnitude:invoke><magnitude:yield_user/>'
    grepGrammarValidator().rejects(input)
    const events = parseWithGrep(input)
    expectNoStructuralError(events)
    expect(getToolInput(events)).toEqual({ pattern: 'TODO' })
  })

  it('09: ambiguous same-line mismatch followed by valid message', () => {
    const input = '<magnitude:message>hello</magnitude:reason> world</magnitude:message><magnitude:message>next</magnitude:message><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectStructuralError(events, {
      variant: 'AmbiguousMagnitudeClose',
      tagName: 'magnitude:reason',
      expectedTagName: 'magnitude:message',
      detailIncludes: ['</magnitude:reason>', 'magnitude:message'],
    })
    expectPreservedInMessage(events, '</magnitude:reason>')
  })

  it('10: reason then tree alias then escaped close-like text in final message', () => {
    const input = '<magnitude:reason>prep</magnitude:reason><magnitude:tree></magnitude:tree><magnitude:message><magnitude:escape></magnitude:reason></magnitude:escape></magnitude:message><magnitude:yield_user/>'
    v().passes(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expect(collectLensChunks(events)).toContain('prep')
    expect(getToolInputs(events).map(x => x.input)).toEqual([{}])
    expectPreservedInMessage(events, '</magnitude:reason>')
  })
})
