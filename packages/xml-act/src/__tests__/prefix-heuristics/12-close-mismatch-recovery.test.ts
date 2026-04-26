/**
 * Rule: mismatched magnitude: close tags recover only with strong runtime evidence.
 */
import { describe, expect, it } from 'vitest'
import {
  grammarValidator,
  parse,
  expectStructuralError,
  expectNoStructuralError,
  expectPreservedInMessage,
  expectPreservedInLens,
  collectMessageChunks,
  collectLensChunks,
  countEvents,
  getToolInput,
} from './helpers'

const v = () => grammarValidator()

describe('prefix heuristics: close mismatch recovery', () => {
  it('01: message closed by reason close on newline', () => {
    const input = `<magnitude:message>hello
</magnitude:reason>
<magnitude:message>next</magnitude:message><magnitude:yield_user/>`
    v().rejects(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expect(collectMessageChunks(events)).toContain('hello')
    expect(collectMessageChunks(events)).toContain('next')
    expect(countEvents(events, 'MessageChunk')).toBeGreaterThan(1)
  })

  it('02: message closed by invoke close on newline', () => {
    const input = `<magnitude:message>hello
</magnitude:invoke>
<magnitude:message>next</magnitude:message><magnitude:yield_user/>`
    v().rejects(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expect(collectMessageChunks(events)).toContain('hello')
    expect(collectMessageChunks(events)).toContain('next')
    expect(countEvents(events, 'MessageChunk')).toBeGreaterThan(1)
  })

  it('03: reason closed by message close on newline', () => {
    const input = `<magnitude:reason>thinking
</magnitude:message>
<magnitude:invoke tool="tree"></magnitude:invoke><magnitude:yield_user/>`
    v().rejects(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expect(collectLensChunks(events)).toContain('thinking')
    expect(getToolInput(events)).toEqual({})
  })

  it('04: reason closed by parameter close on newline', () => {
    const input = `<magnitude:reason>thinking
</magnitude:parameter>
<magnitude:yield_user/>`
    v().rejects(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expect(collectLensChunks(events)).toContain('thinking')
  })

  it('05: shell parameter closed by filter close', () => {
    const input = `<magnitude:invoke tool="shell"><magnitude:parameter name="command">pwd
</magnitude:filter></magnitude:invoke><magnitude:yield_user/>`
    v().rejects(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expect(getToolInput(events)).toEqual({ command: 'pwd\n' })
  })

  it('06: command alias closed by reason close', () => {
    const input = `<magnitude:invoke tool="shell"><magnitude:command>pwd
</magnitude:reason></magnitude:invoke><magnitude:yield_user/>`
    v().rejects(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expect(getToolInput(events)).toEqual({ command: 'pwd\n' })
  })

  it('07: filter closed by parameter close before command alias', () => {
    const input = `<magnitude:invoke tool="shell"><magnitude:filter>*.ts
</magnitude:parameter><magnitude:command>pwd</magnitude:command></magnitude:invoke><magnitude:yield_user/>`
    v().rejects(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expect(getToolInput(events)).toEqual({ command: 'pwd' })
  })

  it('08: tool alias invoke body closed by mismatched reason close', () => {
    const input = `<magnitude:shell><magnitude:command>pwd</magnitude:command>
</magnitude:reason><magnitude:message>done</magnitude:message><magnitude:yield_user/>`
    v().rejects(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expect(getToolInput(events)).toEqual({ command: 'pwd' })
    expect(collectMessageChunks(events)).toContain('done')
  })

  it('09: same-line message mismatch is ambiguous', () => {
    const input = '<magnitude:message>hello</magnitude:reason> world</magnitude:message><magnitude:yield_user/>'
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

  it('10: same-line reason mismatch is ambiguous', () => {
    const input = '<magnitude:reason>hello</magnitude:message> world</magnitude:reason><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectStructuralError(events, {
      variant: 'AmbiguousMagnitudeClose',
      tagName: 'magnitude:message',
      expectedTagName: 'magnitude:reason',
      detailIncludes: ['</magnitude:message>', 'magnitude:reason'],
    })
    expectPreservedInLens(events, '</magnitude:message>')
  })

  it('11: same-line parameter mismatch is ambiguous', () => {
    const input = '<magnitude:invoke tool="shell"><magnitude:parameter name="command">pwd</magnitude:filter> later</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectStructuralError(events, {
      variant: 'AmbiguousMagnitudeClose',
      tagName: 'magnitude:filter',
      expectedTagName: 'magnitude:parameter',
      detailIncludes: ['</magnitude:filter>', 'magnitude:parameter'],
    })
    expect(getToolInput(events)).toEqual({ command: 'pwd</magnitude:filter> later' })
  })

  it('12: same-line filter mismatch is ambiguous', () => {
    const input = '<magnitude:invoke tool="shell"><magnitude:filter>*.ts</magnitude:parameter> later</magnitude:filter></magnitude:invoke><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectStructuralError(events, {
      variant: 'AmbiguousMagnitudeClose',
      tagName: 'magnitude:parameter',
      expectedTagName: 'magnitude:filter',
      detailIncludes: ['</magnitude:parameter>', 'magnitude:filter'],
    })
  })

  it('13: exact close remains valid control', () => {
    const input = '<magnitude:message>hello</magnitude:message><magnitude:yield_user/>'
    v().passes(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expect(collectMessageChunks(events)).toBe('hello')
  })

  it('14: stray unknown close does not use recovery', () => {
    const input = '</magnitude:foo><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectStructuralError(events, {
      variant: 'StrayCloseTag',
      tagName: 'magnitude:foo',
      detailIncludes: ['</magnitude:foo>'],
    })
  })

  it('15: repeated newline mismatch recovery across two messages', () => {
    const input = `<magnitude:message>hello
</magnitude:reason>
<magnitude:message>next
</magnitude:invoke>
<magnitude:yield_user/>`
    v().rejects(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expect(collectMessageChunks(events)).toContain('hello')
    expect(collectMessageChunks(events)).toContain('next')
    expect(countEvents(events, 'MessageChunk')).toBeGreaterThan(1)
  })

  it('16: escape open inside message is invalid, so inner mismatched close is not escaped', () => {
    const input = '<magnitude:message><magnitude:escape></magnitude:reason></magnitude:escape></magnitude:message><magnitude:yield_user/>'
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
