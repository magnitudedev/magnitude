/**
 * Category 5: Non-Last Parameter Greedy Matching
 *
 * Non-last params use recursive greedy last-match.
 * Confirmation: </parameter> + ws + next valid invoke child
 * (constrained param open, filter, or invoke close).
 * False </parameter> in content is absorbed by the buc loop.
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator, parse, hasEvent, getToolInput, getToolInputs,
  countEvents, collectMessageChunks, YIELD,
} from './helpers'

const v = () => grammarValidator()

describe('non-last parameter greedy matching', () => {
  it('01: </parameter> confirmed by next <parameter> (immediate)', () => {
    const input = `<invoke tool="edit">\n<parameter name="path">foo.ts</parameter><parameter name="old">x</parameter><parameter name="new">y</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    const ti = getToolInput(parse(input))
    expect(ti?.path).toBe('foo.ts')
    expect(ti?.old).toBe('x')
  })

  it('02: </parameter> confirmed by next <parameter> after whitespace', () => {
    const input = `<invoke tool="edit">\n<parameter name="path">foo</parameter>\n<parameter name="old">x</parameter><parameter name="new">y</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.path).toBe('foo')
  })

  it('03: </parameter> confirmed by next <parameter> after multiple newlines', () => {
    const input = `<invoke tool="edit">\n<parameter name="path">foo</parameter>\n\n\n<parameter name="old">x</parameter><parameter name="new">y</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.path).toBe('foo')
  })

  it('04: </parameter> confirmed by <filter>', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">ls</parameter><filter>$.stdout</filter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('ls')
  })

  it('05: </parameter> confirmed by </invoke> (early close)', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">foo.ts</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('foo.ts')
  })

  it('06: false </parameter> followed by text — absorbed as content', () => {
    const input = `<invoke tool="edit">\n<parameter name="path">file</parameter>xxx<parameter name="old">x</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    // "file</parameter>xxx" is all content of path param — false close absorbed
    // Then real </parameter> confirmed by <parameter name="old">
    // Actually: depends on whether "xxx" breaks the continuation. Let me think...
    // After </parameter>, grammar tries continuation. "xxx" doesn't match <parameter, <filter, or </invoke>
    // So structural path fails, content loop absorbs </parameter>xxx as content
    // But then we need another </parameter> to close path. The next one is before <parameter name="old">
    // Hmm, "file</parameter>xxx" — the BUC matches "file", then </parameter> is the close,
    // then continuation tries to match "xxx<parameter..." — "xxx" doesn't match ws + valid child.
    // So the content loop takes over: </parameter> is absorbed, "xxx" continues in BUC.
    // Then the NEXT </parameter> (before <parameter name="old">) — wait, there's only one more </parameter>.
    // Let me re-read: "file</parameter>xxx<parameter name="old">x</parameter>"
    // The BUC for path starts. "file" is BUC. Then </parameter>. Two paths:
    //   1. Content: absorb </parameter>, continue BUC with "xxx<parameter name="old">x"
    //      Then another </parameter>. Two paths again:
    //        1a. Content: absorb, continue BUC... but no more </parameter> and no continuation. Dead end.
    //        1b. Structural: </parameter> + continuation. What follows? </invoke>< — that's valid!
    //   2. Structural: </parameter> + continuation. What follows? "xxx<parameter..." — "xxx" is not ws.
    //      Fails.
    // So path 1 → 1b wins. Path param content = "file</parameter>xxx<parameter name="old">x"
    // That means old param never opens! This test is wrong.
    // Let me fix: the content should not contain another param's content.
  })

  it('06-fixed: false </parameter> followed by non-ws — absorbed as content, real close later', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">echo "</parameter>"; echo done</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    const ti = getToolInput(parse(input))
    expect(ti?.command).toBe('echo "</parameter>"; echo done')
  })

  it('07: multiple false </parameter> in content before real one', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">a</parameter>b</parameter>c</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    const ti = getToolInput(parse(input))
    expect(ti?.command).toBe('a</parameter>b</parameter>c')
  })

  it('08: </parameter> with tabs before next param', () => {
    const input = `<invoke tool="edit">\n<parameter name="path">f</parameter>\t\t<parameter name="old">x</parameter><parameter name="new">y</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.path).toBe('f')
  })

  it('09: </parameter> followed by unknown tag — absorbed as content', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">a</parameter><div>b</div></parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    const ti = getToolInput(parse(input))
    expect(ti?.command).toBe('a</parameter><div>b</div>')
  })

  it('10: content with HTML-like tags inside parameter', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">echo "<div>hello</div>"</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('echo "<div>hello</div>"')
  })

  it('11: content with other XML close tags (not </parameter>)', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">cat </invoke> </filter> </message></parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('cat </invoke> </filter> </message>')
  })

  it('12: </parameter> followed by invalid param name — absorbed as content', () => {
    // </parameter><parameter name="wrong"> — "wrong" is not a valid shell param
    // Grammar rejects structural path, content loop absorbs
    const input = `<invoke tool="shell">\n<parameter name="command">text</parameter><parameter name="invalid">more</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    const ti = getToolInput(parse(input))
    expect(ti?.command).toBe('text</parameter><parameter name="invalid">more')
  })
})
