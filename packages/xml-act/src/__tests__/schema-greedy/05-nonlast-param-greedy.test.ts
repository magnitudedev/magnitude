/**
 * Category 5: Non-Last Parameter Greedy Matching
 *
 * Non-last params use recursive greedy last-match.
 * Confirmation: </magnitude:parameter> + ws + next valid invoke child
 * (constrained param open, filter, or invoke close).
 * False </magnitude:parameter> in content is absorbed by the buc loop.
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator, parse, hasEvent, getToolInput, getToolInputs,
  countEvents, collectMessageChunks, YIELD,
} from './helpers'

const v = () => grammarValidator()

describe('non-last parameter greedy matching', () => {
  it('01: </magnitude:parameter> confirmed by next <magnitude:parameter> (immediate)', () => {
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">foo.ts</magnitude:parameter><magnitude:parameter name="old">x</magnitude:parameter><magnitude:parameter name="new">y</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    const ti = getToolInput(parse(input))
    expect(ti?.path).toBe('foo.ts')
    expect(ti?.old).toBe('x')
  })

  it('02: </magnitude:parameter> confirmed by next <magnitude:parameter> after whitespace', () => {
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">foo</magnitude:parameter>\n<magnitude:parameter name="old">x</magnitude:parameter><magnitude:parameter name="new">y</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.path).toBe('foo')
  })

  it('03: </magnitude:parameter> confirmed by next <magnitude:parameter> after multiple newlines', () => {
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">foo</magnitude:parameter>\n\n\n<magnitude:parameter name="old">x</magnitude:parameter><magnitude:parameter name="new">y</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.path).toBe('foo')
  })

  it('04: </magnitude:parameter> confirmed by <magnitude:filter>', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter><magnitude:filter>$.stdout</magnitude:filter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('ls')
  })

  it('05: </magnitude:parameter> confirmed by </magnitude:invoke> (early close)', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">foo.ts</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('foo.ts')
  })

  it('06: false </magnitude:parameter> followed by text — absorbed as content', () => {
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">file</magnitude:parameter>xxx<magnitude:parameter name="old">x</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    // "file</magnitude:parameter>xxx" is all content of path param — false close absorbed
    // Then real </magnitude:parameter> confirmed by <magnitude:parameter name="old">
    // Actually: depends on whether "xxx" breaks the continuation. Let me think...
    // After </magnitude:parameter>, grammar tries continuation. "xxx" doesn't match <magnitude:parameter, <magnitude:filter, or </magnitude:invoke>
    // So structural path fails, content loop absorbs </magnitude:parameter>xxx as content
    // But then we need another </magnitude:parameter> to close path. The next one is before <magnitude:parameter name="old">
    // Hmm, "file</magnitude:parameter>xxx" — the BUC matches "file", then </magnitude:parameter> is the close,
    // then continuation tries to match "xxx<magnitude:parameter..." — "xxx" doesn't match ws + valid child.
    // So the content loop takes over: </magnitude:parameter> is absorbed, "xxx" continues in BUC.
    // Then the NEXT </magnitude:parameter> (before <magnitude:parameter name="old">) — wait, there's only one more </magnitude:parameter>.
    // Let me re-read: "file</magnitude:parameter>xxx<magnitude:parameter name="old">x</magnitude:parameter>"
    // The BUC for path starts. "file" is BUC. Then </magnitude:parameter>. Two paths:
    //   1. Content: absorb </magnitude:parameter>, continue BUC with "xxx<magnitude:parameter name="old">x"
    //      Then another </magnitude:parameter>. Two paths again:
    //        1a. Content: absorb, continue BUC... but no more </magnitude:parameter> and no continuation. Dead end.
    //        1b. Structural: </magnitude:parameter> + continuation. What follows? </magnitude:invoke>< — that's valid!
    //   2. Structural: </magnitude:parameter> + continuation. What follows? "xxx<magnitude:parameter..." — "xxx" is not ws.
    //      Fails.
    // So path 1 → 1b wins. Path param content = "file</magnitude:parameter>xxx<magnitude:parameter name="old">x"
    // That means old param never opens! This test is wrong.
    // Let me fix: the content should not contain another param's content.
  })

  it('06-fixed: false </magnitude:parameter> followed by non-ws — absorbed as content, real close later', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo "</magnitude:parameter>"; echo done</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    const ti = getToolInput(parse(input))
    expect(ti?.command).toBe('echo "</magnitude:parameter>"; echo done')
  })

  it('07: multiple false </magnitude:parameter> in content before real one', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">a</magnitude:parameter>b</magnitude:parameter>c</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    const ti = getToolInput(parse(input))
    expect(ti?.command).toBe('a</magnitude:parameter>b</magnitude:parameter>c')
  })

  it('08: </magnitude:parameter> with tabs before next param', () => {
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:parameter>\t\t<magnitude:parameter name="old">x</magnitude:parameter><magnitude:parameter name="new">y</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.path).toBe('f')
  })

  it('09: </magnitude:parameter> followed by unknown tag — absorbed as content', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">a</magnitude:parameter><div>b</div></magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    const ti = getToolInput(parse(input))
    expect(ti?.command).toBe('a</magnitude:parameter><div>b</div>')
  })

  it('10: content with HTML-like tags inside parameter', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo "<div>hello</div>"</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('echo "<div>hello</div>"')
  })

  it('11: content with other XML close tags (not </magnitude:parameter>)', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">cat </magnitude:invoke> </magnitude:filter> </magnitude:message></magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('cat </magnitude:invoke> </magnitude:filter> </magnitude:message>')
  })

  it('12: </magnitude:parameter> followed by invalid param name — absorbed as content', () => {
    // </magnitude:parameter><magnitude:parameter name="wrong"> — "wrong" is not a valid shell param
    // Grammar rejects structural path, content loop absorbs
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">text</magnitude:parameter><magnitude:parameter name="invalid">more</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    const ti = getToolInput(parse(input))
    expect(ti?.command).toBe('text</magnitude:parameter><magnitude:parameter name="invalid">more')
  })
})
