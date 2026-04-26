/**
 * Category 5: Non-Last Parameter Greedy Matching
 *
 * Non-last params use recursive greedy last-match.
 * Confirmation: </magnitude:parameter> + ws + next valid invoke child.
 * Under the stricter contract, malformed or invalid `<magnitude:` continuations are rejected,
 * not absorbed as body content.
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

  it('04: </magnitude:parameter> followed by filter path is rejected in current grammar', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter><magnitude:filter>$.stdout</magnitude:filter></magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('05: </magnitude:parameter> confirmed by </magnitude:invoke> (early close)', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">foo.ts</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('foo.ts')
  })

  it('06: false </magnitude:parameter> followed by text and then structural open is rejected', () => {
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">file</magnitude:parameter>xxx<magnitude:parameter name="old">x</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('06-fixed: false </magnitude:parameter> followed by non-ws rejects under first-close-wins', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo "</magnitude:parameter>"; echo done</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('07: multiple false </magnitude:parameter> in content before real one rejects', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">a</magnitude:parameter>b</magnitude:parameter>c</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('08: </magnitude:parameter> with tabs before next param', () => {
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:parameter>\t\t<magnitude:parameter name="old">x</magnitude:parameter><magnitude:parameter name="new">y</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.path).toBe('f')
  })

  it('09: </magnitude:parameter> followed by unknown tag rejects under first-close-wins', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">a</magnitude:parameter><div>b</div></magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('10: content with HTML-like tags inside parameter', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo "<div>hello</div>"</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('echo "<div>hello</div>"')
  })

  it('11: content with magnitude close tags — rejected (magnitude closes must be escaped)', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">cat </magnitude:invoke> </magnitude:filter> </magnitude:message></magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('12: </magnitude:parameter> followed by invalid param name is rejected', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">text</magnitude:parameter><magnitude:parameter name="invalid">more</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })
})
