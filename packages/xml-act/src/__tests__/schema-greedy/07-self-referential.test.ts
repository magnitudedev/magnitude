/**
 * Category 7: Self-Referential Content
 *
 * Parameters containing their own close sequences, full tool call examples,
 * and other adversarial content that could confuse greedy matching.
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator, parse, getToolInput, YIELD,
} from './helpers'

const v = () => grammarValidator()

describe('self-referential content in parameters', () => {
  it('01: param contains </magnitude:parameter> as text', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo "</magnitude:parameter>"</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('echo "</magnitude:parameter>"')
  })

  it('02: param contains </magnitude:parameter></magnitude:invoke> as text', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">The sequence </magnitude:parameter></magnitude:invoke> ends it</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('The sequence </magnitude:parameter></magnitude:invoke> ends it')
  })

  it('03: param contains <magnitude:parameter name="..."> as text', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">Use <magnitude:parameter name="foo"> for params</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('Use <magnitude:parameter name="foo"> for params')
  })

  it('04: param contains full tool call example as text', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo '<magnitude:invoke tool="shell"><magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke>'</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toContain('<magnitude:invoke tool="shell">')
  })

  it('05: param contains close sequence minus final < (not confirmed)', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">text</magnitude:parameter></magnitude:invoke>\nstill content</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toContain('text')
  })

  it('06: param content with nested XML', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">cat << 'EOF'\n<root><child attr="val">text</child></root>\nEOF</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toContain('<root>')
  })

  it('07: param contains </magnitude:parameter> followed by invalid param name', () => {
    // In shell context (1 param), this is the last param — deep confirmation needed
    // </magnitude:parameter><magnitude:parameter name="invalid"> — "invalid" is not valid for shell
    // Also <magnitude:parameter name="invalid"> doesn't match </magnitude:invoke>, so structural path fails
    // Content loop absorbs it
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">text</magnitude:parameter><magnitude:parameter name="invalid">more</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('text</magnitude:parameter><magnitude:parameter name="invalid">more')
  })

  it('08: param body is the string "</magnitude:parameter>"', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command"></magnitude:parameter></magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('</magnitude:parameter>')
  })

  it('09: param contains multiple interleaved false sequences', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command"></magnitude:parameter></magnitude:invoke></magnitude:parameter></magnitude:invoke></magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('</magnitude:parameter></magnitude:invoke></magnitude:parameter></magnitude:invoke>')
  })

  it('10: edit non-last param contains </magnitude:parameter><magnitude:parameter name="wrong">', () => {
    // "wrong" is not a valid edit param, so greedy matching rejects structural path
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">a</magnitude:parameter><magnitude:parameter name="wrong">b</magnitude:parameter><magnitude:parameter name="old">c</magnitude:parameter><magnitude:parameter name="new">d</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    // path content should include the false close + invalid param as content
    expect(getToolInput(parse(input))?.path).toBe('a</magnitude:parameter><magnitude:parameter name="wrong">b')
    expect(getToolInput(parse(input))?.old).toBe('c')
  })
})
