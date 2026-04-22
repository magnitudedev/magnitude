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
  it('01: param contains </parameter> as text', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">echo "</parameter>"</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('echo "</parameter>"')
  })

  it('02: param contains </parameter></invoke> as text', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">The sequence </parameter></invoke> ends it</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('The sequence </parameter></invoke> ends it')
  })

  it('03: param contains <parameter name="..."> as text', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">Use <parameter name="foo"> for params</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('Use <parameter name="foo"> for params')
  })

  it('04: param contains full tool call example as text', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">echo '<invoke tool="shell"><parameter name="command">ls</parameter></invoke>'</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toContain('<invoke tool="shell">')
  })

  it('05: param contains close sequence minus final < (not confirmed)', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">text</parameter></invoke>\nstill content</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toContain('text')
  })

  it('06: param content with nested XML', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">cat << 'EOF'\n<root><child attr="val">text</child></root>\nEOF</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toContain('<root>')
  })

  it('07: param contains </parameter> followed by invalid param name', () => {
    // In shell context (1 param), this is the last param — deep confirmation needed
    // </parameter><parameter name="invalid"> — "invalid" is not valid for shell
    // Also <parameter name="invalid"> doesn't match </invoke>, so structural path fails
    // Content loop absorbs it
    const input = `<invoke tool="shell">\n<parameter name="command">text</parameter><parameter name="invalid">more</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('text</parameter><parameter name="invalid">more')
  })

  it('08: param body is the string "</parameter>"', () => {
    const input = `<invoke tool="shell">\n<parameter name="command"></parameter></parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('</parameter>')
  })

  it('09: param contains multiple interleaved false sequences', () => {
    const input = `<invoke tool="shell">\n<parameter name="command"></parameter></invoke></parameter></invoke></parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('</parameter></invoke></parameter></invoke>')
  })

  it('10: edit non-last param contains </parameter><parameter name="wrong">', () => {
    // "wrong" is not a valid edit param, so greedy matching rejects structural path
    const input = `<invoke tool="edit">\n<parameter name="path">a</parameter><parameter name="wrong">b</parameter><parameter name="old">c</parameter><parameter name="new">d</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    // path content should include the false close + invalid param as content
    expect(getToolInput(parse(input))?.path).toBe('a</parameter><parameter name="wrong">b')
    expect(getToolInput(parse(input))?.old).toBe('c')
  })
})
