/**
 * Category 3: Constrained Parameter Names
 *
 * Per-tool parameter names are constrained to the known schema.
 * Any ordering accepted. Duplicates accepted by grammar (parser validates).
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator, parse, hasEvent, getToolInputs, getToolInput, YIELD,
} from './helpers'

const v = () => grammarValidator()

describe('constrained parameter names', () => {
  it('01: known param "command" for shell accepted', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">ls</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))).toEqual({ command: 'ls' })
  })

  it('02: unknown param name for shell rejected by grammar', () => {
    const input = `<invoke tool="shell">\n<parameter name="cmd">ls</parameter></invoke><${YIELD.slice(1)}`
    v().rejects(input)
    // Parser emits error for unknown param
    expect(hasEvent(parse(input), 'ToolParseError')).toBe(true)
  })

  it('03: all edit params accepted (path, old, new)', () => {
    const input = `<invoke tool="edit">\n<parameter name="path">f</parameter><parameter name="old">x</parameter><parameter name="new">y</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))).toEqual({ path: 'f', old: 'x', new: 'y' })
  })

  it('04: edit params in reverse order accepted', () => {
    const input = `<invoke tool="edit">\n<parameter name="new">y</parameter><parameter name="old">x</parameter><parameter name="path">f</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))).toEqual({ path: 'f', old: 'x', new: 'y' })
  })

  it('05: edit params in random order accepted', () => {
    const input = `<invoke tool="edit">\n<parameter name="old">x</parameter><parameter name="path">f</parameter><parameter name="new">y</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))).toEqual({ path: 'f', old: 'x', new: 'y' })
  })

  it('06: unknown param name for edit rejected', () => {
    const input = `<invoke tool="edit">\n<parameter name="file">f</parameter></invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('07: param for 0-param tool rejected', () => {
    const input = `<invoke tool="tree">\n<parameter name="path">x</parameter></invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('08: duplicate param name accepted by grammar', () => {
    const input = `<invoke tool="edit">\n<parameter name="path">a</parameter><parameter name="path">b</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    // Parser uses last value or flags duplicate — implementation detail
  })

  it('09: shell param name in edit context rejected', () => {
    const input = `<invoke tool="edit">\n<parameter name="command">x</parameter></invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('10: edit param name in shell context rejected', () => {
    const input = `<invoke tool="shell">\n<parameter name="path">x</parameter></invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('11: empty param name rejected', () => {
    const input = `<invoke tool="shell">\n<parameter name="">x</parameter></invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('12: case-sensitive param name rejected', () => {
    const input = `<invoke tool="shell">\n<parameter name="Command">x</parameter></invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })
})
