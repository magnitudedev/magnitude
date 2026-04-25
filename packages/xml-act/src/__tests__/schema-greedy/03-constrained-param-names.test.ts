/**
 * Category 3: Constrained Parameter Names
 *
 * Per-tool parameter names are constrained to the known schema.
 * Any ordering accepted. Duplicates are now rejected by the grammar.
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator, parse, hasEvent, getToolInputs, getToolInput, YIELD,
} from './helpers'

const v = () => grammarValidator()

describe('constrained parameter names', () => {
  it('01: known param "command" for shell accepted', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))).toEqual({ command: 'ls' })
  })

  it('02: unknown param name for shell rejected by grammar', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="cmd">ls</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
    // Parser emits error for unknown param
    expect(hasEvent(parse(input), 'ToolParseError')).toBe(true)
  })

  it('03: all edit params accepted (path, old, new)', () => {
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:parameter><magnitude:parameter name="old">x</magnitude:parameter><magnitude:parameter name="new">y</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))).toEqual({ path: 'f', old: 'x', new: 'y' })
  })

  it('04: edit params in reverse order accepted', () => {
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="new">y</magnitude:parameter><magnitude:parameter name="old">x</magnitude:parameter><magnitude:parameter name="path">f</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))).toEqual({ path: 'f', old: 'x', new: 'y' })
  })

  it('05: edit params in random order accepted', () => {
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="old">x</magnitude:parameter><magnitude:parameter name="path">f</magnitude:parameter><magnitude:parameter name="new">y</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))).toEqual({ path: 'f', old: 'x', new: 'y' })
  })

  it('06: unknown param name for edit rejected', () => {
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="file">f</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('07: param for 0-param tool rejected', () => {
    const input = `<magnitude:invoke tool="tree">\n<magnitude:parameter name="path">x</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('08: duplicate param name rejected by grammar', () => {
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">a</magnitude:parameter><magnitude:parameter name="path">b</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('09: shell param name in edit context rejected', () => {
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="command">x</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('10: edit param name in shell context rejected', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="path">x</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('11: empty param name rejected', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="">x</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('12: case-sensitive param name rejected', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="Command">x</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })
})
