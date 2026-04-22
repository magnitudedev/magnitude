/**
 * Category 2: Constrained Tool Names
 *
 * Grammar enumerates known tool names. Unknown tool names are rejected.
 * Parser parses any tool name but emits errors for unknown ones.
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator, parse, hasEvent, getToolInputs, YIELD,
  SHELL_TOOL_DEF, EDIT_TOOL_DEF, TREE_TOOL_DEF,
} from './helpers'
import { buildValidator } from '../../grammar/__tests__/helpers'
import type { GrammarToolDef } from '../../grammar/grammar-builder'

const v = () => grammarValidator()

describe('constrained tool names', () => {
  it('01: known tool "shell" accepted', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">ls</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    const events = parse(input)
    expect(hasEvent(events, 'ToolInputReady')).toBe(true)
    expect(getToolInputs(events)[0].input).toEqual({ command: 'ls' })
  })

  it('02: known tool "edit" accepted', () => {
    const input = `<invoke tool="edit">\n<parameter name="path">f</parameter><parameter name="old">x</parameter><parameter name="new">y</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    const events = parse(input)
    expect(hasEvent(events, 'ToolInputReady')).toBe(true)
    expect(getToolInputs(events)[0].input).toEqual({ path: 'f', old: 'x', new: 'y' })
  })

  it('03: known tool "tree" (0 params) accepted', () => {
    const input = `<invoke tool="tree">\n</invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'ToolInputReady')).toBe(true)
  })

  it('04: unknown tool name rejected by grammar', () => {
    const input = `<invoke tool="unknown">\n</invoke><${YIELD.slice(1)}`
    v().rejects(input)
    // Parser still handles it (emits UnknownTool error)
    const events = parse(input)
    expect(hasEvent(events, 'StructuralParseError')).toBe(true)
  })

  it('05: similar-to-known tool name rejected', () => {
    const input = `<invoke tool="shells">\n<parameter name="command">ls</parameter></invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('06: empty tool name rejected', () => {
    const input = `<invoke tool="">\n</invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('07: case-sensitive — "Shell" rejected', () => {
    const input = `<invoke tool="Shell">\n<parameter name="command">ls</parameter></invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('08: namespaced tool accepted when in schema', () => {
    const nsTool: GrammarToolDef = { tagName: 'fs:read', parameters: [{ name: 'path', field: 'path', type: 'scalar' }] }
    const nv = buildValidator([nsTool])
    const input = `<invoke tool="fs:read">\n<parameter name="path">/tmp</parameter></invoke><${YIELD.slice(1)}`
    nv.passes(input)
  })

  it('09: multiple different tools in one turn', () => {
    const input =
      `<invoke tool="shell">\n<parameter name="command">ls</parameter></invoke>` +
      `<invoke tool="tree">\n</invoke>` +
      `<invoke tool="edit">\n<parameter name="path">f</parameter><parameter name="old">x</parameter><parameter name="new">y</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    const events = parse(input)
    expect(getToolInputs(events).length).toBe(3)
  })

  it('10: same tool used multiple times', () => {
    const input =
      `<invoke tool="shell">\n<parameter name="command">ls</parameter></invoke>` +
      `<invoke tool="shell">\n<parameter name="command">pwd</parameter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    const inputs = getToolInputs(parse(input))
    expect(inputs.length).toBe(2)
    expect(inputs[0].input).toEqual({ command: 'ls' })
    expect(inputs[1].input).toEqual({ command: 'pwd' })
  })

  it('11: tool name with extra space rejected', () => {
    const input = `<invoke tool=" shell">\n<parameter name="command">ls</parameter></invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })
})
