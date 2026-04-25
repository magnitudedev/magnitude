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
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    const events = parse(input)
    expect(hasEvent(events, 'ToolInputReady')).toBe(true)
    expect(getToolInputs(events)[0].input).toEqual({ command: 'ls' })
  })

  it('02: known tool "edit" accepted', () => {
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:parameter><magnitude:parameter name="old">x</magnitude:parameter><magnitude:parameter name="new">y</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    const events = parse(input)
    expect(hasEvent(events, 'ToolInputReady')).toBe(true)
    expect(getToolInputs(events)[0].input).toEqual({ path: 'f', old: 'x', new: 'y' })
  })

  it('03: known tool "tree" (0 params) accepted', () => {
    const input = `<magnitude:invoke tool="tree">\n</magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(hasEvent(parse(input), 'ToolInputReady')).toBe(true)
  })

  it('04: unknown tool name rejected by grammar', () => {
    const input = `<magnitude:invoke tool="unknown">\n</magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
    // Parser still handles it (emits UnknownTool error)
    const events = parse(input)
    expect(hasEvent(events, 'StructuralParseError')).toBe(true)
  })

  it('05: similar-to-known tool name rejected', () => {
    const input = `<magnitude:invoke tool="shells">\n<magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('06: empty tool name rejected', () => {
    const input = `<magnitude:invoke tool="">\n</magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('07: case-sensitive — "Shell" rejected', () => {
    const input = `<magnitude:invoke tool="Shell">\n<magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('08: namespaced tool accepted when in schema', () => {
    const nsTool: GrammarToolDef = { tagName: 'fs:read', parameters: [{ name: 'path', field: 'path', type: 'scalar', required: true }] }
    const nv = buildValidator([nsTool])
    const input = `<magnitude:invoke tool="fs:read">\n<magnitude:parameter name="path">/tmp</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    nv.passes(input)
  })

  it('09: multiple different tools in one turn', () => {
    const input =
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke>` +
      `<magnitude:invoke tool="tree">\n</magnitude:invoke>` +
      `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:parameter><magnitude:parameter name="old">x</magnitude:parameter><magnitude:parameter name="new">y</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    const events = parse(input)
    expect(getToolInputs(events).length).toBe(3)
  })

  it('10: same tool used multiple times', () => {
    const input =
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke>` +
      `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">pwd</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().passes(input)
    const inputs = getToolInputs(parse(input))
    expect(inputs.length).toBe(2)
    expect(inputs[0].input).toEqual({ command: 'ls' })
    expect(inputs[1].input).toEqual({ command: 'pwd' })
  })

  it('11: tool name with extra space rejected', () => {
    const input = `<magnitude:invoke tool=" shell">\n<magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })
})
