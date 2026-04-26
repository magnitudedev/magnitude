/**
 * Rule: parameter aliases remain strict and tool-specific.
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator,
  grepGrammarValidator,
  parse,
  parseWithGrep,
  expectStructuralError,
  expectNoStructuralError,
  getToolInput,
} from './helpers'

const v = () => grammarValidator()
const vg = () => grepGrammarValidator()

describe('prefix heuristics: parameter alias edge cases', () => {
  it('01: unknown foo alias inside shell invoke', () => {
    const input = '<magnitude:invoke tool="shell"><magnitude:foo>bar</magnitude:foo></magnitude:invoke><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectStructuralError(events, {
      variant: 'InvalidMagnitudeOpen',
      tagName: 'magnitude:foo',
      parentTagName: 'magnitude:invoke',
      detailIncludes: ['<magnitude:foo>', 'magnitude:invoke'],
    })
  })

  it('02: edit-only path alias inside shell invoke', () => {
    const input = '<magnitude:invoke tool="shell"><magnitude:path>a.ts</magnitude:path></magnitude:invoke><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectStructuralError(events, {
      variant: 'InvalidMagnitudeOpen',
      tagName: 'magnitude:path',
      parentTagName: 'magnitude:invoke',
      detailIncludes: ['<magnitude:path>', 'magnitude:invoke'],
    })
  })

  it('03: shell command alias inside edit invoke', () => {
    const input = '<magnitude:invoke tool="edit"><magnitude:command>pwd</magnitude:command></magnitude:invoke><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectStructuralError(events, {
      variant: 'InvalidMagnitudeOpen',
      tagName: 'magnitude:command',
      parentTagName: 'magnitude:invoke',
      detailIncludes: ['<magnitude:command>', 'magnitude:invoke'],
    })
  })

  it('04: grep glob alias inside shell invoke', () => {
    const input = '<magnitude:invoke tool="shell"><magnitude:glob>*.ts</magnitude:glob></magnitude:invoke><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectStructuralError(events, {
      variant: 'InvalidMagnitudeOpen',
      tagName: 'magnitude:glob',
      parentTagName: 'magnitude:invoke',
      detailIncludes: ['<magnitude:glob>', 'magnitude:invoke'],
    })
  })

  it('05: duplicate field alias then canonical rejected by grammar', () => {
    const input = '<magnitude:invoke tool="shell"><magnitude:command>pwd</magnitude:command><magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>'
    v().rejects(input)
  })

  it('06: duplicate field canonical then alias rejected by grammar', () => {
    const input = '<magnitude:invoke tool="shell"><magnitude:parameter name="command">pwd</magnitude:parameter><magnitude:command>ls</magnitude:command></magnitude:invoke><magnitude:yield_user/>'
    v().rejects(input)
  })

  it('07: edit aliases out of canonical order', () => {
    const input = '<magnitude:invoke tool="edit"><magnitude:new>y</magnitude:new><magnitude:path>a.ts</magnitude:path><magnitude:old>x</magnitude:old></magnitude:invoke><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectNoStructuralError(events)
    expect(getToolInput(events)).toEqual({ new: 'y', path: 'a.ts', old: 'x' })
  })

  it('08: grep mixed aliases and canonical params', () => {
    const input = '<magnitude:invoke tool="grep"><magnitude:pattern>TODO</magnitude:pattern><magnitude:parameter name="glob">*.ts</magnitude:parameter><magnitude:path>src</magnitude:path></magnitude:invoke><magnitude:yield_user/>'
    vg().rejects(input)
    const events = parseWithGrep(input)
    expectNoStructuralError(events)
    expect(getToolInput(events)).toEqual({ pattern: 'TODO', glob: '*.ts', path: 'src' })
  })

  it('09: unknown alias with attrs inside invoke', () => {
    const input = '<magnitude:invoke tool="edit"><magnitude:foo x="1">bar</magnitude:foo></magnitude:invoke><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectStructuralError(events, {
      variant: 'InvalidMagnitudeOpen',
      tagName: 'magnitude:foo',
      parentTagName: 'magnitude:invoke',
      detailIncludes: ['<magnitude:foo x="1">', 'magnitude:invoke'],
    })
  })

  it('10: self-closing command alias rejected', () => {
    const input = '<magnitude:invoke tool="shell"><magnitude:command/></magnitude:invoke><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectStructuralError(events, {
      variant: 'InvalidMagnitudeOpen',
      tagName: 'magnitude:command',
      parentTagName: 'magnitude:invoke',
      detailIncludes: ['magnitude:command'],
    })
  })

  it('11: self-closing pattern alias rejected', () => {
    const input = '<magnitude:invoke tool="grep"><magnitude:pattern/></magnitude:invoke><magnitude:yield_user/>'
    vg().rejects(input)
    const events = parseWithGrep(input)
    expectStructuralError(events, {
      variant: 'InvalidMagnitudeOpen',
      tagName: 'magnitude:pattern',
      parentTagName: 'magnitude:invoke',
      detailIncludes: ['magnitude:pattern'],
    })
  })

  it('12: escape open inside parameter body is invalid', () => {
    const input = '<magnitude:invoke tool="shell"><magnitude:parameter name="command"><magnitude:escape><magnitude:path>a.ts</magnitude:path></magnitude:escape></magnitude:parameter></magnitude:invoke><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectStructuralError(events, {
      variant: 'InvalidMagnitudeOpen',
      tagName: 'magnitude:escape',
      parentTagName: 'magnitude:parameter',
      detailIncludes: ['<magnitude:escape>', 'magnitude:parameter'],
    })
  })
})
