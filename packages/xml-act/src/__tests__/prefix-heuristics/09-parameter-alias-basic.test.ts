/**
 * Rule: grammar is canonical-only inside invoke, but parser parameter aliases still resolve.
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator,
  grepGrammarValidator,
  parse,
  parseWithGrep,
  expectNoStructuralError,
  expectStructuralError,
  getToolInput,
} from './helpers'

const v = () => grammarValidator()
const vg = () => grepGrammarValidator()

describe('prefix heuristics: parameter alias basic', () => {
  const defaultPositiveCases = [
    {
      label: '01: shell command alias',
      input: '<magnitude:invoke tool="shell"><magnitude:command>pwd</magnitude:command></magnitude:invoke><magnitude:yield_user/>',
      expected: { command: 'pwd' },
    },
    {
      label: '02: shell command alias with filter',
      input: '<magnitude:invoke tool="shell"><magnitude:filter>*.ts</magnitude:filter><magnitude:command>pwd</magnitude:command></magnitude:invoke><magnitude:yield_user/>',
      expected: { command: 'pwd' },
    },
    {
      label: '03: edit path/old/new aliases',
      input: '<magnitude:invoke tool="edit"><magnitude:path>a.ts</magnitude:path><magnitude:old>x</magnitude:old><magnitude:new>y</magnitude:new></magnitude:invoke><magnitude:yield_user/>',
      expected: { path: 'a.ts', old: 'x', new: 'y' },
    },
    {
      label: '04: edit mixed alias and canonical',
      input: '<magnitude:invoke tool="edit"><magnitude:path>a.ts</magnitude:path><magnitude:parameter name="old">x</magnitude:parameter><magnitude:new>y</magnitude:new></magnitude:invoke><magnitude:yield_user/>',
      expected: { path: 'a.ts', old: 'x', new: 'y' },
    },
    {
      label: '09: canonical parameters remain valid control',
      input: '<magnitude:invoke tool="edit"><magnitude:parameter name="path">a.ts</magnitude:parameter><magnitude:parameter name="old">x</magnitude:parameter><magnitude:parameter name="new">y</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>',
      expected: { path: 'a.ts', old: 'x', new: 'y' },
    },
    {
      label: '10: alias after canonical',
      input: '<magnitude:invoke tool="edit"><magnitude:parameter name="path">a.ts</magnitude:parameter><magnitude:old>x</magnitude:old><magnitude:new>y</magnitude:new></magnitude:invoke><magnitude:yield_user/>',
      expected: { path: 'a.ts', old: 'x', new: 'y' },
    },
    {
      label: '11: alias before canonical',
      input: '<magnitude:invoke tool="edit"><magnitude:path>a.ts</magnitude:path><magnitude:parameter name="old">x</magnitude:parameter><magnitude:parameter name="new">y</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>',
      expected: { path: 'a.ts', old: 'x', new: 'y' },
    },
    {
      label: '12: top-level shell alias plus param alias composition',
      input: '<magnitude:shell><magnitude:command>pwd</magnitude:command></magnitude:shell><magnitude:yield_user/>',
      expected: { command: 'pwd' },
    },
  ] as const

  for (const testCase of defaultPositiveCases) {
    it(testCase.label, () => {
      if (testCase.label === '09: canonical parameters remain valid control') {
        v().passes(testCase.input)
      } else {
        v().rejects(testCase.input)
      }
      const events = parse(testCase.input)
      expectNoStructuralError(events)
      expect(getToolInput(events)).toEqual(testCase.expected)
    })
  }

  it('05: tree has no valid param aliases', () => {
    const input = '<magnitude:invoke tool="tree"><magnitude:path>a.ts</magnitude:path></magnitude:invoke><magnitude:yield_user/>'
    v().rejects(input)
    const events = parse(input)
    expectStructuralError(events, {
      variant: 'InvalidMagnitudeOpen',
      tagName: 'magnitude:path',
      parentTagName: 'magnitude:invoke',
      detailIncludes: ['<magnitude:path>', 'magnitude:invoke'],
    })
  })

  it('06: grep pattern alias', () => {
    const input = '<magnitude:invoke tool="grep"><magnitude:pattern>TODO</magnitude:pattern></magnitude:invoke><magnitude:yield_user/>'
    vg().rejects(input)
    const events = parseWithGrep(input)
    expectNoStructuralError(events)
    expect(getToolInput(events)).toEqual({ pattern: 'TODO' })
  })

  it('07: grep optional alias glob', () => {
    const input = '<magnitude:invoke tool="grep"><magnitude:pattern>TODO</magnitude:pattern><magnitude:glob>*.ts</magnitude:glob></magnitude:invoke><magnitude:yield_user/>'
    vg().rejects(input)
    const events = parseWithGrep(input)
    expectNoStructuralError(events)
    expect(getToolInput(events)).toEqual({ pattern: 'TODO', glob: '*.ts' })
  })

  it('08: grep full alias payload', () => {
    const input = '<magnitude:invoke tool="grep"><magnitude:pattern>TODO</magnitude:pattern><magnitude:glob>*.ts</magnitude:glob><magnitude:path>src</magnitude:path><magnitude:limit>10</magnitude:limit></magnitude:invoke><magnitude:yield_user/>'
    vg().rejects(input)
    const events = parseWithGrep(input)
    expectNoStructuralError(events)
    expect(getToolInput(events)).toEqual({
      pattern: 'TODO',
      glob: '*.ts',
      path: 'src',
      limit: '10',
    })
  })
})
