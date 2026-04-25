/**
 * Rule: grammar is canonical-only, but parser alias heuristics still resolve tool aliases.
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator,
  grepGrammarValidator,
  parse,
  parseWithGrep,
  expectNoStructuralError,
  getToolInput,
} from './helpers'

const v = () => grammarValidator()
const vg = () => grepGrammarValidator()

describe('prefix heuristics: tool alias basic', () => {
  const defaultCases = [
    {
      label: '01: shell alias with command alias',
      input: '<magnitude:shell><magnitude:command>pwd</magnitude:command></magnitude:shell><magnitude:yield_user/>',
      expected: { command: 'pwd' },
    },
    {
      label: '02: shell alias with canonical parameter',
      input: '<magnitude:shell><magnitude:parameter name="command">pwd</magnitude:parameter></magnitude:shell><magnitude:yield_user/>',
      expected: { command: 'pwd' },
    },
    {
      label: '03: edit alias with all param aliases',
      input: '<magnitude:edit><magnitude:path>a.ts</magnitude:path><magnitude:old>x</magnitude:old><magnitude:new>y</magnitude:new></magnitude:edit><magnitude:yield_user/>',
      expected: { path: 'a.ts', old: 'x', new: 'y' },
    },
    {
      label: '04: edit alias with canonical params',
      input: '<magnitude:edit><magnitude:parameter name="path">a.ts</magnitude:parameter><magnitude:parameter name="old">x</magnitude:parameter><magnitude:parameter name="new">y</magnitude:parameter></magnitude:edit><magnitude:yield_user/>',
      expected: { path: 'a.ts', old: 'x', new: 'y' },
    },
    {
      label: '05: tree alias zero-param invoke',
      input: '<magnitude:tree></magnitude:tree><magnitude:yield_user/>',
      expected: {},
    },
    {
      label: '06: shell alias after message',
      input: '<magnitude:message>prep</magnitude:message><magnitude:shell><magnitude:command>pwd</magnitude:command></magnitude:shell><magnitude:yield_user/>',
      expected: { command: 'pwd' },
    },
    {
      label: '07: shell alias after reason',
      input: '<magnitude:reason>prep</magnitude:reason><magnitude:shell><magnitude:command>pwd</magnitude:command></magnitude:shell><magnitude:yield_user/>',
      expected: { command: 'pwd' },
    },
    {
      label: '08: shell alias before message',
      input: '<magnitude:shell><magnitude:command>pwd</magnitude:command></magnitude:shell><magnitude:message>done</magnitude:message><magnitude:yield_user/>',
      expected: { command: 'pwd' },
    },
    {
      label: '11: canonical invoke remains valid control',
      input: '<magnitude:invoke tool="shell"><magnitude:command>pwd</magnitude:command></magnitude:invoke><magnitude:yield_user/>',
      expected: { command: 'pwd' },
    },
    {
      label: '12: mixed canonical and alias top-level invokes',
      input: '<magnitude:shell><magnitude:command>pwd</magnitude:command></magnitude:shell><magnitude:invoke tool="tree"></magnitude:invoke><magnitude:yield_user/>',
      expectedInputs: [{ command: 'pwd' }, {}],
    },
  ] as const

  for (const testCase of defaultCases) {
    it(testCase.label, () => {
      v().rejects(testCase.input)
      const events = parse(testCase.input)
      expectNoStructuralError(events)
      if ('expectedInputs' in testCase) {
        expect(getToolInput(events, 0)).toEqual(testCase.expectedInputs[0])
        expect(getToolInput(events, 1)).toEqual(testCase.expectedInputs[1])
      } else {
        expect(getToolInput(events)).toEqual(testCase.expected)
      }
    })
  }

  it('09: grep alias with required param', () => {
    const input = '<magnitude:grep><magnitude:pattern>TODO</magnitude:pattern></magnitude:grep><magnitude:yield_user/>'
    vg().rejects(input)
    const events = parseWithGrep(input)
    expectNoStructuralError(events)
    expect(getToolInput(events)).toEqual({ pattern: 'TODO' })
  })

  it('10: grep alias with all params', () => {
    const input = '<magnitude:grep><magnitude:pattern>TODO</magnitude:pattern><magnitude:glob>*.ts</magnitude:glob><magnitude:path>src</magnitude:path><magnitude:limit>10</magnitude:limit></magnitude:grep><magnitude:yield_user/>'
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
