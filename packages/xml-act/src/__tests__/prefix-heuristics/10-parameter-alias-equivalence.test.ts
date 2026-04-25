/**
 * Rule: parser parameter aliases are semantically equivalent to canonical parameter forms,
 * even though the grammar no longer generates alias syntax.
 */
import { describe, it } from 'vitest'
import {
  grammarValidator,
  grepGrammarValidator,
  expectParameterAliasEquivalent,
  parseWithGrep,
} from './helpers'

const v = () => grammarValidator()
const vg = () => grepGrammarValidator()

describe('prefix heuristics: parameter alias equivalence', () => {
  const defaultCases = [
    {
      label: '01: shell command alias equals canonical',
      aliasInput: '<magnitude:invoke tool="shell"><magnitude:command>pwd</magnitude:command></magnitude:invoke><magnitude:yield_user/>',
      canonicalInput: '<magnitude:invoke tool="shell"><magnitude:parameter name="command">pwd</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>',
    },
    {
      label: '02: edit aliases equal canonical',
      aliasInput: '<magnitude:invoke tool="edit"><magnitude:path>a.ts</magnitude:path><magnitude:old>x</magnitude:old><magnitude:new>y</magnitude:new></magnitude:invoke><magnitude:yield_user/>',
      canonicalInput: '<magnitude:invoke tool="edit"><magnitude:parameter name="path">a.ts</magnitude:parameter><magnitude:parameter name="old">x</magnitude:parameter><magnitude:parameter name="new">y</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>',
    },
    {
      label: '03: mixed edit alias/canonical equals fully canonical',
      aliasInput: '<magnitude:invoke tool="edit"><magnitude:path>a.ts</magnitude:path><magnitude:parameter name="old">x</magnitude:parameter><magnitude:new>y</magnitude:new></magnitude:invoke><magnitude:yield_user/>',
      canonicalInput: '<magnitude:invoke tool="edit"><magnitude:parameter name="path">a.ts</magnitude:parameter><magnitude:parameter name="old">x</magnitude:parameter><magnitude:parameter name="new">y</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>',
    },
    {
      label: '06: alias equivalence with preceding filter',
      aliasInput: '<magnitude:invoke tool="shell"><magnitude:filter>*.ts</magnitude:filter><magnitude:command>pwd</magnitude:command></magnitude:invoke><magnitude:yield_user/>',
      canonicalInput: '<magnitude:invoke tool="shell"><magnitude:filter>*.ts</magnitude:filter><magnitude:parameter name="command">pwd</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>',
    },
    {
      label: '07: top-level shell alias + command alias equals canonical invoke',
      aliasInput: '<magnitude:shell><magnitude:command>pwd</magnitude:command></magnitude:shell><magnitude:yield_user/>',
      canonicalInput: '<magnitude:invoke tool="shell"><magnitude:parameter name="command">pwd</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>',
    },
    {
      label: '08: long scalar alias body equals canonical',
      aliasInput: '<magnitude:invoke tool="shell"><magnitude:command>printf "a\nb\nc"</magnitude:command></magnitude:invoke><magnitude:yield_user/>',
      canonicalInput: '<magnitude:invoke tool="shell"><magnitude:parameter name="command">printf "a\nb\nc"</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>',
    },
    {
      label: '09: alias equivalence across two invokes',
      aliasInput: '<magnitude:invoke tool="edit"><magnitude:path>a.ts</magnitude:path><magnitude:old>x</magnitude:old><magnitude:new>y</magnitude:new></magnitude:invoke><magnitude:invoke tool="shell"><magnitude:command>pwd</magnitude:command></magnitude:invoke><magnitude:yield_user/>',
      canonicalInput: '<magnitude:invoke tool="edit"><magnitude:parameter name="path">a.ts</magnitude:parameter><magnitude:parameter name="old">x</magnitude:parameter><magnitude:parameter name="new">y</magnitude:parameter></magnitude:invoke><magnitude:invoke tool="shell"><magnitude:parameter name="command">pwd</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>',
    },
  ] as const

  for (const testCase of defaultCases) {
    it(testCase.label, () => {
      v().rejects(testCase.aliasInput)
      v().passes(testCase.canonicalInput)
      expectParameterAliasEquivalent(testCase.aliasInput, testCase.canonicalInput)
    })
  }

  it('04: grep pattern alias equals canonical', () => {
    const aliasInput = '<magnitude:invoke tool="grep"><magnitude:pattern>TODO</magnitude:pattern></magnitude:invoke><magnitude:yield_user/>'
    const canonicalInput = '<magnitude:invoke tool="grep"><magnitude:parameter name="pattern">TODO</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>'
    vg().rejects(aliasInput)
    expectParameterAliasEquivalent(aliasInput, canonicalInput, parseWithGrep)
  })

  it('05: grep full alias payload equals canonical', () => {
    const aliasInput = '<magnitude:invoke tool="grep"><magnitude:pattern>TODO</magnitude:pattern><magnitude:glob>*.ts</magnitude:glob><magnitude:path>src</magnitude:path><magnitude:limit>10</magnitude:limit></magnitude:invoke><magnitude:yield_user/>'
    const canonicalInput = '<magnitude:invoke tool="grep"><magnitude:parameter name="pattern">TODO</magnitude:parameter><magnitude:parameter name="glob">*.ts</magnitude:parameter><magnitude:parameter name="path">src</magnitude:parameter><magnitude:parameter name="limit">10</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>'
    vg().rejects(aliasInput)
    vg().passes(canonicalInput)
    expectParameterAliasEquivalent(aliasInput, canonicalInput, parseWithGrep)
  })

  it('10: grep alias equivalence with optional params omitted', () => {
    const aliasInput = '<magnitude:invoke tool="grep"><magnitude:pattern>TODO</magnitude:pattern></magnitude:invoke><magnitude:yield_user/>'
    const canonicalInput = '<magnitude:invoke tool="grep"><magnitude:parameter name="pattern">TODO</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>'
    vg().rejects(aliasInput)
    expectParameterAliasEquivalent(aliasInput, canonicalInput, parseWithGrep)
  })
})
