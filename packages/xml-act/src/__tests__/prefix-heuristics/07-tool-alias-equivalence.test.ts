/**
 * Rule: tool aliases are semantically equivalent to canonical invoke forms.
 */
import { describe, it } from 'vitest'
import {
  grammarValidator,
  grepGrammarValidator,
  expectToolAliasEquivalent,
  parseWithGrep,
} from './helpers'

const v = () => grammarValidator()
const vg = () => grepGrammarValidator()

describe('prefix heuristics: tool alias equivalence', () => {
  const defaultCases = [
    {
      label: '01: shell alias equals canonical',
      aliasInput: '<magnitude:shell><magnitude:command>pwd</magnitude:command></magnitude:shell><magnitude:yield_user/>',
      canonicalInput: '<magnitude:invoke tool="shell"><magnitude:parameter name="command">pwd</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>',
    },
    {
      label: '02: shell alias with canonical param equals canonical',
      aliasInput: '<magnitude:shell><magnitude:parameter name="command">pwd</magnitude:parameter></magnitude:shell><magnitude:yield_user/>',
      canonicalInput: '<magnitude:invoke tool="shell"><magnitude:parameter name="command">pwd</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>',
    },
    {
      label: '03: edit alias equals canonical',
      aliasInput: '<magnitude:edit><magnitude:path>a.ts</magnitude:path><magnitude:old>x</magnitude:old><magnitude:new>y</magnitude:new></magnitude:edit><magnitude:yield_user/>',
      canonicalInput: '<magnitude:invoke tool="edit"><magnitude:parameter name="path">a.ts</magnitude:parameter><magnitude:parameter name="old">x</magnitude:parameter><magnitude:parameter name="new">y</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>',
    },
    {
      label: '04: tree alias equals canonical',
      aliasInput: '<magnitude:tree></magnitude:tree><magnitude:yield_user/>',
      canonicalInput: '<magnitude:invoke tool="tree"></magnitude:invoke><magnitude:yield_user/>',
    },
    {
      label: '06: alias equivalence after message',
      aliasInput: '<magnitude:message>prep</magnitude:message><magnitude:shell><magnitude:command>pwd</magnitude:command></magnitude:shell><magnitude:yield_user/>',
      canonicalInput: '<magnitude:message>prep</magnitude:message><magnitude:invoke tool="shell"><magnitude:parameter name="command">pwd</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>',
    },
    {
      label: '07: alias equivalence after reason',
      aliasInput: '<magnitude:reason>prep</magnitude:reason><magnitude:shell><magnitude:command>pwd</magnitude:command></magnitude:shell><magnitude:yield_user/>',
      canonicalInput: '<magnitude:reason>prep</magnitude:reason><magnitude:invoke tool="shell"><magnitude:parameter name="command">pwd</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>',
    },
    {
      label: '08: alias equivalence with filter child',
      aliasInput: '<magnitude:shell><magnitude:filter>*.ts</magnitude:filter><magnitude:command>pwd</magnitude:command></magnitude:shell><magnitude:yield_user/>',
      canonicalInput: '<magnitude:invoke tool="shell"><magnitude:filter>*.ts</magnitude:filter><magnitude:parameter name="command">pwd</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>',
    },
    {
      label: '09: alias equivalence in two-invoke sequence',
      aliasInput: '<magnitude:shell><magnitude:command>pwd</magnitude:command></magnitude:shell><magnitude:tree></magnitude:tree><magnitude:yield_user/>',
      canonicalInput: '<magnitude:invoke tool="shell"><magnitude:parameter name="command">pwd</magnitude:parameter></magnitude:invoke><magnitude:invoke tool="tree"></magnitude:invoke><magnitude:yield_user/>',
    },
  ] as const

  for (const testCase of defaultCases) {
    it(testCase.label, () => {
      v().passes(testCase.aliasInput)
      v().passes(testCase.canonicalInput)
      expectToolAliasEquivalent(testCase.aliasInput, testCase.canonicalInput)
    })
  }

  it('05: grep alias equals canonical', () => {
    const aliasInput = '<magnitude:grep><magnitude:pattern>TODO</magnitude:pattern></magnitude:grep><magnitude:yield_user/>'
    const canonicalInput = '<magnitude:invoke tool="grep"><magnitude:parameter name="pattern">TODO</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>'
    vg().passes(aliasInput)
    vg().passes(canonicalInput)
    expectToolAliasEquivalent(aliasInput, canonicalInput, parseWithGrep)
  })

  it('10: grep full payload equivalence', () => {
    const aliasInput = '<magnitude:grep><magnitude:pattern>TODO</magnitude:pattern><magnitude:glob>*.ts</magnitude:glob><magnitude:path>src</magnitude:path><magnitude:limit>10</magnitude:limit></magnitude:grep><magnitude:yield_user/>'
    const canonicalInput = '<magnitude:invoke tool="grep"><magnitude:parameter name="pattern">TODO</magnitude:parameter><magnitude:parameter name="glob">*.ts</magnitude:parameter><magnitude:parameter name="path">src</magnitude:parameter><magnitude:parameter name="limit">10</magnitude:parameter></magnitude:invoke><magnitude:yield_user/>'
    vg().passes(aliasInput)
    vg().passes(canonicalInput)
    expectToolAliasEquivalent(aliasInput, canonicalInput, parseWithGrep)
  })
})
