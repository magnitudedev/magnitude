import { describe, expect, test } from 'bun:test'
import { parseSkill } from '../parser'
import { createWorkflowState, getCurrentPrompt, reduce, validateFields, type WorkflowState } from '../runtime'

function mapOf(entries: Array<[string, string]>): ReadonlyMap<string, string> {
  return new Map(entries)
}

function loadFeatureSkill() {
  const featureSkill = `---
name: feature
description: Plan, approve, build, and review a feature
---

<phase name="explore">
  <submit>
    <file name="findings" type="md" description="Exploration findings covering architecture, patterns, and integration points"/>
  </submit>
</phase>

Use parallelized explorers to understand all code areas relevant to this feature. Map out the architecture, existing patterns, dependencies, and integration points.

<phase name="plan">
  <submit>
    <file name="plan" type="md" description="Detailed implementation plan"/>
  </submit>
</phase>

Based on {{explore.findings}}, create a detailed implementation plan.

<phase name="approve">
  <criteria>
    <user-approval>
      Review the implementation plan at {{plan.plan}}. Ready to begin implementation?
    </user-approval>
  </criteria>
</phase>

Discuss and iterate with the user.

<phase name="build">
  <submit>
    <file name="test_script" type="sh" description="Script that runs tests for the new feature"/>
  </submit>
  <criteria>
    <shell-succeed>bash {{build.test_script}}</shell-succeed>
  </criteria>
</phase>

Implement the plan at {{plan.plan}}.

<phase name="review">
  <criteria>
    <shell-succeed>bash {{build.test_script}}</shell-succeed>
    <agent-approval subagent="reviewer">
      Review implementation against {{plan.plan}}.
    </agent-approval>
  </criteria>
</phase>

Address review issues.
`
  return parseSkill(featureSkill)
}

describe('workflow runtime', () => {
  test('create state initializes first phase as active', () => {
    const skill = loadFeatureSkill()
    const state = createWorkflowState(skill)

    expect(state.status).toBe('active')
    expect(state.currentPhaseIndex).toBe(0)
    expect(state.phaseStatuses).toEqual(['active', 'pending', 'pending', 'pending', 'pending'])
  })

  test('current prompt includes preamble for first phase', () => {
    const state = createWorkflowState(loadFeatureSkill())
    const prompt = getCurrentPrompt(state)

    expect(prompt).toContain('Use parallelized explorers')
  })

  test('validateFields reports missing required fields', () => {
    const state = createWorkflowState(loadFeatureSkill())

    const result = validateFields(state, mapOf([]))
    expect(result.valid).toBe(false)
    if (!result.valid) {
      expect(result.errors[0]?.name).toBe('findings')
      expect(result.errors[0]?.type).toBe('missing')
    }
  })

  test('submit on no-criteria phase stores submissions without advancing', () => {
    let state = createWorkflowState(loadFeatureSkill())

    state = reduce(state, {
      type: 'submit',
      fields: mapOf([['findings', '/tmp/findings.md']]),
    })

    expect(state.currentPhaseIndex).toBe(0)
    expect(state.phaseStatuses[0]).toBe('active')
    expect(state.submissions.get('explore.findings')).toBe('/tmp/findings.md')
  })

  test('criteria phase submit moves phase to awaiting-criteria', () => {
    let state: WorkflowState = createWorkflowState(loadFeatureSkill())
    state = reduce(state, { type: 'submit', fields: mapOf([['findings', '/tmp/findings.md']]) })
    state = reduce(state, { type: 'advance' })
    state = reduce(state, { type: 'submit', fields: mapOf([['plan', '/tmp/plan.md']]) })
    state = reduce(state, { type: 'advance' })

    state = reduce(state, { type: 'submit', fields: mapOf([]) })

    expect(state.currentPhaseIndex).toBe(2)
    expect(state.phaseStatuses[2]).toBe('awaiting-criteria')
  })

  test('advance moves to next phase', () => {
    let state: WorkflowState = createWorkflowState(loadFeatureSkill())
    state = reduce(state, { type: 'submit', fields: mapOf([['findings', '/tmp/findings.md']]) })
    state = reduce(state, { type: 'advance' })
    state = reduce(state, { type: 'submit', fields: mapOf([['plan', '/tmp/plan.md']]) })
    state = reduce(state, { type: 'advance' })
    state = reduce(state, { type: 'submit', fields: mapOf([]) })

    state = reduce(state, { type: 'advance' })

    expect(state.currentPhaseIndex).toBe(3)
    expect(state.phaseStatuses[2]).toBe('completed')
    expect(state.phaseStatuses[3]).toBe('active')
  })

  test('criteria-failed keeps same phase and resets to active', () => {
    let state: WorkflowState = createWorkflowState(loadFeatureSkill())
    state = reduce(state, { type: 'submit', fields: mapOf([['findings', '/tmp/findings.md']]) })
    state = reduce(state, { type: 'advance' })
    state = reduce(state, { type: 'submit', fields: mapOf([['plan', '/tmp/plan.md']]) })
    state = reduce(state, { type: 'advance' })
    state = reduce(state, { type: 'submit', fields: mapOf([]) })

    state = reduce(state, { type: 'criteria-failed', results: [{ type: 'failed', reason: 'need revisions' }] })

    expect(state.currentPhaseIndex).toBe(2)
    expect(state.phaseStatuses[2]).toBe('active')
  })

  test('walks full workflow to completion', () => {
    let state: WorkflowState = createWorkflowState(loadFeatureSkill())

    state = reduce(state, { type: 'submit', fields: mapOf([['findings', '/tmp/findings.md']]) })
    state = reduce(state, { type: 'advance' })
    state = reduce(state, { type: 'submit', fields: mapOf([['plan', '/tmp/plan.md']]) })
    state = reduce(state, { type: 'advance' })
    state = reduce(state, { type: 'submit', fields: mapOf([]) })
    state = reduce(state, { type: 'advance' })

    state = reduce(state, { type: 'submit', fields: mapOf([['test_script', '/tmp/test.sh']]) })
    expect(state.phaseStatuses[3]).toBe('awaiting-criteria')
    state = reduce(state, { type: 'advance' })

    state = reduce(state, { type: 'submit', fields: mapOf([]) })
    expect(state.phaseStatuses[4]).toBe('awaiting-criteria')
    state = reduce(state, { type: 'advance' })

    expect(state.status).toBe('completed')
    expect(state.currentPhaseIndex).toBe(5)
    expect(getCurrentPrompt(state)).toBe('')
  })
})
