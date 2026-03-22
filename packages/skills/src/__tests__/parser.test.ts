import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { parseSkill } from '../parser'
import { resolveTemplates } from '../template'

const featureContent = readFileSync(resolve(import.meta.dir, '../../../../.magnitude/skills/feature/SKILL.md'), 'utf8')
const bugContent = readFileSync(resolve(import.meta.dir, '../../../../.magnitude/skills/bug/SKILL.md'), 'utf8')
const refactorContent = readFileSync(resolve(import.meta.dir, '../../../../.magnitude/skills/refactor/SKILL.md'), 'utf8')

describe('parseSkill', () => {
  test('parses feature skill', () => {
    const skill = parseSkill(featureContent)

    expect(skill.name).toBe('feature')
    expect(skill.description).toBe('Plan, approve, build, and review a feature')
    expect(skill.preamble).toBe('')
    expect(skill.phases.map((p) => p.name)).toEqual(['explore', 'plan', 'build', 'review'])

    expect(skill.phases[0].submit?.fields).toEqual([
      {
        type: 'file',
        name: 'findings',
        fileType: 'md',
        description: 'Exploration findings covering architecture, patterns, and integration points',
      },
    ])

    expect(skill.phases[1].criteria).toEqual([
      {
        type: 'user-approval',
        name: '',
        message: 'Review the implementation plan. Ready to begin implementation?',
      },
    ])

    expect(skill.phases[3].criteria).toEqual([
      { type: 'shell-succeed', name: 'tests', command: 'bash {{build.test_script}}' },
      {
        type: 'agent-approval',
        name: 'review',
        subagent: 'reviewer',
        prompt:
          'Review the implementation for correctness, code quality, test coverage, and adherence to the plan at {{plan.plan}}.',
      },
    ])

    expect(skill.phases[1].prompt).toContain('Based on `{{explore.findings}}`, create a detailed implementation plan.')
  })

  test('parses bug skill', () => {
    const skill = parseSkill(bugContent)

    expect(skill.name).toBe('bug')
    expect(skill.description).toBe('Systematically diagnose, reproduce, and fix a bug')
    expect(skill.preamble).toBe('')
    expect(skill.phases.map((p) => p.name)).toEqual(['investigate', 'reproduce', 'fix', 'verify'])

    expect(skill.phases[0].submit?.fields).toEqual([
      {
        type: 'file',
        name: 'analysis',
        fileType: 'md',
        description: 'Analysis of involved systems, potential causes, and debugging strategy',
      },
    ])

    expect(skill.phases[1].criteria).toEqual([
      { type: 'shell-succeed', name: 'repro', command: '! bash {{reproduce.repro_test}}' },
    ])
    expect(skill.phases[3].criteria?.[1]).toEqual({
      type: 'agent-approval',
      name: 'review',
      subagent: 'reviewer',
      prompt:
        'Review the fix for correctness and ensure it addresses the root cause identified in {{investigate.analysis}} without introducing regressions.',
    })

    expect(skill.phases[1].prompt).toContain('The reproduction script must fail (exit non-zero)')
  })

  test('parses refactor skill', () => {
    const skill = parseSkill(refactorContent)

    expect(skill.name).toBe('refactor')
    expect(skill.description).toBe('Safely refactor code with test-verified behavior parity')
    expect(skill.preamble).toBe('')
    expect(skill.phases.map((p) => p.name)).toEqual(['scope', 'baseline', 'refactor', 'verify'])

    expect(skill.phases[0].submit?.fields).toEqual([
      {
        type: 'file',
        name: 'scope_doc',
        fileType: 'md',
        description: 'Document describing refactor scope and approach',
      },
      {
        type: 'file',
        name: 'test_script',
        fileType: 'sh',
        description: 'Script that runs the relevant tests and captures results',
      },
    ])

    expect(skill.phases[1].submit?.fields).toEqual([
      {
        type: 'file',
        name: 'baseline_results',
        fileType: undefined,
        description: 'Baseline test results',
      },
    ])

    expect(skill.phases[2].criteria).toEqual([
      { type: 'shell-succeed', name: 'parity', command: 'bash {{refactor.verify_script}}' },
    ])
    expect(skill.phases[3].criteria?.[1]).toEqual({
      type: 'agent-approval',
      name: 'review',
      subagent: 'reviewer',
      prompt:
        'Review the refactored code for quality, readability, and adherence to the scope defined in {{scope.scope_doc}}.',
    })

    expect(skill.phases[0].prompt).toContain('Identify the scope of this refactor.')
  })

  test('supports skills with no phases', () => {
    const content = `---
name: docs
description: no phase doc
---

# Hello

This is preamble content only.
`
    const skill = parseSkill(content)

    expect(skill.name).toBe('docs')
    expect(skill.description).toBe('no phase doc')
    expect(skill.phases).toEqual([])
    expect(skill.preamble).toContain('# Hello')
  })
})

describe('resolveTemplates', () => {
  test('replaces {{phase.field}} values', () => {
    const resolved = resolveTemplates(
      'Use {{plan.file}} and {{build.script}} (keep {{missing.value}})',
      new Map([
        ['plan.file', 'plan.md'],
        ['build.script', 'test.sh'],
      ]),
    )

    expect(resolved).toBe('Use plan.md and test.sh (keep {{missing.value}})')
  })
})
