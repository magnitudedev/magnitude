import { describe, it, expect } from 'vitest'
import { buildValidator } from './helpers'
import type { GrammarToolDef } from '../grammar-builder'

const TOOLS: GrammarToolDef[] = [
  {
    tagName: 'shell',
    parameters: [{ name: 'command', field: 'command', type: 'scalar', required: true }],
  },
  {
    tagName: 'create_task',
    parameters: [
      { name: 'id', field: 'id', type: 'scalar', required: true },
      { name: 'type', field: 'type', type: 'scalar', required: true },
      { name: 'title', field: 'title', type: 'scalar', required: true },
      { name: 'parent', field: 'parent', type: 'scalar', required: true },
    ],
  },
  {
    tagName: 'update_task',
    parameters: [
      { name: 'id', field: 'id', type: 'scalar', required: true },
      { name: 'status', field: 'status', type: 'scalar', required: true },
    ],
  },
  {
    tagName: 'spawn_worker',
    parameters: [
      { name: 'id', field: 'id', type: 'scalar', required: true },
      { name: 'role', field: 'role', type: 'scalar', required: true },
      { name: 'message', field: 'message', type: 'scalar', required: true },
    ],
  },
]

describe('grammar compliance - Fireworks reproduction cases', () => {
  const v = buildValidator(TOOLS)

  describe('valid turns', () => {
    it('accepts simple shell invoke with yield', () => {
      v.passes(
        '<magnitude:reason>I need to list files.</magnitude:reason>\n' +
        '<magnitude:invoke tool="shell"><magnitude:parameter name="command">ls -la</magnitude:parameter></magnitude:invoke>\n' +
        '<magnitude:yield_invoke/>'
      )
    })

    it('accepts canonical shell invoke with parameter tag', () => {
      v.passes(
        '<magnitude:reason>I need to list files.</magnitude:reason>\n' +
        '<magnitude:invoke tool="shell"><magnitude:parameter name="command">ls -la</magnitude:parameter></magnitude:invoke>\n' +
        '<magnitude:yield_invoke/>'
      )
    })

    it('accepts multiple reason tags then tools then yield', () => {
      v.passes(
        '<magnitude:reason about="alignment">\nUser approved the plan.\n</magnitude:reason>\n' +
        '<magnitude:reason about="tasks">\nCreate tasks for each phase.\n</magnitude:reason>\n' +
        '<magnitude:reason about="turn">\nI will create the phase tasks.\n</magnitude:reason>\n' +
        '<magnitude:message to="user">\nStarting the build now.\n</magnitude:message>\n' +
        '<magnitude:invoke tool="create_task">\n' +
        '<magnitude:parameter name="id">phase1</magnitude:parameter>\n' +
        '<magnitude:parameter name="type">implementation</magnitude:parameter>\n' +
        '<magnitude:parameter name="title">Phase 1: Project Setup</magnitude:parameter>\n' +
        '<magnitude:parameter name="parent">kanban</magnitude:parameter>\n' +
        '</magnitude:invoke>\n' +
        '<magnitude:yield_invoke/>'
      )
    })

    it('accepts multiple create_task invocations', () => {
      v.passes(
        '<magnitude:reason about="alignment">Planning.</magnitude:reason>\n' +
        '<magnitude:invoke tool="create_task">\n' +
        '<magnitude:parameter name="id">phase1</magnitude:parameter>\n' +
        '<magnitude:parameter name="type">implementation</magnitude:parameter>\n' +
        '<magnitude:parameter name="title">Phase 1</magnitude:parameter>\n' +
        '<magnitude:parameter name="parent">kanban</magnitude:parameter>\n' +
        '</magnitude:invoke>\n' +
        '<magnitude:invoke tool="create_task">\n' +
        '<magnitude:parameter name="id">phase2</magnitude:parameter>\n' +
        '<magnitude:parameter name="type">implementation</magnitude:parameter>\n' +
        '<magnitude:parameter name="title">Phase 2</magnitude:parameter>\n' +
        '<magnitude:parameter name="parent">kanban</magnitude:parameter>\n' +
        '</magnitude:invoke>\n' +
        '<magnitude:yield_invoke/>'
      )
    })
  })

  describe('empty bodies - should these be rejected?', () => {
    it('should reject empty shell alias body', () => {
      v.rejects(
        '<magnitude:reason>thinking</magnitude:reason>\n' +
        '<magnitude:shell></magnitude:shell>\n' +
        '<magnitude:yield_invoke/>'
      )
    })

    it('should reject empty invoke body with no parameters', () => {
      v.rejects(
        '<magnitude:reason>thinking</magnitude:reason>\n' +
        '<magnitude:invoke tool="shell"></magnitude:invoke>\n' +
        '<magnitude:yield_invoke/>'
      )
    })
  })

  describe('malformed tags - should be rejected', () => {
    it('should reject malformed parameter attribute (missing = and quote)', () => {
      v.rejects(
        '<magnitude:reason about="alignment">\nUser approved.\n</magnitude:reason>\n' +
        '<magnitude:invoke tool="create_task">\n' +
        '<magnitude:parameter name="id">phase3</magnitude:parameter>\n' +
        '<magnitude:parameter name">Phase 3: Board</magnitude:parameter>\n' +
        '</magnitude:invoke>\n' +
        '<magnitude:yield_invoke/>'
      )
    })

    it('should reject unknown parameter names', () => {
      v.rejects(
        '<magnitude:reason>thinking</magnitude:reason>\n' +
        '<magnitude:invoke tool="create_task">\n' +
        '<magnitude:parameter name="bogus">value</magnitude:parameter>\n' +
        '</magnitude:invoke>\n' +
        '<magnitude:yield_invoke/>'
      )
    })

    it('should reject unknown tool names', () => {
      v.rejects(
        '<magnitude:reason>thinking</magnitude:reason>\n' +
        '<magnitude:invoke tool="nonexistent">\n' +
        '</magnitude:invoke>\n' +
        '<magnitude:yield_invoke/>'
      )
    })
  })

  describe('yield termination', () => {
    it('accepts yield_user', () => {
      v.passes(
        '<magnitude:reason>Done.</magnitude:reason>\n' +
        '<magnitude:yield_user/>'
      )
    })

    it('accepts yield_invoke', () => {
      v.passes(
        '<magnitude:reason>Running tools.</magnitude:reason>\n' +
        '<magnitude:invoke tool="shell"><magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke>\n' +
        '<magnitude:yield_invoke/>'
      )
    })

    it('rejects turn with no yield', () => {
      v.rejects(
        '<magnitude:reason>thinking</magnitude:reason>\n' +
        '<magnitude:invoke tool="shell"><magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke>'
      )
    })
  })
})
