import { describe, it } from 'vitest'
import { buildValidator } from './helpers'
import type { GrammarToolDef } from '../grammar-builder'

/**
 * Grammar strictness test suite — asserts DESIRED behavior.
 *
 * Covers three change categories:
 * 1. Required parameter enforcement (close gated on requiredCount)
 * 2. Greedy body tightening (no <magnitude: absorption after false close)
 * 3. Alias removal from constrained positions (canonical forms only)
 *
 * Many tests currently FAIL — they document the target behavior.
 */

// =============================================================================
// Tool definitions with required/optional metadata
// =============================================================================

// Note: GrammarParameterDef doesn't have `required` yet. These defs use the
// current interface. Once `required` is added, update these with explicit values.
// For now, the tests assert behavior as if ALL params are required (which is
// the desired default for tools where all params are semantically required).

const SHELL: GrammarToolDef = {
  tagName: 'shell',
  parameters: [{ name: 'command', field: 'command', type: 'scalar', required: true }],
}

const CREATE_TASK: GrammarToolDef = {
  tagName: 'create_task',
  parameters: [
    { name: 'id', field: 'id', type: 'scalar', required: true },
    { name: 'type', field: 'type', type: 'scalar', required: true },
    { name: 'title', field: 'title', type: 'scalar', required: true },
    { name: 'parent', field: 'parent', type: 'scalar', required: true },
  ],
}

const EDIT: GrammarToolDef = {
  tagName: 'edit',
  parameters: [
    { name: 'path', field: 'path', type: 'scalar', required: true },
    { name: 'old', field: 'old', type: 'scalar', required: true },
    { name: 'new', field: 'new', type: 'scalar', required: true },
  ],
}

const TREE: GrammarToolDef = {
  tagName: 'tree',
  parameters: [],
}

const READ: GrammarToolDef = {
  tagName: 'read',
  parameters: [
    { name: 'path', field: 'path', type: 'scalar', required: true },
    { name: 'offset', field: 'offset', type: 'scalar', required: true },
    { name: 'limit', field: 'limit', type: 'scalar', required: true },
  ],
}

const ALL_TOOLS = [SHELL, CREATE_TASK, EDIT, TREE, READ]

describe('grammar strictness', () => {
  const v = buildValidator(ALL_TOOLS)

  // =========================================================================
  // 1. ALIAS REMOVAL — canonical forms only at constrained positions
  // =========================================================================
  describe('alias removal: opening tags must be canonical', () => {
    describe('top-level tool opens', () => {
      it('accepts canonical invoke open', () => {
        v.passes(
          '<magnitude:reason>r</magnitude:reason>\n' +
          '<magnitude:invoke tool="shell">\n' +
          '<magnitude:parameter name="command">ls</magnitude:parameter>\n' +
          '</magnitude:invoke>\n' +
          '<magnitude:yield_invoke/>'
        )
      })

      it('rejects tool alias open (shell)', () => {
        v.rejects(
          '<magnitude:reason>r</magnitude:reason>\n' +
          '<magnitude:shell>\n' +
          '<magnitude:command>ls</magnitude:command>\n' +
          '</magnitude:shell>\n' +
          '<magnitude:yield_invoke/>'
        )
      })

      it('rejects tool alias open (edit)', () => {
        v.rejects(
          '<magnitude:reason>r</magnitude:reason>\n' +
          '<magnitude:edit>\n' +
          '<magnitude:path>foo.ts</magnitude:path>\n' +
          '</magnitude:edit>\n' +
          '<magnitude:yield_invoke/>'
        )
      })

      it('rejects tool alias open (read)', () => {
        v.rejects(
          '<magnitude:reason>r</magnitude:reason>\n' +
          '<magnitude:read>\n' +
          '<magnitude:path>foo.ts</magnitude:path>\n' +
          '</magnitude:read>\n' +
          '<magnitude:yield_invoke/>'
        )
      })

      it('rejects tool alias open (create_task)', () => {
        v.rejects(
          '<magnitude:reason>r</magnitude:reason>\n' +
          '<magnitude:create_task>\n' +
          '<magnitude:id>t1</magnitude:id>\n' +
          '</magnitude:create_task>\n' +
          '<magnitude:yield_invoke/>'
        )
      })
    })

    describe('parameter opens', () => {
      it('accepts canonical parameter open', () => {
        v.passes(
          '<magnitude:reason>r</magnitude:reason>\n' +
          '<magnitude:invoke tool="shell">\n' +
          '<magnitude:parameter name="command">ls</magnitude:parameter>\n' +
          '</magnitude:invoke>\n' +
          '<magnitude:yield_invoke/>'
        )
      })

      it('rejects parameter alias open (command)', () => {
        v.rejects(
          '<magnitude:reason>r</magnitude:reason>\n' +
          '<magnitude:invoke tool="shell">\n' +
          '<magnitude:command>ls</magnitude:command>\n' +
          '</magnitude:invoke>\n' +
          '<magnitude:yield_invoke/>'
        )
      })

      it('rejects parameter alias open (path)', () => {
        v.rejects(
          '<magnitude:reason>r</magnitude:reason>\n' +
          '<magnitude:invoke tool="edit">\n' +
          '<magnitude:path>foo.ts</magnitude:path>\n' +
          '</magnitude:invoke>\n' +
          '<magnitude:yield_invoke/>'
        )
      })
    })

    describe('invoke close must be canonical', () => {
      it('accepts canonical invoke close', () => {
        v.passes(
          '<magnitude:reason>r</magnitude:reason>\n' +
          '<magnitude:invoke tool="shell">\n' +
          '<magnitude:parameter name="command">ls</magnitude:parameter>\n' +
          '</magnitude:invoke>\n' +
          '<magnitude:yield_invoke/>'
        )
      })

      it('accepts alias invoke close (</magnitude:shell>)', () => {
        v.passes(
          '<magnitude:reason>r</magnitude:reason>\n' +
          '<magnitude:invoke tool="shell">\n' +
          '<magnitude:parameter name="command">ls</magnitude:parameter>\n' +
          '</magnitude:shell>\n' +
          '<magnitude:yield_invoke/>'
        )
      })

      it('accepts alias invoke close (</magnitude:edit>)', () => {
        v.passes(
          '<magnitude:reason>r</magnitude:reason>\n' +
          '<magnitude:invoke tool="edit">\n' +
          '<magnitude:parameter name="path">foo.ts</magnitude:parameter>\n' +
          '<magnitude:parameter name="old">a</magnitude:parameter>\n' +
          '<magnitude:parameter name="new">b</magnitude:parameter>\n' +
          '</magnitude:edit>\n' +
          '<magnitude:yield_invoke/>'
        )
      })
    })
  })

  // =========================================================================
  // 2. REQUIRED PARAMETER ENFORCEMENT
  // =========================================================================
  describe('required parameter enforcement', () => {
    describe('empty tool bodies', () => {
      it('rejects empty invoke body (shell)', () => {
        v.rejects(
          '<magnitude:reason>r</magnitude:reason>\n' +
          '<magnitude:invoke tool="shell"></magnitude:invoke>\n' +
          '<magnitude:yield_invoke/>'
        )
      })

      it('rejects empty invoke body (create_task)', () => {
        v.rejects(
          '<magnitude:reason>r</magnitude:reason>\n' +
          '<magnitude:invoke tool="create_task"></magnitude:invoke>\n' +
          '<magnitude:yield_invoke/>'
        )
      })

      it('rejects empty invoke body (edit)', () => {
        v.rejects(
          '<magnitude:reason>r</magnitude:reason>\n' +
          '<magnitude:invoke tool="edit"></magnitude:invoke>\n' +
          '<magnitude:yield_invoke/>'
        )
      })

      it('allows empty invoke body for zero-param tool (tree)', () => {
        v.passes(
          '<magnitude:reason>r</magnitude:reason>\n' +
          '<magnitude:invoke tool="tree"></magnitude:invoke>\n' +
          '<magnitude:yield_invoke/>'
        )
      })
    })

    describe('partial required parameters', () => {
      it('rejects create_task with 1 of 4 params', () => {
        v.rejects(
          '<magnitude:reason>r</magnitude:reason>\n' +
          '<magnitude:invoke tool="create_task">\n' +
          '<magnitude:parameter name="id">t1</magnitude:parameter>\n' +
          '</magnitude:invoke>\n' +
          '<magnitude:yield_invoke/>'
        )
      })

      it('rejects create_task with 2 of 4 params', () => {
        v.rejects(
          '<magnitude:reason>r</magnitude:reason>\n' +
          '<magnitude:invoke tool="create_task">\n' +
          '<magnitude:parameter name="id">t1</magnitude:parameter>\n' +
          '<magnitude:parameter name="title">Task</magnitude:parameter>\n' +
          '</magnitude:invoke>\n' +
          '<magnitude:yield_invoke/>'
        )
      })

      it('rejects create_task with 3 of 4 params', () => {
        v.rejects(
          '<magnitude:reason>r</magnitude:reason>\n' +
          '<magnitude:invoke tool="create_task">\n' +
          '<magnitude:parameter name="id">t1</magnitude:parameter>\n' +
          '<magnitude:parameter name="type">implement</magnitude:parameter>\n' +
          '<magnitude:parameter name="title">Task</magnitude:parameter>\n' +
          '</magnitude:invoke>\n' +
          '<magnitude:yield_invoke/>'
        )
      })

      it('accepts create_task with all 4 params', () => {
        v.passes(
          '<magnitude:reason>r</magnitude:reason>\n' +
          '<magnitude:invoke tool="create_task">\n' +
          '<magnitude:parameter name="id">t1</magnitude:parameter>\n' +
          '<magnitude:parameter name="type">implement</magnitude:parameter>\n' +
          '<magnitude:parameter name="title">Task</magnitude:parameter>\n' +
          '<magnitude:parameter name="parent">root</magnitude:parameter>\n' +
          '</magnitude:invoke>\n' +
          '<magnitude:yield_invoke/>'
        )
      })

      it('rejects edit with 1 of 3 params', () => {
        v.rejects(
          '<magnitude:reason>r</magnitude:reason>\n' +
          '<magnitude:invoke tool="edit">\n' +
          '<magnitude:parameter name="path">foo.ts</magnitude:parameter>\n' +
          '</magnitude:invoke>\n' +
          '<magnitude:yield_invoke/>'
        )
      })

      it('rejects edit with 2 of 3 params', () => {
        v.rejects(
          '<magnitude:reason>r</magnitude:reason>\n' +
          '<magnitude:invoke tool="edit">\n' +
          '<magnitude:parameter name="path">foo.ts</magnitude:parameter>\n' +
          '<magnitude:parameter name="old">old</magnitude:parameter>\n' +
          '</magnitude:invoke>\n' +
          '<magnitude:yield_invoke/>'
        )
      })

      it('accepts edit with all 3 params', () => {
        v.passes(
          '<magnitude:reason>r</magnitude:reason>\n' +
          '<magnitude:invoke tool="edit">\n' +
          '<magnitude:parameter name="path">foo.ts</magnitude:parameter>\n' +
          '<magnitude:parameter name="old">old</magnitude:parameter>\n' +
          '<magnitude:parameter name="new">new</magnitude:parameter>\n' +
          '</magnitude:invoke>\n' +
          '<magnitude:yield_invoke/>'
        )
      })

      it('accepts shell with its one param', () => {
        v.passes(
          '<magnitude:reason>r</magnitude:reason>\n' +
          '<magnitude:invoke tool="shell">\n' +
          '<magnitude:parameter name="command">ls</magnitude:parameter>\n' +
          '</magnitude:invoke>\n' +
          '<magnitude:yield_invoke/>'
        )
      })
    })
  })

  // =========================================================================
  // 3. GREEDY BODY TIGHTENING — <magnitude: never body text
  // =========================================================================
  describe('greedy body: <magnitude: is always structural', () => {
    it('rejects <magnitude: inside parameter body', () => {
      v.rejects(
        '<magnitude:reason>r</magnitude:reason>\n' +
        '<magnitude:invoke tool="shell">\n' +
        '<magnitude:parameter name="command">echo <magnitude:foo>bar</magnitude:foo></magnitude:parameter>\n' +
        '</magnitude:invoke>\n' +
        '<magnitude:yield_invoke/>'
      )
    })

    it('rejects <magnitude: inside reason body', () => {
      v.rejects(
        '<magnitude:reason>I want to use <magnitude:broken tag</magnitude:reason>\n' +
        '<magnitude:yield_user/>'
      )
    })

    it('rejects <magnitude: inside message body', () => {
      v.rejects(
        '<magnitude:reason>r</magnitude:reason>\n' +
        '<magnitude:message to="user">Here is <magnitude:something>weird</magnitude:something></magnitude:message>\n' +
        '<magnitude:yield_user/>'
      )
    })

    it('rejects <magnitude: after false close in parameter body', () => {
      v.rejects(
        '<magnitude:reason>r</magnitude:reason>\n' +
        '<magnitude:invoke tool="shell">\n' +
        '<magnitude:parameter name="command">echo </magnitude:parameter> <magnitude:garbage>stuff</magnitude:parameter>\n' +
        '</magnitude:invoke>\n' +
        '<magnitude:yield_invoke/>'
      )
    })

    it('rejects <magnitude: after false close in filter body', () => {
      v.rejects(
        '<magnitude:reason>r</magnitude:reason>\n' +
        '<magnitude:invoke tool="shell">\n' +
        '<magnitude:parameter name="command">ls</magnitude:parameter>\n' +
        '<magnitude:filter>$.stdout </magnitude:filter> <magnitude:garbage>x</magnitude:filter>\n' +
        '</magnitude:invoke>\n' +
        '<magnitude:yield_invoke/>'
      )
    })
  })

  // =========================================================================
  // 4. MALFORMED PARAMETER ATTRIBUTES
  // =========================================================================
  describe('malformed parameter attributes', () => {
    it('rejects parameter with missing = and quote (name">)', () => {
      v.rejects(
        '<magnitude:reason>r</magnitude:reason>\n' +
        '<magnitude:invoke tool="create_task">\n' +
        '<magnitude:parameter name="id">t1</magnitude:parameter>\n' +
        '<magnitude:parameter name">Phase 3</magnitude:parameter>\n' +
        '</magnitude:invoke>\n' +
        '<magnitude:yield_invoke/>'
      )
    })

    it('rejects parameter with missing closing quote', () => {
      v.rejects(
        '<magnitude:reason>r</magnitude:reason>\n' +
        '<magnitude:invoke tool="create_task">\n' +
        '<magnitude:parameter name="id>t1</magnitude:parameter>\n' +
        '</magnitude:invoke>\n' +
        '<magnitude:yield_invoke/>'
      )
    })

    it('rejects parameter with no name attribute', () => {
      v.rejects(
        '<magnitude:reason>r</magnitude:reason>\n' +
        '<magnitude:invoke tool="create_task">\n' +
        '<magnitude:parameter>t1</magnitude:parameter>\n' +
        '</magnitude:invoke>\n' +
        '<magnitude:yield_invoke/>'
      )
    })

    it('rejects parameter with wrong attribute name', () => {
      v.rejects(
        '<magnitude:reason>r</magnitude:reason>\n' +
        '<magnitude:invoke tool="create_task">\n' +
        '<magnitude:parameter label="id">t1</magnitude:parameter>\n' +
        '</magnitude:invoke>\n' +
        '<magnitude:yield_invoke/>'
      )
    })
  })

  // =========================================================================
  // 5. UNKNOWN NAMES
  // =========================================================================
  describe('unknown names', () => {
    it('rejects unknown tool name', () => {
      v.rejects(
        '<magnitude:reason>r</magnitude:reason>\n' +
        '<magnitude:invoke tool="nonexistent"></magnitude:invoke>\n' +
        '<magnitude:yield_invoke/>'
      )
    })

    it('rejects unknown parameter name', () => {
      v.rejects(
        '<magnitude:reason>r</magnitude:reason>\n' +
        '<magnitude:invoke tool="shell">\n' +
        '<magnitude:parameter name="bogus">ls</magnitude:parameter>\n' +
        '</magnitude:invoke>\n' +
        '<magnitude:yield_invoke/>'
      )
    })
  })

  // =========================================================================
  // 6. PRESERVED LENIENCY — close-tag greedy last-match
  // =========================================================================
  describe('first-close-wins strictness', () => {
    it('rejects false close followed by more text then real close in parameter body', () => {
      v.rejects(
        '<magnitude:reason>r</magnitude:reason>\n' +
        '<magnitude:invoke tool="shell">\n' +
        '<magnitude:parameter name="command">echo </magnitude:parameter> more text </magnitude:parameter>\n' +
        '</magnitude:invoke>\n' +
        '<magnitude:yield_invoke/>'
      )
    })

    it('rejects false close followed by HTML then real close in parameter body', () => {
      v.rejects(
        '<magnitude:reason>r</magnitude:reason>\n' +
        '<magnitude:invoke tool="shell">\n' +
        '<magnitude:parameter name="command">cat </magnitude:parameter> <div>html</div> </magnitude:parameter>\n' +
        '</magnitude:invoke>\n' +
        '<magnitude:yield_invoke/>'
      )
    })

    it('rejects multiple false closes then real close in reason body', () => {
      v.rejects(
        '<magnitude:reason>The tag </magnitude:reason> is used for </magnitude:reason> reasoning</magnitude:reason>\n' +
        '<magnitude:yield_user/>'
      )
    })

    it('rejects false close in message body', () => {
      v.rejects(
        '<magnitude:reason>r</magnitude:reason>\n' +
        '<magnitude:message to="user">Use </magnitude:message> for messages</magnitude:message>\n' +
        '<magnitude:yield_user/>'
      )
    })
  })

  // =========================================================================
  // 7. ESCAPE REMOVAL
  // =========================================================================
  describe('escape removal', () => {
    it('rejects escape block in parameter body', () => {
      v.rejects(
        '<magnitude:reason>r</magnitude:reason>\n' +
        '<magnitude:invoke tool="shell">\n' +
        '<magnitude:parameter name="command">echo <magnitude:escape><magnitude:foo/></magnitude:escape> done</magnitude:parameter>\n' +
        '</magnitude:invoke>\n' +
        '<magnitude:yield_invoke/>'
      )
    })

    it('rejects escape block in reason body', () => {
      v.rejects(
        '<magnitude:reason>Use <magnitude:escape><magnitude:invoke tool="x"></magnitude:escape> for tools</magnitude:reason>\n' +
        '<magnitude:yield_user/>'
      )
    })
  })

  // =========================================================================
  // 8. VALID COMPLETE TURNS — sanity checks
  // =========================================================================
  describe('valid complete turns', () => {
    it('reason + shell invoke + yield_invoke', () => {
      v.passes(
        '<magnitude:reason>listing files</magnitude:reason>\n' +
        '<magnitude:invoke tool="shell">\n' +
        '<magnitude:parameter name="command">ls -la</magnitude:parameter>\n' +
        '</magnitude:invoke>\n' +
        '<magnitude:yield_invoke/>'
      )
    })

    it('reason + message + yield_user', () => {
      v.passes(
        '<magnitude:reason>responding</magnitude:reason>\n' +
        '<magnitude:message to="user">Hello!</magnitude:message>\n' +
        '<magnitude:yield_user/>'
      )
    })

    it('multiple reasons + message + multiple tools + yield', () => {
      v.passes(
        '<magnitude:reason about="alignment">Aligned.</magnitude:reason>\n' +
        '<magnitude:reason about="tasks">Tasks planned.</magnitude:reason>\n' +
        '<magnitude:message to="user">Starting.</magnitude:message>\n' +
        '<magnitude:invoke tool="create_task">\n' +
        '<magnitude:parameter name="id">t1</magnitude:parameter>\n' +
        '<magnitude:parameter name="type">implement</magnitude:parameter>\n' +
        '<magnitude:parameter name="title">Task 1</magnitude:parameter>\n' +
        '<magnitude:parameter name="parent">root</magnitude:parameter>\n' +
        '</magnitude:invoke>\n' +
        '<magnitude:invoke tool="shell">\n' +
        '<magnitude:parameter name="command">echo hello</magnitude:parameter>\n' +
        '</magnitude:invoke>\n' +
        '<magnitude:yield_invoke/>'
      )
    })

    it('zero-param tool', () => {
      v.passes(
        '<magnitude:reason>checking</magnitude:reason>\n' +
        '<magnitude:invoke tool="tree"></magnitude:invoke>\n' +
        '<magnitude:yield_invoke/>'
      )
    })

    it('reason-only + yield_user', () => {
      v.passes(
        '<magnitude:reason>thinking</magnitude:reason>\n' +
        '<magnitude:yield_user/>'
      )
    })

    it('tool with multiple params in different order', () => {
      v.passes(
        '<magnitude:reason>editing</magnitude:reason>\n' +
        '<magnitude:invoke tool="edit">\n' +
        '<magnitude:parameter name="new">new text</magnitude:parameter>\n' +
        '<magnitude:parameter name="path">foo.ts</magnitude:parameter>\n' +
        '<magnitude:parameter name="old">old text</magnitude:parameter>\n' +
        '</magnitude:invoke>\n' +
        '<magnitude:yield_invoke/>'
      )
    })
  })

  // =========================================================================
  // 9. YIELD TERMINATION
  // =========================================================================
  describe('yield termination', () => {
    it('rejects turn with no yield', () => {
      v.rejects(
        '<magnitude:reason>r</magnitude:reason>\n' +
        '<magnitude:invoke tool="shell">\n' +
        '<magnitude:parameter name="command">ls</magnitude:parameter>\n' +
        '</magnitude:invoke>'
      )
    })

    it('accepts yield_user', () => {
      v.passes(
        '<magnitude:reason>done</magnitude:reason>\n' +
        '<magnitude:yield_user/>'
      )
    })

    it('accepts yield_invoke', () => {
      v.passes(
        '<magnitude:reason>r</magnitude:reason>\n' +
        '<magnitude:invoke tool="shell">\n' +
        '<magnitude:parameter name="command">ls</magnitude:parameter>\n' +
        '</magnitude:invoke>\n' +
        '<magnitude:yield_invoke/>'
      )
    })

    it('accepts yield_worker', () => {
      v.passes(
        '<magnitude:reason>delegating</magnitude:reason>\n' +
        '<magnitude:yield_worker/>'
      )
    })
  })
})
