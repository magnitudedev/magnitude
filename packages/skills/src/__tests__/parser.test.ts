import { describe, it, expect } from 'vitest'
import { Effect } from 'effect'
import { parseSkill } from '../parser'

const runParseSkill = (content: string) => Effect.runSync(parseSkill(content))

describe('parseSkill', () => {
  it('plain skill (no markers) — body goes to shared', () => {
    const content = `---
name: foo
description: does foo
---

Just instructions here.
`
    const skill = runParseSkill(content)
    expect(skill.name).toBe('foo')
    expect(skill.description).toBe('does foo')
    expect(skill.sections.shared).toBe('Just instructions here.')
    expect(skill.sections.lead).toBe('')
    expect(skill.sections.worker).toBe('')
    expect(skill.sections.handoff).toBe('')
  })

  it('full skill (all markers) — each section populated', () => {
    const content = `---
name: implement
description: Build a code change
---

<!-- @lead -->
Lead orchestration guidance.

<!-- @worker -->
Worker instructions.

<!-- @handoff -->
What to check on return.
`
    const skill = runParseSkill(content)
    expect(skill.sections.shared).toBe('')
    expect(skill.sections.lead).toBe('Lead orchestration guidance.')
    expect(skill.sections.worker).toBe('Worker instructions.')
    expect(skill.sections.handoff).toBe('What to check on return.')
  })

  it('accumulation — two @worker blocks join with double newline', () => {
    const content = `---
name: multi
description: test
---

<!-- @worker -->
First worker block.

<!-- @worker -->
Second worker block.
`
    const skill = runParseSkill(content)
    expect(skill.sections.worker).toBe('First worker block.\n\nSecond worker block.')
  })

  it('mixed preamble + markers — preamble goes to shared', () => {
    const content = `---
name: mixed
description: test
---

Shared preamble content.

<!-- @lead -->
Lead-only content.
`
    const skill = runParseSkill(content)
    expect(skill.sections.shared).toBe('Shared preamble content.')
    expect(skill.sections.lead).toBe('Lead-only content.')
  })

  it('explicit @shared after other markers — shared accumulates', () => {
    const content = `---
name: explicit-shared
description: test
---

<!-- @worker -->
Worker content.

<!-- @shared -->
Explicit shared content.
`
    const skill = runParseSkill(content)
    expect(skill.sections.worker).toBe('Worker content.')
    expect(skill.sections.shared).toBe('Explicit shared content.')
  })

  it('thinking parsing — produces ThinkingLens array', () => {
    const content = `---
name: think
description: test
thinking:
  - lens: quality
    trigger: When editing files
    description: Does this hold up to project standards?
  - lens: turn
    trigger: When deciding next action
    description: What is the most focused next action?
---

Body content.
`
    const skill = runParseSkill(content)
    expect(skill.thinking).toHaveLength(2)
    expect(skill.thinking[0]).toEqual({
      lens: 'quality',
      trigger: 'When editing files',
      description: 'Does this hold up to project standards?',
    })
    expect(skill.thinking[1]).toEqual({
      lens: 'turn',
      trigger: 'When deciding next action',
      description: 'What is the most focused next action?',
    })
  })

  it('missing frontmatter — name/description empty, body goes to shared', () => {
    const content = `Just some content without frontmatter.`
    const skill = runParseSkill(content)
    expect(skill.name).toBe('')
    expect(skill.description).toBe('')
    expect(skill.sections.shared).toBe('Just some content without frontmatter.')
  })

  it('empty skill — all fields empty', () => {
    const skill = runParseSkill('')
    expect(skill.name).toBe('')
    expect(skill.description).toBe('')
    expect(skill.thinking).toHaveLength(0)
    expect(skill.sections.shared).toBe('')
    expect(skill.sections.lead).toBe('')
    expect(skill.sections.worker).toBe('')
    expect(skill.sections.handoff).toBe('')
  })
})
