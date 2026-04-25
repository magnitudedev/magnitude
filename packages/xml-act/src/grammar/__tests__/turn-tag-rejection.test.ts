import { describe, it } from 'vitest'
import { shellValidator } from './helpers'

/**
 * Regression test: the LLM produced `<turn>...</turn>` instead of
 * `<magnitude:reason about="turn">...</magnitude:reason>`. The grammar should reject this
 * because `<turn>` is not a valid top-level tag.
 *
 * Observed raw output:
 *   <magnitude:reason about="skills">...</magnitude:reason>
 *   <magnitude:reason about="alignment">...</magnitude:reason>
 *   <magnitude:reason about="tasks">...</magnitude:reason>
 *   <magnitude:reason about="diligence">...</magnitude:reason>
 *   <turn>
 *   I'll start by activating the relevant skills...
 *   </turn>
 *   <magnitude:invoke tool="skill">...
 */

const REASON = (name: string, content: string) =>
  `<magnitude:reason about="${name}">\n${content}\n</magnitude:reason>\n`

describe('<turn> tag rejection', () => {
  it('rejects <turn>...</turn> after reason blocks', () => {
    const v = shellValidator()
    v.rejects(
      REASON('skills', 'This is a refactor task.') +
      REASON('alignment', 'The user provided a detailed spec.') +
      REASON('tasks', 'This is a large multi-step refactor.') +
      REASON('diligence', 'Need to verify the codebase.') +
      '<turn>\nI\'ll start by activating the relevant skills.\n</turn>\n' +
      '<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>\n' +
      '<magnitude:yield_invoke/>'
    )
  })

  it('rejects <turn>...</turn> as first element', () => {
    const v = shellValidator()
    v.rejects(
      '<turn>\nSome content.\n</turn>\n' +
      '<magnitude:yield_invoke/>'
    )
  })

  it('rejects <turn>...</turn> between reasons and invoke', () => {
    const v = shellValidator()
    v.rejects(
      REASON('turn', 'planning') +
      '<turn>\nDoing stuff.\n</turn>\n' +
      '<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo hi</magnitude:parameter>\n</magnitude:invoke>\n' +
      '<magnitude:yield_invoke/>'
    )
  })

  it('accepts <magnitude:reason about="turn"> (the correct form)', () => {
    const v = shellValidator()
    v.passes(
      REASON('skills', 'refactor skill applies') +
      REASON('turn', 'I\'ll start by activating skills.') +
      '<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>\n' +
      '<magnitude:yield_invoke/>'
    )
  })
})