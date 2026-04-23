import { describe, it } from 'vitest'
import { shellValidator } from './helpers'

/**
 * Regression test: the LLM produced `<turn>...</turn>` instead of
 * `<reason about="turn">...</reason>`. The grammar should reject this
 * because `<turn>` is not a valid top-level tag.
 *
 * Observed raw output:
 *   <reason about="skills">...</reason>
 *   <reason about="alignment">...</reason>
 *   <reason about="tasks">...</reason>
 *   <reason about="diligence">...</reason>
 *   <turn>
 *   I'll start by activating the relevant skills...
 *   </turn>
 *   <invoke tool="skill">...
 */

const REASON = (name: string, content: string) =>
  `<reason about="${name}">\n${content}\n</reason>\n`

describe('<turn> tag rejection', () => {
  it('rejects <turn>...</turn> after reason blocks', () => {
    const v = shellValidator()
    v.rejects(
      REASON('skills', 'This is a refactor task.') +
      REASON('alignment', 'The user provided a detailed spec.') +
      REASON('tasks', 'This is a large multi-step refactor.') +
      REASON('diligence', 'Need to verify the codebase.') +
      '<turn>\nI\'ll start by activating the relevant skills.\n</turn>\n' +
      '<invoke tool="shell">\n<parameter name="command">ls</parameter>\n</invoke>\n' +
      '<yield_invoke/>'
    )
  })

  it('rejects <turn>...</turn> as first element', () => {
    const v = shellValidator()
    v.rejects(
      '<turn>\nSome content.\n</turn>\n' +
      '<yield_invoke/>'
    )
  })

  it('rejects <turn>...</turn> between reasons and invoke', () => {
    const v = shellValidator()
    v.rejects(
      REASON('turn', 'planning') +
      '<turn>\nDoing stuff.\n</turn>\n' +
      '<invoke tool="shell">\n<parameter name="command">echo hi</parameter>\n</invoke>\n' +
      '<yield_invoke/>'
    )
  })

  it('accepts <reason about="turn"> (the correct form)', () => {
    const v = shellValidator()
    v.passes(
      REASON('skills', 'refactor skill applies') +
      REASON('turn', 'I\'ll start by activating skills.') +
      '<invoke tool="shell">\n<parameter name="command">ls</parameter>\n</invoke>\n' +
      '<yield_invoke/>'
    )
  })
})