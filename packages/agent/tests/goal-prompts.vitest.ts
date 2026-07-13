import { describe, expect, it } from 'vitest'

import { renderGoalEarlyStopInjection, renderGoalStartedInjection } from '../src/prompts/goal'

describe('goal prompt injections', () => {
  it('renders a workflow-oriented initial goal prompt', () => {
    const prompt = renderGoalStartedInjection('Ship the feature')

    expect(prompt).toContain('<objective>\nShip the feature\n</objective>')
    expect(prompt).toContain('Goal workflow:')
    expect(prompt).toContain('Preserve the full objective')
    expect(prompt).toContain('Work from current evidence')
    expect(prompt).toContain('Completion audit before calling `finish_goal`')
    expect(prompt).toContain('Treat missing, weak, indirect, or merely plausible evidence as incomplete')
  })

  it('renders a compact early-stop continuation prompt', () => {
    const prompt = renderGoalEarlyStopInjection('Ship the feature')

    expect(prompt).toContain('<goal_incomplete>')
    expect(prompt).toContain('You stopped while the active goal is unfinished.')
    expect(prompt).toContain('Continue working now:')
    expect(prompt).toContain('Do not stop again merely to summarize incomplete progress.')
    expect(prompt).toContain('Call `finish_goal` only when current evidence proves the full goal is complete')
  })
})
