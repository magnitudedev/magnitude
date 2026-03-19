import { describe, expect, test } from 'bun:test'
import { getXmlActProtocol } from './index'

describe('subagent turn-control prompt text', () => {
  test('includes interruption model and parent-status-before-ending rule', () => {
    const prompt = getXmlActProtocol('parent', [], 'subagent')

    expect(prompt).toContain('The final tag in your response determines whether you will idle until further action from parent.')
    expect(prompt).toContain('you may reply to the user')
    expect(prompt).toContain('Before ending your turn, still send the required status message to your parent.')
  })
})