import { describe, expect, it } from 'vitest'
import { windowToPrompt } from '../src/window/render'
import type { ForkWindowState } from '../src/window'

describe('failed attempt prompt history', () => {
  it('renders error feedback without manufacturing an empty assistant message', () => {
    const windowState: ForkWindowState = {
      messages: [{
        type: 'attempt_feedback',
        source: 'agent',
        turnId: 'failed-turn',
        feedback: [{ kind: 'error', message: 'request was rejected' }],
        estimatedTokens: 1,
      }],
      queuedTimeline: [],
      currentTurnId: null,
      currentChainId: null,
      nextQueueSeq: 0,
      _activeMessageIsCoordinator: false,
      _coordinatorChars: 0,
      tokenEstimate: 0,
      messageTokens: 0,
      systemPromptTokens: 0,
      lastAnchoredTotal: null,
      lastAnchoredMessageTokens: null,
      autopilotEnabled: false,
      consumerAutopilotKnowledge: { advisor: null, leader: null },
    }

    const prompt = windowToPrompt({
      windowState,
      systemPrompt: 'system',
      timezone: null,
      formatter: () => [],
      autopilotEnabled: false,
      leaderLastAutopilotKnowledge: null,
      includeImageData: false,
    })

    expect(prompt.messages.some((message) => message._tag === 'AssistantMessage')).toBe(false)
    expect(prompt.messages).toEqual([
      expect.objectContaining({
        _tag: 'UserMessage',
        parts: [expect.objectContaining({ _tag: 'TextPart', text: expect.stringContaining('request was rejected') })],
      }),
    ])
  })
})
