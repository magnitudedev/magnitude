import { describe, expect, it } from 'vitest'
import { Option } from 'effect'
import type { ProviderToolCallId, ToolCallId } from '@magnitudedev/ai'
import { windowToPrompt } from '../src/window/render'
import type { ForkWindowState } from '../src/window'

function makeWindowState(messages: ForkWindowState['messages']): ForkWindowState {
  return {
    messages,
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
}

describe('failed attempt prompt history', () => {
  it('renders error feedback without manufacturing an empty assistant message', () => {
    const windowState = makeWindowState([{
        type: 'attempt_feedback',
        source: 'agent',
        turnId: 'failed-turn',
        feedback: [{ kind: 'error', message: 'request was rejected' }],
        estimatedTokens: 1,
    }])

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

  it('renders tool-call limit feedback with its dedicated wrapper', () => {
    const windowState = makeWindowState([{
        type: 'attempt_feedback',
        source: 'agent',
        turnId: 'limited-turn',
        feedback: [{ kind: 'tool_limit_reached', limit: 10 }],
        estimatedTokens: 1,
    }])

    const prompt = windowToPrompt({
      windowState,
      systemPrompt: 'system',
      timezone: null,
      formatter: () => [],
      autopilotEnabled: false,
      leaderLastAutopilotKnowledge: null,
      includeImageData: false,
    })

    expect(prompt.messages).toEqual([{
      _tag: 'UserMessage',
      parts: [{
        _tag: 'TextPart',
        text: '<tool_limit_reached>\nAt most 10 tool calls are allowed per turn.\n</tool_limit_reached>',
      }],
    }])
  })

  it('places tool-call limit feedback after accepted calls and results', () => {
    const toolCallId = 'call-1' as ToolCallId
    const providerToolCallId = 'provider-call-1' as ProviderToolCallId
    const windowState = makeWindowState([{
        type: 'assistant_turn',
        source: 'agent',
        strategyId: 'native',
        estimatedTokens: 1,
        turn: {
          turnId: 'limited-turn',
          assistant: {
            _tag: 'AssistantMessage',
            reasoning: Option.none(),
            text: Option.none(),
            toolCalls: Option.some([{
              _tag: 'ToolCallPart',
              id: toolCallId,
              providerToolCallId,
              name: 'fileRead',
              input: { path: 'a.ts' },
            }]),
          },
          toolResults: [{
            toolCallId,
            providerToolCallId,
            toolName: 'fileRead',
            result: { _tag: 'Success', output: 'contents' },
          }],
          feedback: [{ kind: 'tool_limit_reached', limit: 10 }],
          clean: true,
        },
    }])

    const prompt = windowToPrompt({
      windowState,
      systemPrompt: 'system',
      timezone: null,
      formatter: () => [{ _tag: 'TextPart', text: 'contents' }],
      autopilotEnabled: false,
      leaderLastAutopilotKnowledge: null,
      includeImageData: false,
    })

    expect(prompt.messages.map((message) => message._tag)).toEqual([
      'AssistantMessage',
      'ToolResultMessage',
      'UserMessage',
    ])
    expect(prompt.messages[2]).toEqual({
      _tag: 'UserMessage',
      parts: [{
        _tag: 'TextPart',
        text: '<tool_limit_reached>\nAt most 10 tool calls are allowed per turn.\n</tool_limit_reached>',
      }],
    })
  })
})
