import { describe, expect, it } from 'vitest'
import { Effect, Option } from 'effect'
import {
  finalizeChatCompletionsRequest,
  nativeChatCompletionsCodec,
  type AssistantMessage,
  type ProviderToolCallId,
  type ToolCallId,
} from '@magnitudedev/ai'
import type { ToolResult, ToolResultEntry } from '@magnitudedev/harness'
import { windowToPrompt } from '../src/window/render'
import type { ForkWindowState } from '../src/window'

const toolCallId = 'semantic-call' as ToolCallId
const providerToolCallId = 'provider-call' as ProviderToolCallId

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

function assistant(toolCalls: AssistantMessage['toolCalls']): AssistantMessage {
  return {
    _tag: 'AssistantMessage',
    reasoning: Option.none(),
    text: Option.none(),
    toolCalls,
  }
}

function toolTurn(result: ToolResult): ForkWindowState['messages'][number] {
  return {
    type: 'assistant_turn',
    source: 'agent',
    strategyId: 'native',
    estimatedTokens: 1,
    turn: {
      turnId: 'tool-turn',
      assistant: assistant(Option.some([{
        _tag: 'ToolCallPart',
        id: toolCallId,
        providerToolCallId,
        name: 'lookup',
        input: { query: 'magnitude' },
      }])),
      toolResults: [{ toolCallId, providerToolCallId, toolName: 'lookup', result }],
      feedback: [],
      clean: true,
    },
  }
}

const initialUser = {
  type: 'session_context' as const,
  source: 'system' as const,
  content: [{ _tag: 'ContextText' as const, text: 'Look this up' }],
  estimatedTokens: 1,
}

const formatResult = (entry: ToolResultEntry) => [{
  _tag: 'TextPart' as const,
  text: entry.result._tag === 'Success'
    ? String(entry.result.output)
    : entry.result._tag === 'Error'
      ? entry.result.error.message
      : entry.result._tag === 'InputRejected'
        ? 'Tool input rejected'
        : entry.result._tag,
}]

async function encode(messages: ForkWindowState['messages']) {
  const prompt = windowToPrompt({
    windowState: makeWindowState(messages),
    systemPrompt: 'system',
    timezone: null,
    formatter: formatResult,
    autopilotEnabled: false,
    leaderLastAutopilotKnowledge: null,
    includeImageData: false,
  })
  return Effect.runPromise(
    nativeChatCompletionsCodec.encodePrompt('test-model', prompt, []).pipe(
      Effect.flatMap((encoded) => finalizeChatCompletionsRequest({ ...encoded, stream: true })),
    ),
  )
}

function expectStrictAlternation(messages: readonly { readonly role: string; readonly tool_calls?: unknown }[]) {
  const counted = messages.filter((message) =>
    message.role === 'user' || (message.role === 'assistant' && message.tool_calls === undefined),
  )
  expect(counted.map((message) => message.role)).toEqual(
    counted.map((_, index) => index % 2 === 0 ? 'user' : 'assistant'),
  )
}

describe('strict-alternation tool history encoding', () => {
  it.each([
    ['successful result', { _tag: 'Success', output: 'result' } satisfies ToolResult],
    ['rejected input', {
      _tag: 'InputRejected',
      issue: { path: [], message: 'query is required' },
      partialInput: {},
    } as ToolResult],
    ['failed tool', { _tag: 'Error', error: { message: 'lookup failed' } } satisfies ToolResult],
  ])('keeps a %s in a tool-role message', async (_label, result) => {
    const request = await encode([initialUser, toolTurn(result)])

    expect(request.messages.map((message) => message.role)).toEqual(['system', 'user', 'assistant', 'tool'])
    expect(request.messages[3]).toMatchObject({
      role: 'tool',
      tool_call_id: providerToolCallId,
    })
    expectStrictAlternation(request.messages.filter((message) => message.role !== 'system'))
  })

  it('keeps a later user message after the assistant continuation', async () => {
    const continuation = {
      type: 'assistant_turn' as const,
      source: 'agent' as const,
      strategyId: 'native' as const,
      estimatedTokens: 1,
      turn: {
        turnId: 'continuation',
        assistant: {
          ...assistant(Option.none()),
          text: Option.some('The result is ready.'),
        },
        toolResults: [],
        feedback: [],
        clean: true,
      },
    }
    const laterUser = {
      type: 'session_context' as const,
      source: 'system' as const,
      content: [{ _tag: 'ContextText' as const, text: 'What about another query?' }],
      estimatedTokens: 1,
    }

    const request = await encode([
      initialUser,
      toolTurn({ _tag: 'Success', output: 'result' }),
      continuation,
      laterUser,
    ])

    expect(request.messages.map((message) => message.role)).toEqual([
      'system', 'user', 'assistant', 'tool', 'assistant', 'user',
    ])
    expectStrictAlternation(request.messages.filter((message) => message.role !== 'system'))
  })
})
