import { describe, expect, it } from 'vitest'
import { Option } from 'effect'
import type { Message, ProviderToolCallId, ToolCallId, JsonValue } from '@magnitudedev/ai'
import { observerWindowToPrompt } from '../src/observer'
import type { ForkWindowState, WindowEntry } from '../src/window'

function windowState(messages: readonly WindowEntry[]): ForkWindowState {
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

function textFromMessages(messages: readonly Message[]): string {
  return messages
    .flatMap((message) => {
      if (message._tag === 'UserMessage' || message._tag === 'ToolResultMessage') {
        return message.parts
          .filter((part) => part._tag === 'TextPart')
          .map((part) => part.text)
      }
      return Option.isSome(message.text) ? [message.text.value] : []
    })
    .join('\n')
}

const callId = (id: string) => id as ToolCallId
const providerCallId = (id: string) => id as ProviderToolCallId

describe('observerWindowToPrompt', () => {
  it('renders context with shared time markers and coalesced user messages', () => {
    const prompt = observerWindowToPrompt({
      windowState: windowState([
        {
          type: 'context',
          source: 'system',
          timeline: [
            { kind: 'user_message', timestamp: Date.UTC(2026, 5, 12, 17, 14, 3), text: 'first', attachments: [], synthetic: Option.none() },
            { kind: 'observation', timestamp: Date.UTC(2026, 5, 12, 17, 14, 10), parts: [{ _tag: 'TextPart', text: 'bookkeeping' }] },
            { kind: 'turn_start', timestamp: Date.UTC(2026, 5, 12, 17, 14, 12), turnId: 'turn-0' },
            {
              kind: 'user_message',
              timestamp: Date.UTC(2026, 5, 12, 17, 14, 21),
              text: 'second',
              synthetic: Option.none(),
              attachments: [{
                kind: 'mention',
                attachment: {
                  type: 'mention_file_range',
                  path: 'packages/agent/src/observer/prompt.ts',
                  startLine: 10,
                  endLine: 12,
                },
                resolution: {
                  status: 'resolved',
                  content: 'const marker = true',
                  truncated: true,
                  originalBytes: 123,
                },
              }],
            },
            { kind: 'user_message', timestamp: Date.UTC(2026, 5, 12, 17, 15, 0), text: 'third', attachments: [], synthetic: Option.none() },
          ],
          estimatedTokens: 1,
        },
      ]),
      systemPrompt: 'SYSTEM',
      observedForkId: null,
      timezone: 'UTC',
    })

    const text = textFromMessages(prompt.messages)
    expect(text).toContain('--- 2026-06-12 17:14:03 ---')
    expect(text).toContain('--- 17:15:00 ---')
    expect(text).toContain('<user>\nsecond\n<mention path="packages/agent/src/observer/prompt.ts" type="file" lines="10-12" truncated="true" original_bytes="123">const marker = true</mention>\n</user>')
    expect(text).toContain('<user>\nthird\n</user>')
    expect(text).not.toContain('at=')
    expect(text).not.toContain('<message from="user">')
  })

  it('omits session and fork context wrappers from the observer transcript', () => {
    const prompt = observerWindowToPrompt({
      windowState: windowState([
        {
          type: 'session_context',
          source: 'system',
          content: [{ _tag: 'TextPart', text: '<session_context>hidden session</session_context>' }],
          estimatedTokens: 1,
        },
        {
          type: 'fork_context',
          source: 'system',
          content: [{ _tag: 'TextPart', text: '<fork_context>hidden fork</fork_context>' }],
          estimatedTokens: 1,
        },
        {
          type: 'context',
          source: 'system',
          timeline: [
            { kind: 'user_message', timestamp: Date.UTC(2026, 5, 12, 17, 14, 3), text: 'visible user request', attachments: [], synthetic: Option.none() },
          ],
          estimatedTokens: 1,
        },
      ]),
      systemPrompt: 'SYSTEM',
      observedForkId: null,
      timezone: 'UTC',
    })

    const text = textFromMessages(prompt.messages)
    expect(text).toContain('visible user request')
    expect(text).not.toContain('hidden session')
    expect(text).not.toContain('hidden fork')
    expect(text).not.toContain('<session_context>')
    expect(text).not.toContain('<fork_context>')
  })

  it('renders observed turns with thoughts, message, feedback, and generic tool blocks', () => {
    const shellCallId = callId('call-shell')
    const grepCallId = callId('call-grep')
    const readCallId = callId('call-read')
    const noteCallId = callId('call-note')

    const prompt = observerWindowToPrompt({
      windowState: windowState([
        {
          type: 'assistant_turn',
          source: 'agent',
          strategyId: 'native',
          estimatedTokens: 1,
          turn: {
            turnId: 'turn-1',
            assistant: {
              _tag: 'AssistantMessage',
              reasoning: Option.some('the exact thought text'),
              text: Option.some('I found it.'),
              toolCalls: Option.some([
                {
                  _tag: 'ToolCallPart',
                  id: shellCallId,
                  providerToolCallId: providerCallId('provider-shell'),
                  name: 'shell',
                  input: { command: 'bunx --bun vitest run observer-window-prompt.vitest.ts', cwd: 'packages/agent' } as JsonValue,
                },
                {
                  _tag: 'ToolCallPart',
                  id: grepCallId,
                  providerToolCallId: providerCallId('provider-grep'),
                  name: 'grep',
                  input: { pattern: 'observerWindowToPrompt', path: 'packages/agent/src' } as JsonValue,
                },
                {
                  _tag: 'ToolCallPart',
                  id: readCallId,
                  providerToolCallId: providerCallId('provider-read'),
                  name: 'read',
                  input: { path: 'packages/agent/src/observer/prompt.ts' } as JsonValue,
                },
                {
                  _tag: 'ToolCallPart',
                  id: noteCallId,
                  providerToolCallId: providerCallId('provider-note'),
                  name: 'note',
                  input: 'plain input' as JsonValue,
                },
              ]),
            },
            toolResults: [
              {
                toolCallId: shellCallId,
                providerToolCallId: providerCallId('provider-shell'),
                toolName: 'shell',
                result: { _tag: 'Success', output: { exitCode: 0, stdout: 'ok' } },
              },
              {
                toolCallId: grepCallId,
                providerToolCallId: providerCallId('provider-grep'),
                toolName: 'grep',
                result: { _tag: 'Error', error: { message: 'grep failed' } },
              },
              {
                toolCallId: readCallId,
                providerToolCallId: providerCallId('provider-read'),
                toolName: 'read',
                result: { _tag: 'Interrupted' },
              },
              {
                toolCallId: noteCallId,
                providerToolCallId: providerCallId('provider-note'),
                toolName: 'note',
                result: { _tag: 'Success', output: 'plain output' },
              },
            ],
            feedback: [{ kind: 'interrupted' }],
            clean: true,
          },
        },
      ]),
      systemPrompt: 'SYSTEM',
      observedForkId: null,
      timezone: 'UTC',
    })

    const text = textFromMessages(prompt.messages)
    expect(text).toContain('<magnitude>')
    expect(text).toContain('<thoughts>\nthe exact thought text\n</thoughts>')
    expect(text).toContain('<message>\nI found it.\n</message>')
    expect(text).toContain('<feedback from="user">')
    expect(text).toContain('<shell>\n<params>')
    expect(text).toContain('<command>bunx --bun vitest run observer-window-prompt.vitest.ts</command>')
    expect(text).toContain('<cwd>packages/agent</cwd>')
    expect(text).not.toContain('<command>"')
    expect(text).not.toContain('<cwd>"')
    expect(text).toContain('<result>{exitCode: 0, stdout: "ok"}</result>')
    expect(text).toContain('<grep>\n<params>')
    expect(text).toContain('<error>{message: "grep failed"}</error>')
    expect(text).toContain('<read>\n<params>')
    expect(text).toContain('<interrupted/>')
    expect(text).toContain('<note>\n<params>\n<value>plain input</value>\n</params>\n<result>plain output</result>\n</note>')
    expect(text).not.toContain('<agent_turn')
    expect(text).not.toContain('<assistant_message>')
    expect(text).not.toContain('status=')
    expect(text).not.toContain('<tools shell=')
  })

  it('bounds tool params and results through JSON truncation', () => {
    const toolCallId = callId('call-large')
    const providerId = providerCallId('provider-large')
    const longCommand = 'x'.repeat(6000)
    const longOutput = 'z'.repeat(6000)

    const prompt = observerWindowToPrompt({
      windowState: windowState([
        {
          type: 'assistant_turn',
          source: 'agent',
          strategyId: 'native',
          estimatedTokens: 1,
          turn: {
            turnId: 'turn-large',
            assistant: {
              _tag: 'AssistantMessage',
              reasoning: Option.none(),
              text: Option.none(),
              toolCalls: Option.some([{
                _tag: 'ToolCallPart',
                id: toolCallId,
                providerToolCallId: providerId,
                name: 'shell',
                input: { command: longCommand, stable: 'kept' } as JsonValue,
              }]),
            },
            toolResults: [{
              toolCallId,
              providerToolCallId: providerId,
              toolName: 'shell',
              result: { _tag: 'Success', output: { stdout: longOutput } },
            }],
            feedback: [],
            clean: true,
          },
        },
      ]),
      systemPrompt: 'SYSTEM',
      observedForkId: null,
      timezone: 'UTC',
    })

    const text = textFromMessages(prompt.messages)
    expect(text).toContain('<params>')
    expect(text).toContain('<stable>kept</stable>')
    expect(text).not.toContain('<stable>"kept"</stable>')
    expect(text).toContain('<result>{')
    expect(text).toContain('stdout: "')
    expect(text).not.toContain(longCommand)
    expect(text).not.toContain(longOutput)
  })

  it('keeps prior observer turns as real pass/escalate tool history', () => {
    const prompt = observerWindowToPrompt({
      windowState: windowState([
        {
          type: 'observer_turn',
          source: 'system',
          observerTurnId: 'observer-1',
          estimatedTokens: 1,
          justification: null,
          escalate: false,
          reasoning: 'Agent is operating acceptably.',
        },
      ]),
      systemPrompt: 'SYSTEM',
      observedForkId: null,
      timezone: 'UTC',
    })

    const assistant = prompt.messages.find((message) => message._tag === 'AssistantMessage')
    const toolResult = prompt.messages.find((message) => message._tag === 'ToolResultMessage')
    expect(assistant?._tag).toBe('AssistantMessage')
    if (assistant?._tag !== 'AssistantMessage') throw new Error('expected assistant message')
    expect(Option.getOrNull(assistant.toolCalls)?.[0]?.name).toBe('pass')
    expect(toolResult?._tag).toBe('ToolResultMessage')
    if (toolResult?._tag !== 'ToolResultMessage') throw new Error('expected tool result')
    expect(toolResult.toolName).toBe('pass')
    expect(textFromMessages(prompt.messages)).not.toContain('<observer_history>')
  })

  it('keeps prior observer escalate turns as real escalate tool history', () => {
    const prompt = observerWindowToPrompt({
      windowState: windowState([
        {
          type: 'observer_turn',
          source: 'system',
          observerTurnId: 'observer-2',
          estimatedTokens: 1,
          justification: 'churn',
          escalate: true,
          reasoning: 'Agent has repeated the same approach multiple times.',
        },
      ]),
      systemPrompt: 'SYSTEM',
      observedForkId: null,
      timezone: 'UTC',
    })

    const assistant = prompt.messages.find((message) => message._tag === 'AssistantMessage')
    const toolResult = prompt.messages.find((message) => message._tag === 'ToolResultMessage')
    expect(assistant?._tag).toBe('AssistantMessage')
    if (assistant?._tag !== 'AssistantMessage') throw new Error('expected assistant message')
    expect(Option.getOrNull(assistant.toolCalls)?.[0]?.name).toBe('escalate')
    expect(Option.getOrNull(assistant.toolCalls)?.[0]?.input).toEqual({ justification: 'churn' })
    expect(toolResult?._tag).toBe('ToolResultMessage')
    if (toolResult?._tag !== 'ToolResultMessage') throw new Error('expected tool result')
    expect(toolResult.toolName).toBe('escalate')
  })
})
