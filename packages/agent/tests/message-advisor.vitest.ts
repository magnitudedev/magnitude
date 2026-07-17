import { describe, expect, it } from 'vitest'
import { Effect, Layer, Stream, Option } from 'effect'
import { FetchHttpClient } from '@effect/platform'
import { ModelFamilyIdSchema, ProviderIdSchema, ProviderModelIdSchema, ReasoningEffortSchema, ReasoningProperty, VisionProperty } from '@magnitudedev/ai'
import { AmbientServiceTag, Fork, type AmbientService } from '@magnitudedev/event-core'
import {
  ModelStreamTerminal,
  Prompt,
  type BoundModel,
  type Message,
  type ModelSpec,
  type ModelStreamResult,
  type ResponseStreamEvent,
  type ToolCallId,
  type ProviderToolCallId,
  type ToolDefinition,
} from '@magnitudedev/ai'
import type {
  BaseCallOptions,
  ProviderRejection,
  MagnitudeModelInfo,
} from '@magnitudedev/sdk'
import { SlotReady, type SlotStates } from '@magnitudedev/sdk'
import type { JsonValue } from '@magnitudedev/ai'
import { advisorWindowToPrompt } from '../src/window/render'
import type { ForkWindowState, WindowEntry } from '../src/window'
import { executeMessageAdvisor } from '../src/tools/advisor'
import { WindowStateReaderTag } from '../src/tools/window-reader'
import { AgentStateReaderTag } from '../src/tools/fork'
import { AgentModelResolver } from '../src/model/model-resolver'
import { makeAgentBoundModel } from '../src/model/agent-model'
import { leaderToolkit, isToolKey, selectAgentToolKeys } from '../src/tools/toolkits'
import { buildConfigStateFromSlots } from '../src/ambient/config-ambient'

const { ForkContext } = Fork

const profile = {
  contextWindow: 100_000,
  maxOutputTokens: 4096,
}

const testModelSpec: ModelSpec<BaseCallOptions> = {
  modelId: 'role/advisor',
  endpoint: 'http://test',
  bind: () => { throw new Error('not used') },
  _execute: () => { throw new Error('not used') },
}

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

function eventsFor(text: string): readonly ResponseStreamEvent[] {
  return [
    { _tag: 'message_start' },
    { _tag: 'message_delta', text },
    { _tag: 'message_end' },
    {
      _tag: 'stream_end',
      terminal: ModelStreamTerminal.StreamCompleted({
        call: { provider: 'test', model: 'test', method: 'POST', url: 'http://test' },
        response: { status: 200, headers: [], requestId: null },
        finishReason: 'stop',
        progress: { dataPayloadsDecoded: 1, modelEventsEmitted: 1 },
        usage: { _tag: 'UsageNotReported', reason: 'provider_does_not_report_usage' },
      }),
    },
  ]
}

function makeWindowFixture(): ForkWindowState {
  const shellCallId1 = 'call-shell-1' as ToolCallId
  const shellProviderCallId1 = 'provider-call-shell-1' as ProviderToolCallId
  const patchCallId = 'call-patch-1' as ToolCallId
  const patchProviderCallId = 'provider-call-patch-1' as ProviderToolCallId
  const shellCallId2 = 'call-shell-2' as ToolCallId
  const shellProviderCallId2 = 'provider-call-shell-2' as ProviderToolCallId

  return windowState([
    {
      type: 'session_context',
      source: 'system',
      content: [{ _tag: 'ContextText', text: 'SESSION SHOULD NOT APPEAR' }],
      estimatedTokens: 1,
    },
    {
      type: 'compacted',
      source: 'system',
      content: [{ _tag: 'ContextText', text: 'COMPACT SUMMARY' }],
      estimatedTokens: 1,
    },
    {
      type: 'fork_context',
      source: 'system',
      content: [{ _tag: 'ContextText', text: 'FORK SHOULD NOT APPEAR' }],
      estimatedTokens: 1,
    },
    {
      type: 'context',
      source: 'system',
      timeline: [
        { kind: 'user_message', timestamp: 1, items: [{ kind: 'body', parts: [{ _tag: 'ContextText', text: 'Please implement advisor.' }] }], synthetic: Option.none() },
        { kind: 'observation', timestamp: 2, parts: [{ _tag: 'ContextText', text: 'RAW OBSERVATION OUTPUT' }] },
      ],
      estimatedTokens: 1,
    },
    {
      type: 'assistant_turn',
      source: 'agent',
      strategyId: 'native',
      estimatedTokens: 1,
      turn: {
        turnId: 'turn-1',
        assistant: {
          _tag: 'AssistantMessage',
          reasoning: Option.none(),
          text: Option.some('I am wiring the tool.'),
          toolCalls: Option.some([
            {
              _tag: 'ToolCallPart',
              id: shellCallId1,
              providerToolCallId: shellProviderCallId1,
              name: 'shell',
              input: { cmd: 'echo secret' } as JsonValue,
            },
            {
              _tag: 'ToolCallPart',
              id: patchCallId,
              providerToolCallId: patchProviderCallId,
              name: 'apply_patch',
              input: { patch: 'secret patch' } as JsonValue,
            },
          ]),
        },
        toolResults: [
          {
            toolCallId: shellCallId1,
            providerToolCallId: shellProviderCallId1,
            toolName: 'shell',
            result: { _tag: 'Success', output: 'SECRET RAW OUTPUT' },
          },
          {
            toolCallId: patchCallId,
            providerToolCallId: patchProviderCallId,
            toolName: 'apply_patch',
            result: { _tag: 'Success', output: 'SECRET PATCH OUTPUT' },
          },
        ],
        feedback: [],
        clean: true,
      },
    },
    {
      type: 'assistant_turn',
      source: 'agent',
      strategyId: 'native',
      estimatedTokens: 1,
      turn: {
        turnId: 'turn-2',
        assistant: {
          _tag: 'AssistantMessage',
          reasoning: Option.none(),
          text: Option.none(),
          toolCalls: Option.some([{
            _tag: 'ToolCallPart',
            id: shellCallId2,
            providerToolCallId: shellProviderCallId2,
            name: 'shell',
            input: { cmd: 'echo secret again' } as JsonValue,
          }]),
        },
        toolResults: [{
          toolCallId: shellCallId2,
          providerToolCallId: shellProviderCallId2,
          toolName: 'shell',
          result: { _tag: 'Success', output: 'SECOND SECRET RAW OUTPUT' },
        }],
        feedback: [],
        clean: true,
      },
    },
  ])
}

function makeModel(text: string) {
  const calls: Array<{
    readonly prompt: Prompt
    readonly tools: readonly ToolDefinition[]
    readonly options: (BaseCallOptions & { readonly generateToolCallId?: () => ToolCallId }) | undefined
  }> = []

  const model: BoundModel<BaseCallOptions> = {
    stream: (prompt, tools, options) => {
      calls.push({ prompt, tools, options })
      const result: ModelStreamResult = {
        events: Stream.fromIterable(eventsFor(text)),
        parsers: new Map(),
        logprobs: [],
        requestId: null,
      }
      return Effect.succeed(result)
    },
  }

  return { model, calls }
}

function makeResolver(model: BoundModel<BaseCallOptions>) {
  const resolved = makeAgentBoundModel({
    rawModel: model,
    modelSource: { slotId: 'primary' },
    modelId: 'role/advisor',
    providerId: 'magnitude',
    profile,
    debug: false,
    agentId: 'advisor',
    roleId: 'advisor',
  })

  return Layer.succeed(AgentModelResolver, {
    resolveSlotConfig: () => Effect.succeed(resolved),
    resolvePrimary: () => Effect.succeed(resolved),
    resolveSecondary: () => Effect.die('not used'),
  })
}

function runAdvisor(
  effect: ReturnType<typeof executeMessageAdvisor>,
  model: BoundModel<BaseCallOptions>,
  state: ForkWindowState = makeWindowFixture(),
) {
  const ambientService: AmbientService = {
    register: () => Effect.void,
    getValue: () => { throw new Error('not used') },
    update: () => Effect.void,
  }

  const layer = Layer.mergeAll(
    Layer.succeed(ForkContext, { forkId: null, roleId: 'leader' }),
    Layer.succeed(AmbientServiceTag, ambientService),
    Layer.succeed(WindowStateReaderTag, {
      getWindowState: () => Effect.succeed(state),
    }),
    Layer.succeed(AgentStateReaderTag, {
      getAgentState: () => Effect.succeed({ agents: new Map(), agentByForkId: new Map(), rootWork: { phase: 'idle', chainStartedAt: null, lastChainMs: 0, activity: null, activeChildCount: 0, _currentTurnId: null, _thinkingCharCount: null, _activeToolKey: null } }),
      getAgent: () => Effect.sync(() => undefined),
    }),
    makeResolver(model),
    FetchHttpClient.layer,
  )

  return Effect.runPromise(effect.pipe(Effect.provide(layer)))
}

describe('advisorWindowToPrompt', () => {
  it('renders a filtered context window for advisor consultation', () => {
    const prompt = advisorWindowToPrompt({
      windowState: makeWindowFixture(),
      systemPrompt: 'ADVISOR SYSTEM',
      autopilotEnabled: false,
      advisorLastAutopilotKnowledge: null,
      messageAdvisorText: 'What am I missing?',
    })

    const text = `${prompt.system}\n${textFromMessages(prompt.messages)}`

    expect(text).toContain('ADVISOR SYSTEM')
    expect(text).toContain('COMPACT SUMMARY')
    expect(text).toContain('<message from="user">Please implement advisor.</message>')
    expect(text).toContain('<message>I am wiring the tool.</message>')
    expect(text).toContain('<tools shell=1 apply_patch=1 />')
    expect(text).toContain('<message_advisor>')
    expect(text).toContain('What am I missing?')
    expect(text).not.toContain('SESSION SHOULD NOT APPEAR')
    expect(text).not.toContain('FORK SHOULD NOT APPEAR')
    expect(text).not.toContain('RAW OBSERVATION OUTPUT')
    expect(text).not.toContain('SECRET RAW OUTPUT')
    expect(text).not.toContain('SECRET PATCH OUTPUT')
    expect(text).not.toContain('SECOND SECRET RAW OUTPUT')
    expect(text).not.toContain('<work_summary')
    expect(text).not.toContain('- shell: success')
  })
})

describe('message_advisor execution', () => {
  it('calls role/advisor with no tools and returns the message text', async () => {
    const { model, calls } = makeModel('Run the focused test and inspect hidden-tool rendering.')

    const response = await runAdvisor(
      executeMessageAdvisor({ message: 'Check my plan.' }),
      model,
    )

    expect(response).toBe('Run the focused test and inspect hidden-tool rendering.')
    expect(calls).toHaveLength(1)
    expect(calls[0]?.tools).toEqual([])
    expect(calls[0]?.options?.maxTokens).toBe(1200)
    expect(textFromMessages(calls[0]?.prompt.messages ?? [])).toContain('Check my plan.')
  })

  it('rejects empty advisor responses', async () => {
    const { model } = makeModel('   ')

    await expect(
      runAdvisor(executeMessageAdvisor({ message: 'Check my plan.' }), model),
    ).rejects.toThrow('Advisor returned an empty response.')
  })
})

describe('messageAdvisor toolkit registration', () => {
  it('keeps static advisor registration but removes it from effective toolkits while disabled', () => {
    const catalogModels: readonly MagnitudeModelInfo[] = [
      { providerModelId: ProviderModelIdSchema.make('primary-model'), providerId: ProviderIdSchema.make('magnitude'), modelFamilyId: ModelFamilyIdSchema.make('unknown'), availability: { _tag: 'Available' }, defaultReasoningEffort: ReasoningEffortSchema.make('none'), properties: { vision: new VisionProperty.states.Resolved({ value: true }), reasoning: new ReasoningProperty.states.Resolved({ value: [ReasoningEffortSchema.make('none')] }) }, slots: ['primary'] as readonly ("primary" | "secondary")[], displayName: 'Primary', contextWindow: 200_000, maxOutputTokens: 16_384, object: 'model' as const, owned_by: 'magnitude', roles: [], pricing: { input: 0, output: 0, cached_input: null } },
      { providerModelId: ProviderModelIdSchema.make('secondary-model'), providerId: ProviderIdSchema.make('magnitude'), modelFamilyId: ModelFamilyIdSchema.make('unknown'), availability: { _tag: 'Available' }, defaultReasoningEffort: ReasoningEffortSchema.make('none'), properties: { vision: new VisionProperty.states.Resolved({ value: true }), reasoning: new ReasoningProperty.states.Resolved({ value: [ReasoningEffortSchema.make('none')] }) }, slots: ['secondary'] as readonly ("primary" | "secondary")[], displayName: 'Secondary', contextWindow: 200_000, maxOutputTokens: 16_384, object: 'model' as const, owned_by: 'magnitude', roles: [], pricing: { input: 0, output: 0, cached_input: null } },
    ]
    const slots: SlotStates = {
      primary: new SlotReady({ slotId: 'primary', selection: { providerId: catalogModels[0]!.providerId, providerModelId: catalogModels[0]!.providerModelId, reasoningEffort: catalogModels[0]!.defaultReasoningEffort }, source: 'automatic', modelDisplayName: 'Primary', contextWindow: 200_000, maxOutputTokens: 16_384 }),
      secondary: new SlotReady({ slotId: 'secondary', selection: { providerId: catalogModels[1]!.providerId, providerModelId: catalogModels[1]!.providerModelId, reasoningEffort: catalogModels[1]!.defaultReasoningEffort }, source: 'automatic', modelDisplayName: 'Secondary', contextWindow: 200_000, maxOutputTokens: 16_384 }),
    }
    const config = buildConfigStateFromSlots(catalogModels, slots, {
      softCapRatio: 0.9,
      softCapMaxTokens: 200_000,
    })

    expect(leaderToolkit.entries.messageAdvisor).toBeDefined()
    expect(isToolKey('messageAdvisor')).toBe(true)
    expect(selectAgentToolKeys({ roleId: 'leader', configState: config, solo: false, vcsAvailable: false })).not.toContain('messageAdvisor')
    expect(selectAgentToolKeys({ roleId: 'engineer', configState: config, solo: false, vcsAvailable: false })).not.toContain('messageAdvisor')
  })
})
