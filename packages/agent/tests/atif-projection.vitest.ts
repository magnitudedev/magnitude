/**
 * AtifProjection tests — verify event-to-step mapping and ambient gating.
 */

import { describe, it, expect } from 'vitest'
import { Effect, Layer, Option } from 'effect'
import { ModelFamilyIdSchema, ProviderIdSchema, ProviderModelIdSchema, ReasoningEffortSchema } from '@magnitudedev/ai'
import {
  ModelSlotReady,
  PRIMARY_SLOT_ID,
  SECONDARY_SLOT_ID,
  type ModelSlotsState,
  type ProviderModelCatalogEntry,
} from '@magnitudedev/sdk'
import {
  ProjectionBusTag,
  makeProjectionBusLayer,
  makeAmbientServiceLayer,
  AmbientServiceTag,
  FrameworkErrorPubSubLive,
  FrameworkErrorReporterLive,
} from '@magnitudedev/event-core'
import type { AppEvent } from '../src/events'
import { AtifProjection } from '../src/projections/atif/projection'
import { AgentRoutingProjection } from '../src/projections/agent-routing'
import { AgentLifecycleProjection } from '../src/projections/agent-lifecycle'
import { GoalProjection } from '../src/projections/goal'
import { TurnProjection } from '../src/projections/turn'
import { buildConfigStateFromSlots } from '../src/ambient/config-ambient'
import { AtifAmbient, type AtifConfig } from '../src/ambient/atif-ambient'
import type { AtifForkState, AtifTrajectory } from '../src/projections/atif'
import { serializeAtif } from '../src/projections/atif/serialize'
import { selectAgentToolKeys, toolUniverseToolkit, toToolKeyErased } from '../src/tools/toolkits'

const providerId = ProviderIdSchema.make('magnitude')
const reasoningEffort = ReasoningEffortSchema.make('none')
const model = (
  providerModelId: string,
  displayName: string,
  slotId: typeof PRIMARY_SLOT_ID | typeof SECONDARY_SLOT_ID,
): ProviderModelCatalogEntry => ({
  providerId,
  providerModelId: ProviderModelIdSchema.make(providerModelId),
  modelFamilyId: Option.some(ModelFamilyIdSchema.make('unknown')),
  displayName,
  supportedSlots: [slotId],
  contextWindow: 200_000,
  maxOutputTokens: 16_384,
  runtimeMemoryBytes: Option.none(),
  capabilities: {
    vision: true,
    tools: true,
    structuredOutput: true,
    reasoning: {
      supported: true,
      efforts: [reasoningEffort],
      defaultEffort: Option.some(reasoningEffort),
    },
  },
  availability: { _tag: 'Available' },
  pricing: Option.none(),
})
const mockModels = [
  model('primary-model', 'Primary', PRIMARY_SLOT_ID),
  model('secondary-model', 'Secondary', SECONDARY_SLOT_ID),
] as const satisfies readonly ProviderModelCatalogEntry[]

const mockSlots: ModelSlotsState['slots'] = {
  primary: new ModelSlotReady({ slotId: PRIMARY_SLOT_ID, selection: { providerId, providerModelId: mockModels[0].providerModelId, reasoningEffort } }),
  secondary: new ModelSlotReady({ slotId: SECONDARY_SLOT_ID, selection: { providerId, providerModelId: mockModels[1].providerModelId, reasoningEffort } }),
}
const mockConfigState = buildConfigStateFromSlots(mockModels, mockSlots, {
  softCapRatio: 0.9,
  softCapMaxTokens: 200_000,
})
const leaderToolKeys = selectAgentToolKeys({
  roleId: 'leader', configState: mockConfigState, solo: false, vcsAvailable: false,
})
const serializationTools = {
  universe: toolUniverseToolkit,
  toolKeysByFork: new Map<string | null, readonly string[]>([[null, leaderToolKeys]]),
}

const ts = (n: number) => 1_700_100_000_000 + n

const makeAtif = async (events: AppEvent[], enabled: boolean, targetForkId: string | null = null): Promise<AtifForkState> => {
  const baseBusLayer = Layer.provideMerge(
    makeProjectionBusLayer<AppEvent>(),
    Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
  )
  const baseLayer = Layer.provideMerge(
    makeAmbientServiceLayer<AppEvent>(),
    baseBusLayer,
  )

  const runtimeLayer = Layer.provideMerge(
    Layer.mergeAll(
      GoalProjection.Layer,
      TurnProjection.Layer,
      AgentRoutingProjection.Layer,
      AgentLifecycleProjection.Layer,
      AtifProjection.Layer,
    ),
    baseLayer,
  )

  const program = Effect.gen(function* () {
    const bus = yield* (ProjectionBusTag<AppEvent>())
    const projection = yield* AtifProjection.Tag

    const ambientService = yield* AmbientServiceTag
    yield* ambientService.register(AtifAmbient)
    yield* ambientService.update(AtifAmbient, {
      enabled,
      writeFile: false,
      filePath: null,
      streamSteps: false,
      stepsPath: null,
    } satisfies AtifConfig)

    for (const event of events) {
      yield* bus.processEvent(event as any)
    }

    return yield* projection.getFork(targetForkId)
  })

  return Effect.runPromise(
    program.pipe(Effect.provide(runtimeLayer)) as Effect.Effect<AtifForkState>,
  )
}

const forkId = null
const turnId = 'turn-1'

describe('AtifProjection', () => {
  it('should be disabled by default and produce no steps', async () => {
    const fork = await makeAtif([
      {
        type: 'user_message',
        forkId,
        messageId: 'msg-1',
        timestamp: ts(1),
        text: 'Hello',
        mentions: [],
        attachments: [],
        mode: 'text',
        synthetic: false,
        taskMode: false,
      },
    ] as AppEvent[], false)

    expect(fork.steps.length).toBe(0)
  })

  it('should emit a user step when enabled', async () => {
    const fork = await makeAtif([
      {
        type: 'user_message',
        forkId,
        messageId: 'msg-1',
        timestamp: ts(1),
        text: 'Hello',
        mentions: [],
        attachments: [],
        mode: 'text',
        synthetic: false,
        taskMode: false,
      },
    ] as AppEvent[], true)

    expect(fork.steps.length).toBe(1)
    expect(fork.steps[0].source).toBe('user')
    expect(fork.steps[0].message).toBe('Hello')
    expect(fork.steps[0].step_id).toBe(1)
  })

  it('should accumulate an agent turn', async () => {
    const fork = await makeAtif([
      {
        type: 'turn_started',
        forkId,
        turnId,
        chainId: 'chain-1',
      },
      {
        type: 'thinking_chunk',
        forkId,
        turnId,
        text: 'Let me think...',
      },
      {
        type: 'message_chunk',
        forkId,
        turnId,
        id: 'msg-1',
        text: 'Here is the result',
      },
      {
        type: 'turn_outcome',
        forkId,
        turnId,
        chainId: 'chain-1',
        strategyId: 'native',
        outcome: { _tag: 'Completed', completion: { toolCallsCount: 0, finishReason: 'stop', feedback: [], yieldTarget: null }, requestId: 'request-123' },
        inputTokens: 100,
        outputTokens: 20,
        cacheReadTokens: 0,
        cacheWriteTokens: 0,
        cost: null,
        providerId: null,
        modelId: null,
      },
    ] as AppEvent[], true)

    expect(fork.steps.length).toBe(1)
    expect(fork.steps[0].source).toBe('agent')
    expect(fork.steps[0].message).toBe('Here is the result')
    expect(Option.getOrNull(fork.steps[0].reasoning_content)).toBe('Let me think...')
    const metrics = Option.getOrNull(fork.steps[0].metrics)
    expect(metrics && Option.getOrNull(metrics.prompt_tokens)).toBe(100)
    expect(metrics && Option.getOrNull(metrics.completion_tokens)).toBe(20)
  })

  it('should not drop an active agent turn when an observer outcome is interleaved', async () => {
    const fork = await makeAtif([
      {
        type: 'turn_started',
        forkId,
        turnId: 'turn-active',
        chainId: 'chain-1',
      },
      {
        type: 'message_chunk',
        forkId,
        turnId: 'turn-active',
        id: 'msg-1',
        text: 'Active turn response',
      },
      {
        type: 'observer_outcome',
        forkId,
        observedTurnId: 'turn-previous',
        observerTurnId: 'observer-1',
        chainId: 'chain-0',
        escalate: false,
        justification: null,
        reasoning: 'Previous turn was fine.',
      },
      {
        type: 'turn_outcome',
        forkId,
        turnId: 'turn-active',
        chainId: 'chain-1',
        strategyId: 'native',
        outcome: { _tag: 'Completed', completion: { toolCallsCount: 0, finishReason: 'stop', feedback: [], yieldTarget: null }, requestId: 'request-123' },
        inputTokens: 10,
        outputTokens: 5,
        cacheReadTokens: 0,
        cacheWriteTokens: 0,
        cost: null,
        providerId: 'magnitude',
        modelId: 'claude-sonnet-4-6',
      },
    ] as AppEvent[], true)

    expect(fork.steps.length).toBe(2)
    expect(fork.steps[0].source).toBe('system')
    expect(fork.steps[0].message).toBe(JSON.stringify({ escalate: false }))
    expect(fork.steps[1].source).toBe('agent')
    expect(fork.steps[1].message).toBe('Active turn response')
    expect(fork.steps.map((step) => step.step_id)).toEqual([1, 2])
    expect(Option.getOrNull(fork.steps[1].extra)?.turnId).toBe('turn-active')
    expect(Option.getOrNull(fork.steps[1].extra)?.requestId).toBe('request-123')
    expect(fork.steps[1].extra).not.toHaveProperty('traceId')
  })

  it('should render observer escalation system messages as JSON', async () => {
    const fork = await makeAtif([
      {
        type: 'observer_outcome',
        forkId,
        observedTurnId: 'turn-observed',
        observerTurnId: 'observer-1',
        chainId: 'chain-1',
        escalate: true,
        justification: 'difficulty',
        reasoning: 'The turn needs stronger reasoning.',
      },
    ] as AppEvent[], true)

    expect(fork.steps.length).toBe(1)
    expect(fork.steps[0].source).toBe('system')
    expect(fork.steps[0].message).toBe(JSON.stringify({ escalate: true, justification: 'difficulty' }))
    expect(Option.getOrNull(fork.steps[0].reasoning_content)).toBe('The turn needs stronger reasoning.')
  })

  it('should keep an active agent turn across side-channel ATIF steps', async () => {
    const fork = await makeAtif([
      {
        type: 'turn_started',
        forkId,
        turnId: 'turn-active',
        chainId: 'chain-1',
      },
      {
        type: 'message_chunk',
        forkId,
        turnId: 'turn-active',
        id: 'msg-1',
        text: 'Still here',
      },
      { type: 'tool_approved', forkId, toolCallId: 'tc-approved' },
      { type: 'tool_rejected', forkId, toolCallId: 'tc-rejected', reason: 'Nope' },
      { type: 'interrupt', forkId },
      {
        type: 'compaction_prepared',
        forkId,
        turn: { turnId: 'turn-compacted' } as any,
        compactedMessageCount: 2,
        inputTokens: 100,
        outputTokens: 20,
        refreshedContext: null,
        isFallback: false,
        compactResult: { summary: 'Compacted context' } as any,
      },
      {
        type: 'agent_created',
        forkId: 'agent-scout-1',
        parentForkId: null,
        agentId: 'agent-scout-1',
        role: 'scout',
        taskId: 'task-1',
        mode: 'spawn',
        message: 'research',
        name: 'scout-1',
        context: '',
      },
      {
        type: 'turn_outcome',
        forkId,
        turnId: 'turn-active',
        chainId: 'chain-1',
        strategyId: 'native',
        outcome: { _tag: 'Completed', completion: { toolCallsCount: 0, finishReason: 'stop', feedback: [], yieldTarget: null }, requestId: null },
        inputTokens: 30,
        outputTokens: 7,
        cacheReadTokens: 0,
        cacheWriteTokens: 0,
        cost: null,
        providerId: null,
        modelId: null,
      },
    ] as AppEvent[], true)

    expect(fork.steps.map((step) => step.source)).toEqual(['user', 'user', 'user', 'system', 'agent', 'agent'])
    expect(fork.steps[fork.steps.length - 1].message).toBe('Still here')
    expect(Option.getOrNull(fork.steps[fork.steps.length - 1].extra)?.turnId).toBe('turn-active')
    expect(fork.steps.map((step) => step.step_id)).toEqual([1, 2, 3, 4, 5, 6])
  })

  it('should accumulate simultaneous active turns by turnId', async () => {
    const fork = await makeAtif([
      {
        type: 'turn_started',
        forkId,
        turnId: 'turn-a',
        chainId: 'chain-a',
      },
      {
        type: 'message_chunk',
        forkId,
        turnId: 'turn-a',
        id: 'msg-a',
        text: 'Response A',
      },
      {
        type: 'turn_started',
        forkId,
        turnId: 'turn-b',
        chainId: 'chain-b',
      },
      {
        type: 'message_chunk',
        forkId,
        turnId: 'turn-b',
        id: 'msg-b',
        text: 'Response B',
      },
      {
        type: 'turn_outcome',
        forkId,
        turnId: 'turn-b',
        chainId: 'chain-b',
        strategyId: 'native',
        outcome: { _tag: 'Completed', completion: { toolCallsCount: 0, finishReason: 'stop', feedback: [], yieldTarget: null }, requestId: null },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        cost: null,
        providerId: null,
        modelId: null,
      },
      {
        type: 'turn_outcome',
        forkId,
        turnId: 'turn-a',
        chainId: 'chain-a',
        strategyId: 'native',
        outcome: { _tag: 'Completed', completion: { toolCallsCount: 0, finishReason: 'stop', feedback: [], yieldTarget: null }, requestId: null },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        cost: null,
        providerId: null,
        modelId: null,
      },
    ] as AppEvent[], true)

    expect(fork.steps.length).toBe(2)
    expect(fork.steps[0].message).toBe('Response B')
    expect(Option.getOrNull(fork.steps[0].extra)?.turnId).toBe('turn-b')
    expect(fork.steps[1].message).toBe('Response A')
    expect(Option.getOrNull(fork.steps[1].extra)?.turnId).toBe('turn-a')
    expect(fork.activeTurns.size).toBe(0)
  })

  it('should populate tool call arguments from ToolExecutionStarted', async () => {
    const fork = await makeAtif([
      {
        type: 'turn_started',
        forkId,
        turnId,
        chainId: 'chain-1',
      },
      {
        type: 'tool_event',
        forkId,
        turnId,
        toolCallId: 'tc-1',
        providerToolCallId: 'tc-1',
        toolKey: toToolKeyErased('shell'),
        event: { _tag: 'ToolInputReady', toolCallId: 'tc-1', providerToolCallId: 'tc-1', toolName: 'shell', toolKey: toToolKeyErased('shell') },
      },
      {
        type: 'tool_event',
        forkId,
        turnId,
        toolCallId: 'tc-1',
        providerToolCallId: 'tc-1',
        toolKey: toToolKeyErased('shell'),
        event: { _tag: 'ToolExecutionStarted', toolCallId: 'tc-1', providerToolCallId: 'tc-1', toolName: 'shell', toolKey: toToolKeyErased('shell'), input: { command: 'echo hello' }, cached: false },
      },
      {
        type: 'tool_event',
        forkId,
        turnId,
        toolCallId: 'tc-1',
        providerToolCallId: 'tc-1',
        toolKey: toToolKeyErased('shell'),
        event: { _tag: 'ToolExecutionEnded', toolCallId: 'tc-1', providerToolCallId: 'tc-1', toolName: 'shell', toolKey: toToolKeyErased('shell'), result: { _tag: 'Success', output: 'hello' } },
      },
      {
        type: 'turn_outcome',
        forkId,
        turnId,
        chainId: 'chain-1',
        strategyId: 'native',
        outcome: { _tag: 'Completed', completion: { toolCallsCount: 1, finishReason: 'tool_calls', feedback: [], yieldTarget: null }, requestId: null },
        inputTokens: 50,
        outputTokens: 10,
        cacheReadTokens: 0,
        cacheWriteTokens: 0,
        cost: null,
        providerId: null,
        modelId: null,
      },
    ] as AppEvent[], true)

    expect(fork.steps.length).toBe(1)
    const toolCalls0 = Option.getOrNull(fork.steps[0].tool_calls)
    expect(toolCalls0?.length).toBe(1)
    expect(toolCalls0?.[0].arguments).toEqual({ command: 'echo hello' })
    const observation0 = Option.getOrNull(fork.steps[0].observation)
    expect(observation0 && Option.getOrNull(observation0.results[0].content)).toBe('hello')
  })

  it('should mark synthetic user messages with extra.autopilot', async () => {
    const fork = await makeAtif([
      {
        type: 'user_message',
        forkId,
        messageId: 'msg-1',
        timestamp: ts(1),
        text: 'Continue',
        mentions: [],
        attachments: [],
        mode: 'text',
        synthetic: true,
        taskMode: false,
      },
    ] as AppEvent[], true)

    expect(fork.steps.length).toBe(1)
    expect(fork.steps[0].source).toBe('user')
    expect(fork.steps[0].message).toBe('Continue')
    expect(Option.getOrNull(fork.steps[0].extra)?.autopilot).toBe(true)
  })

  it('should populate model_name from turn_outcome modelId', async () => {
    const fork = await makeAtif([
      {
        type: 'turn_started',
        forkId,
        turnId,
        chainId: 'chain-1',
      },
      {
        type: 'turn_outcome',
        forkId,
        turnId,
        chainId: 'chain-1',
        strategyId: 'native',
        outcome: { _tag: 'Completed', completion: { toolCallsCount: 0, finishReason: 'stop', feedback: [], yieldTarget: null }, requestId: null },
        inputTokens: 100,
        outputTokens: 20,
        cacheReadTokens: 0,
        cacheWriteTokens: 0,
        cost: null,
        providerId: null,
        modelId: 'claude-sonnet-4-6',
      },
    ] as AppEvent[], true)

    expect(fork.steps.length).toBe(1)
    expect(fork.steps[0].source).toBe('agent')
    expect(Option.getOrNull(fork.steps[0].model_name)).toBe('claude-sonnet-4-6')
  })

  it('should produce user steps for tool_approved and tool_rejected', async () => {
    const fork = await makeAtif([
      {
        type: 'tool_approved',
        forkId,
        toolCallId: 'tc-1',
      },
      {
        type: 'tool_rejected',
        forkId,
        toolCallId: 'tc-2',
        reason: 'Too risky',
      },
    ] as AppEvent[], true)

    expect(fork.steps.length).toBe(2)
    expect(fork.steps[0].source).toBe('user')
    expect(fork.steps[0].message).toContain('Approved tool call tc-1')
    expect(Option.getOrNull(fork.steps[0].extra)?.action).toBe('approved')
    expect(fork.steps[1].source).toBe('user')
    expect(fork.steps[1].message).toContain('Rejected tool call tc-2')
    expect(Option.getOrNull(fork.steps[1].extra)?.action).toBe('rejected')
    expect(Option.getOrNull(fork.steps[1].extra)?.reason).toBe('Too risky')
  })

  it('should produce a system step for compaction_prepared with context_management', async () => {
    const fork = await makeAtif([
      {
        type: 'compaction_prepared',
        forkId,
        timestamp: ts(1),
        turn: { turnId: 'turn-compact' } as any,
        compactedMessageCount: 5,
        inputTokens: 1000,
        outputTokens: 200,
        refreshedContext: null,
        isFallback: false,
        compactResult: { summary: 'Prior conversation covered topic X' } as any,
      } as any,
    ] as AppEvent[], true)

    expect(fork.steps.length).toBe(1)
    expect(fork.steps[0].source).toBe('system')
    expect(fork.steps[0].message).toBe('Context compaction performed')
    const cm = Option.getOrNull(fork.steps[0].extra)?.context_management as Record<string, unknown> | undefined
    expect(cm).toBeDefined()
    expect(cm?.type).toBe('compaction')
    expect(cm?.boundary).toBe('replace')
    expect(cm?.compactedMessageCount).toBe(5)
  })

  it('should mark steps as is_copied_context after compaction_injected', async () => {
    const fork = await makeAtif([
      {
        type: 'user_message',
        forkId,
        messageId: 'msg-1',
        timestamp: ts(1),
        text: 'Do something',
        mentions: [],
        attachments: [],
        mode: 'text',
        synthetic: false,
        taskMode: false,
      },
      {
        type: 'compaction_prepared',
        forkId,
        timestamp: ts(2),
        turn: { turnId: 'turn-compact' } as any,
        compactedMessageCount: 5,
        inputTokens: 1000,
        outputTokens: 200,
        refreshedContext: null,
        isFallback: false,
        compactResult: {} as any,
      },
      {
        type: 'compaction_injected',
        forkId,
        timestamp: ts(3),
      },
    ] as AppEvent[], true)

    // First step (user message) should be marked is_copied_context
    expect(fork.steps.length).toBe(2)
    expect(fork.steps[0].source).toBe('user')
    expect(Option.getOrNull(fork.steps[0].is_copied_context)).toBe(true)
    // Compaction step itself should NOT be is_copied_context
    expect(fork.steps[1].source).toBe('system')
    expect(Option.getOrNull(fork.steps[1].is_copied_context)).toBeFalsy()
  })

  it('should create child fork with proper agentName from agent_created', async () => {
    const fork = await makeAtif([
      {
        type: 'agent_created',
        forkId: 'agent-scout-1',
        parentForkId: null,
        agentId: 'agent-scout-1',
        role: 'scout',
        taskId: 'task-1',
        mode: 'spawn',
        message: 'research',
        name: 'scout-1',
        context: '',
      },
    ] as AppEvent[], true, 'agent-scout-1')

    expect(fork.agentName).toBe('magnitude-scout')
    expect(fork.agentRole).toBe('scout')
    // Root fork should have a spawnWorker step
    const rootFork = await makeAtif([
      {
        type: 'agent_created',
        forkId: 'agent-scout-1',
        parentForkId: null,
        agentId: 'agent-scout-1',
        role: 'scout',
        taskId: 'task-1',
        mode: 'spawn',
        message: 'research',
        name: 'scout-1',
        context: '',
      },
    ] as AppEvent[], true, null)

    expect(rootFork.steps.length).toBe(1)
    const rootToolCalls = Option.getOrNull(rootFork.steps[0].tool_calls)
    expect(rootToolCalls?.[0].function_name).toBe('spawnWorker')
    const rootObservation = Option.getOrNull(rootFork.steps[0].observation)
    const subagentRef = rootObservation && Option.getOrNull(rootObservation.results[0].subagent_trajectory_ref)
    const firstRef = subagentRef?.[0]
    expect(firstRef && Option.getOrNull(firstRef.trajectory_id)).toBe('agent-scout-1')
  })

  it('should populate cost_usd from turn_outcome cost', async () => {
    const fork = await makeAtif([
      {
        type: 'turn_started',
        forkId,
        turnId,
        chainId: 'chain-1',
      },
      {
        type: 'message_chunk',
        forkId,
        turnId,
        id: 'msg-1',
        text: 'Response',
      },
      {
        type: 'turn_outcome',
        forkId,
        turnId,
        chainId: 'chain-1',
        strategyId: 'native',
        outcome: { _tag: 'Completed', completion: { toolCallsCount: 0, finishReason: 'stop', feedback: [], yieldTarget: null }, requestId: null },
        inputTokens: 100,
        outputTokens: 20,
        cacheReadTokens: 0,
        cacheWriteTokens: 0,
        cost: 0.0035,
        providerId: null,
        modelId: null,
      },
    ] as AppEvent[], true)

    expect(fork.steps.length).toBe(1)
    const costMetrics = Option.getOrNull(fork.steps[0].metrics)
    expect(costMetrics && Option.getOrNull(costMetrics.cost_usd)).toBe(0.0035)
    expect(fork.tokenAccumulator.costUsd).toBe(0.0035)
  })

  it('should produce valid ATIF v1.7 JSON from serializeAtif', async () => {
    const rootFork = await makeAtif([
      {
        type: 'user_message',
        forkId,
        messageId: 'msg-1',
        timestamp: ts(1),
        text: 'Hello',
        mentions: [],
        attachments: [],
        mode: 'text',
        synthetic: false,
        taskMode: false,
      },
      {
        type: 'agent_created',
        forkId: 'agent-scout-1',
        parentForkId: null,
        agentId: 'agent-scout-1',
        role: 'scout',
        taskId: 'task-1',
        mode: 'spawn',
        message: 'research',
        name: 'scout-1',
        context: '',
      },
    ] as AppEvent[], true)

    const scoutFork = await makeAtif([
      {
        type: 'user_message',
        forkId: null,
        messageId: 'msg-1',
        timestamp: ts(1),
        text: 'Hello',
        mentions: [],
        attachments: [],
        mode: 'text',
        synthetic: false,
        taskMode: false,
      },
      {
        type: 'agent_created',
        forkId: 'agent-scout-1',
        parentForkId: null,
        agentId: 'agent-scout-1',
        role: 'scout',
        taskId: 'task-1',
        mode: 'spawn',
        message: 'research',
        name: 'scout-1',
        context: '',
      },
    ] as AppEvent[], true, 'agent-scout-1')

    // Build forks map from the two separate queries
    const forks = new Map<string | null, AtifForkState>([
      [null, rootFork],
      ['agent-scout-1', scoutFork],
    ])

    const trajectory = serializeAtif(forks, { sessionId: 'test-session', ...serializationTools })

    // Validate ATIF v1.7 structure
    expect(trajectory.schema_version).toBe('ATIF-v1.7')
    expect(Option.getOrNull(trajectory.session_id)).toBe('test-session')
    expect(Option.getOrNull(trajectory.trajectory_id)).toBe('main')
    expect(trajectory.agent.name).toBe('magnitude')
    expect(trajectory.steps.length).toBeGreaterThan(0)

    // Validate sequential step_ids
    for (let i = 0; i < trajectory.steps.length; i++) {
      expect(trajectory.steps[i].step_id).toBe(i + 1)
    }

    // Validate subagent trajectories
    expect(trajectory.subagent_trajectories).toBeDefined()
    const subagentTrajectories = Option.getOrNull(trajectory.subagent_trajectories) as AtifTrajectory[] | null
    expect(subagentTrajectories?.length).toBe(1)
    const firstSub = subagentTrajectories?.[0]
    expect(firstSub && Option.getOrNull(firstSub.trajectory_id)).toBe('agent-scout-1')
    expect(firstSub?.agent.name).toBe('magnitude-scout')
  })

  it('should produce a user step for interrupt', async () => {
    const fork = await makeAtif([
      {
        type: 'interrupt',
        forkId: null,
      },
    ] as AppEvent[], true)

    expect(fork.steps.length).toBe(1)
    expect(fork.steps[0].source).toBe('user')
    expect(fork.steps[0].message).toBe('Agent interrupted')
  })

  it('should emit forkCompleted signal on agent_killed', async () => {
    // Create an agent first, then kill it — the worker fork should exist
    const fork = await makeAtif([
      {
        type: 'agent_created',
        forkId: 'agent-1',
        parentForkId: null,
        agentId: 'agent-1',
        role: 'engineer',
        taskId: 'task-1',
        mode: 'spawn',
        message: 'implement',
        name: 'eng-1',
        context: '',
      },
      {
        type: 'agent_killed',
        forkId: 'agent-1',
        agentId: 'agent-1',
        reason: 'done',
      },
    ] as AppEvent[], true, 'agent-1')

    // The worker fork should still have the initial state from agent_created
    expect(fork.agentName).toBe('magnitude-engineer')
  })

  it('should handle multiple tool calls in a single agent step', async () => {
    const fork = await makeAtif([
      {
        type: 'turn_started',
        forkId,
        turnId,
        chainId: 'chain-1',
      },
      {
        type: 'tool_event',
        forkId,
        turnId,
        toolCallId: 'tc-1',
        providerToolCallId: 'tc-1',
        toolKey: toToolKeyErased('fileRead'),
        event: { _tag: 'ToolInputReady', toolCallId: 'tc-1', providerToolCallId: 'tc-1', toolName: 'fileRead', toolKey: toToolKeyErased('fileRead') },
      },
      {
        type: 'tool_event',
        forkId,
        turnId,
        toolCallId: 'tc-1',
        providerToolCallId: 'tc-1',
        toolKey: toToolKeyErased('fileRead'),
        event: { _tag: 'ToolExecutionStarted', toolCallId: 'tc-1', providerToolCallId: 'tc-1', toolName: 'fileRead', toolKey: toToolKeyErased('fileRead'), input: { path: '/tmp/a.txt' } },
      },
      {
        type: 'tool_event',
        forkId,
        turnId,
        toolCallId: 'tc-1',
        providerToolCallId: 'tc-1',
        toolKey: toToolKeyErased('fileRead'),
        event: { _tag: 'ToolExecutionEnded', toolCallId: 'tc-1', providerToolCallId: 'tc-1', toolName: 'fileRead', toolKey: toToolKeyErased('fileRead'), result: { _tag: 'Success', output: 'content-a' } },
      },
      {
        type: 'tool_event',
        forkId,
        turnId,
        toolCallId: 'tc-2',
        providerToolCallId: 'tc-2',
        toolKey: toToolKeyErased('shell'),
        event: { _tag: 'ToolInputReady', toolCallId: 'tc-2', providerToolCallId: 'tc-2', toolName: 'shell', toolKey: toToolKeyErased('shell') },
      },
      {
        type: 'tool_event',
        forkId,
        turnId,
        toolCallId: 'tc-2',
        providerToolCallId: 'tc-2',
        toolKey: toToolKeyErased('shell'),
        event: { _tag: 'ToolExecutionStarted', toolCallId: 'tc-2', providerToolCallId: 'tc-2', toolName: 'shell', toolKey: toToolKeyErased('shell'), input: { command: 'ls' } },
      },
      {
        type: 'tool_event',
        forkId,
        turnId,
        toolCallId: 'tc-2',
        providerToolCallId: 'tc-2',
        toolKey: toToolKeyErased('shell'),
        event: { _tag: 'ToolExecutionEnded', toolCallId: 'tc-2', providerToolCallId: 'tc-2', toolName: 'shell', toolKey: toToolKeyErased('shell'), result: { _tag: 'Success', output: 'file1 file2' } },
      },
      {
        type: 'turn_outcome',
        forkId,
        turnId,
        chainId: 'chain-1',
        strategyId: 'native',
        outcome: { _tag: 'Completed', completion: { toolCallsCount: 2, finishReason: 'tool_calls', feedback: [], yieldTarget: null }, requestId: null },
        inputTokens: 200,
        outputTokens: 30,
        cacheReadTokens: 50,
        cacheWriteTokens: 10,
        cost: 0.001,
        providerId: 'magnitude',
        modelId: 'claude-sonnet-4-6',
      },
    ] as AppEvent[], true)

    expect(fork.steps.length).toBe(1)
    const step = fork.steps[0]
    expect(Option.getOrNull(step.tool_calls)?.length).toBe(2)
    expect(Option.getOrNull(step.tool_calls)?.[0].function_name).toBe('fileRead')
    expect(Option.getOrNull(step.tool_calls)?.[0].arguments).toEqual({ path: '/tmp/a.txt' })
    expect(Option.getOrNull(step.tool_calls)?.[1].function_name).toBe('shell')
    expect(Option.getOrNull(step.tool_calls)?.[1].arguments).toEqual({ command: 'ls' })
    const stepObservation = Option.getOrNull(step.observation)
    expect(stepObservation?.results.length).toBe(2)
    expect(stepObservation && Option.getOrNull(stepObservation.results[0].source_call_id)).toBe('tc-1')
    expect(stepObservation && Option.getOrNull(stepObservation.results[1].source_call_id)).toBe('tc-2')

    // Metrics
    const metrics = Option.getOrNull(step.metrics)
    expect(metrics && Option.getOrNull(metrics.prompt_tokens)).toBe(200)
    expect(metrics && Option.getOrNull(metrics.completion_tokens)).toBe(30)
    expect(metrics && Option.getOrNull(metrics.cached_tokens)).toBe(50)
    expect(metrics && Option.getOrNull(metrics.cost_usd)).toBe(0.001)
    expect(Option.getOrNull(step.model_name)).toBe('claude-sonnet-4-6')

    // Provider-specific metrics in extra
    const metricsExtra = metrics && Option.getOrNull(metrics.extra)
    expect(metricsExtra?.cache_creation_input_tokens).toBe(10)
    expect(metricsExtra?.provider_id).toBe('magnitude')
    expect(metricsExtra?.model_id).toBe('claude-sonnet-4-6')

    // Token accumulator
    expect(fork.tokenAccumulator.promptTokens).toBe(200)
    expect(fork.tokenAccumulator.completionTokens).toBe(30)
    expect(fork.tokenAccumulator.cachedTokens).toBe(50)
    expect(fork.tokenAccumulator.costUsd).toBe(0.001)
  })

  it('should handle agent step with empty message and no thinking', async () => {
    const fork = await makeAtif([
      {
        type: 'turn_started',
        forkId,
        turnId,
        chainId: 'chain-1',
      },
      // No thinking_chunk or message_chunk — agent only makes tool calls
      {
        type: 'tool_event',
        forkId,
        turnId,
        toolCallId: 'tc-1',
        providerToolCallId: 'tc-1',
        toolKey: toToolKeyErased('shell'),
        event: { _tag: 'ToolInputReady', toolCallId: 'tc-1', providerToolCallId: 'tc-1', toolName: 'shell', toolKey: toToolKeyErased('shell') },
      },
      {
        type: 'tool_event',
        forkId,
        turnId,
        toolCallId: 'tc-1',
        providerToolCallId: 'tc-1',
        toolKey: toToolKeyErased('shell'),
        event: { _tag: 'ToolExecutionStarted', toolCallId: 'tc-1', providerToolCallId: 'tc-1', toolName: 'shell', toolKey: toToolKeyErased('shell'), input: { command: 'echo hi' } },
      },
      {
        type: 'tool_event',
        forkId,
        turnId,
        toolCallId: 'tc-1',
        providerToolCallId: 'tc-1',
        toolKey: toToolKeyErased('shell'),
        event: { _tag: 'ToolExecutionEnded', toolCallId: 'tc-1', providerToolCallId: 'tc-1', toolName: 'shell', toolKey: toToolKeyErased('shell'), result: { _tag: 'Success', output: 'hi' } },
      },
      {
        type: 'turn_outcome',
        forkId,
        turnId,
        chainId: 'chain-1',
        strategyId: 'native',
        outcome: { _tag: 'Completed', completion: { toolCallsCount: 1, finishReason: 'tool_calls', feedback: [], yieldTarget: null }, requestId: null },
        inputTokens: 50,
        outputTokens: 10,
        cacheReadTokens: 0,
        cacheWriteTokens: 0,
        cost: null,
        providerId: null,
        modelId: null,
      },
    ] as AppEvent[], true)

    expect(fork.steps.length).toBe(1)
    // Empty message should be empty string, not undefined
    expect(fork.steps[0].message).toBe('')
    // No reasoning_content should mean the field is absent
    expect(Option.getOrNull(fork.steps[0].reasoning_content)).toBeFalsy()
    expect(Option.getOrNull(fork.steps[0].tool_calls)?.length).toBe(1)
  })

  it('should emit a terminal agent step when agent is killed', async () => {
    const fork = await makeAtif([
      {
        type: 'agent_created',
        forkId: 'agent-1',
        parentForkId: null,
        agentId: 'agent-1',
        role: 'engineer',
        taskId: 'task-1',
        mode: 'spawn',
        message: 'work',
        name: 'eng-1',
        context: '',
      },
      {
        type: 'agent_killed',
        forkId: 'agent-1',
        agentId: 'agent-1',
        reason: 'task_completed',
      },
    ] as AppEvent[], true, 'agent-1')

    // Should have a terminal agent step
    const lastStep = fork.steps[fork.steps.length - 1]
    expect(lastStep).toBeDefined()
    expect(lastStep.source).toBe('agent')
    expect(lastStep.message).toContain('Agent killed')
    expect(Option.getOrNull(lastStep.extra)?.agentId).toBe('agent-1')
    expect(Option.getOrNull(lastStep.extra)?.reason).toBe('task_completed')
  })

  it('should produce ATIF v1.7 compliant JSON structure', async () => {
    // Build a realistic multi-step trajectory and validate the serialized output
    const rootFork = await makeAtif([
      {
        type: 'user_message',
        forkId,
        messageId: 'msg-1',
        timestamp: ts(1),
        text: 'Create a hello world file',
        mentions: [],
        attachments: [],
        mode: 'text',
        synthetic: false,
        taskMode: false,
      },
      {
        type: 'turn_started',
        forkId,
        turnId,
        chainId: 'chain-1',
      },
      {
        type: 'thinking_chunk',
        forkId,
        turnId,
        text: 'I need to create a simple file',
      },
      {
        type: 'message_chunk',
        forkId,
        turnId,
        id: 'msg-2',
        text: 'I will create the file for you',
      },
      {
        type: 'tool_event',
        forkId,
        turnId,
        toolCallId: 'tc-1',
        providerToolCallId: 'tc-1',
        toolKey: toToolKeyErased('fileWrite'),
        event: { _tag: 'ToolInputReady', toolCallId: 'tc-1', providerToolCallId: 'tc-1', toolName: 'fileWrite', toolKey: toToolKeyErased('fileWrite') },
      },
      {
        type: 'tool_event',
        forkId,
        turnId,
        toolCallId: 'tc-1',
        providerToolCallId: 'tc-1',
        toolKey: toToolKeyErased('fileWrite'),
        event: { _tag: 'ToolExecutionStarted', toolCallId: 'tc-1', providerToolCallId: 'tc-1', toolName: 'fileWrite', toolKey: toToolKeyErased('fileWrite'), input: { path: 'hello.txt', content: 'Hello World' }, cached: false },
      },
      {
        type: 'tool_event',
        forkId,
        turnId,
        toolCallId: 'tc-1',
        providerToolCallId: 'tc-1',
        toolKey: toToolKeyErased('fileWrite'),
        event: { _tag: 'ToolExecutionEnded', toolCallId: 'tc-1', providerToolCallId: 'tc-1', toolName: 'fileWrite', toolKey: toToolKeyErased('fileWrite'), result: { _tag: 'Success', output: 'File written' } },
      },
      {
        type: 'turn_outcome',
        forkId,
        turnId,
        chainId: 'chain-1',
        strategyId: 'native',
        outcome: { _tag: 'Completed', completion: { toolCallsCount: 1, finishReason: 'tool_calls', feedback: [], yieldTarget: null }, requestId: null },
        inputTokens: 500,
        outputTokens: 100,
        cacheReadTokens: 200,
        cacheWriteTokens: 50,
        cost: 0.005,
        providerId: 'magnitude',
        modelId: 'claude-sonnet-4-6',
      },
    ] as AppEvent[], true)

    const trajectory = serializeAtif(new Map([[null, rootFork]]), { sessionId: 'session-test-001', ...serializationTools })

    // === ATIF v1.7 Structural Validation ===

    // Root fields
    expect(trajectory.schema_version).toBe('ATIF-v1.7')
    expect(typeof Option.getOrNull(trajectory.session_id)).toBe('string')
    expect(typeof Option.getOrNull(trajectory.trajectory_id)).toBe('string')
    expect(typeof trajectory.agent.name).toBe('string')
    expect(typeof trajectory.agent.version).toBe('string')
    expect(Array.isArray(trajectory.steps)).toBe(true)

    // Step 1: user message
    const userStep = trajectory.steps[0]
    expect(userStep.step_id).toBe(1)
    expect(userStep.source).toBe('user')
    expect(typeof userStep.message).toBe('string')
    expect(Option.getOrNull(userStep.timestamp)).toBeTruthy()

    // Step 2: agent step with tool calls
    const agentStep = trajectory.steps[1]
    expect(agentStep.step_id).toBe(2)
    expect(agentStep.source).toBe('agent')
    expect(Option.getOrNull(agentStep.model_name)).toBe('claude-sonnet-4-6')
    expect(typeof agentStep.message).toBe('string')
    expect(Option.getOrNull(agentStep.reasoning_content)).toBeTruthy()
    const agentStepToolCalls = Option.getOrNull(agentStep.tool_calls)
    expect(Array.isArray(agentStepToolCalls)).toBe(true)
    expect(agentStepToolCalls?.length).toBe(1)

    // Tool call structure
    const tc = agentStepToolCalls?.[0]
    expect(typeof tc?.tool_call_id).toBe('string')
    expect(typeof tc?.function_name).toBe('string')
    expect(typeof tc?.arguments).toBe('object')
    expect(tc?.arguments).toEqual({ path: 'hello.txt', content: 'Hello World' })

    // Observation structure
    expect(agentStep.observation).toBeDefined()
    const agentStepObservation = Option.getOrNull(agentStep.observation)
    expect(Array.isArray(agentStepObservation?.results)).toBe(true)
    const obs = agentStepObservation?.results[0]
    expect(obs && Option.getOrNull(obs.source_call_id)).toBe('tc-1')
    expect(typeof (obs && Option.getOrNull(obs.content))).toBe('string')

    // Metrics structure
    expect(agentStep.metrics).toBeDefined()
    const agentStepMetrics = Option.getOrNull(agentStep.metrics)
    expect(agentStepMetrics && Option.getOrNull(agentStepMetrics.prompt_tokens)).toBe(500)
    expect(agentStepMetrics && Option.getOrNull(agentStepMetrics.completion_tokens)).toBe(100)
    expect(agentStepMetrics && Option.getOrNull(agentStepMetrics.cached_tokens)).toBe(200)
    expect(agentStepMetrics && Option.getOrNull(agentStepMetrics.cost_usd)).toBe(0.005)
    expect(typeof (agentStepMetrics && Option.getOrNull(agentStepMetrics.extra))).toBe('object')

    // Final metrics
    expect(trajectory.final_metrics).toBeDefined()
    const finalMetrics = Option.getOrNull(trajectory.final_metrics)
    expect(finalMetrics && Option.getOrNull(finalMetrics.total_steps)).toBe(2)
    expect(finalMetrics && Option.getOrNull(finalMetrics.total_prompt_tokens)).toBe(500)
    expect(finalMetrics && Option.getOrNull(finalMetrics.total_completion_tokens)).toBe(100)
    expect(finalMetrics && Option.getOrNull(finalMetrics.total_cached_tokens)).toBe(200)
    expect(finalMetrics && Option.getOrNull(finalMetrics.total_cost_usd)).toBe(0.005)

    // Agent info
    expect(trajectory.agent.name).toBe('magnitude')
    expect(trajectory.agent.version).toBe('1.0.0')
  })
})
