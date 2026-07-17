import { describe, expect, it } from 'vitest'
import { Effect, Layer, Stream } from 'effect'
import { FetchHttpClient } from '@effect/platform'
import { EventEngine } from '@magnitudedev/event-core'
import {
  createStreamingFieldParser,
  ModelStreamTerminal,
  type BoundModel,
  type ModelSpec,
  type ModelStreamResult,
  type Prompt,
  type ProviderToolCallId,
  type ResponseStreamEvent,
  type ToolCallId,
  type ToolDefinition,
} from '@magnitudedev/ai'
import type { BaseCallOptions, ProviderRejection } from '@magnitudedev/sdk'

import type { AppEvent, TurnOutcomeEvent } from '../src/events'
import { AgentModelResolver } from '../src/model/model-resolver'
import { makeAgentBoundModel } from '../src/model/agent-model'
import { ObserverStateLive, ObserverWorker } from '../src/observer'
import { AgentLifecycleProjection } from '../src/projections/agent-lifecycle'
import { DetachedProcessProjection } from '../src/projections/detached-process'
import { HarnessStateProjection } from '../src/projections/harness-state'
import { OutboundMessagesProjection } from '../src/projections/outbound-messages'
import { SessionContextProjection } from '../src/projections/session-context'
import { TaskGraphProjection } from '../src/projections/task-graph'
import { TaskAssignmentProjection } from '../src/projections/task-assignment'
import { TurnProjection } from '../src/projections/turn'
import { UserMessageResolutionProjection } from '../src/projections/user-message-resolution'
import { WorkerActivityProjection } from '../src/projections/worker-activity'
import { WindowProjection } from '../src/window'
import { ToolUniverseSourceLive } from '../src/tools/tool-universe-live'

const profile = {
  contextWindow: 100_000,
  maxOutputTokens: 4096,
}

const testModelSpec: ModelSpec<BaseCallOptions> = {
  modelId: 'observer-scheduler-test',
  endpoint: 'http://test',
  bind: () => { throw new Error('not used') },
  _execute: () => { throw new Error('not used') },
}

const TestAgent = EventEngine.make<AppEvent>()({
  name: 'ObserverSchedulerTestAgent',
  schemaVersion: 'test',
  projections: [
    AgentLifecycleProjection,
    WorkerActivityProjection,
    OutboundMessagesProjection,
    SessionContextProjection,
    UserMessageResolutionProjection,
    TaskGraphProjection,
    TaskAssignmentProjection,
    DetachedProcessProjection,
    HarnessStateProjection,
    TurnProjection,
    WindowProjection,
  ],
  workers: [ObserverWorker],
})

type ObserverOutcomeEvent = Extract<AppEvent, { type: 'observer_outcome' }>

function isRootObserverOutcome(event: AppEvent): event is ObserverOutcomeEvent {
  return event.type === 'observer_outcome' && event.forkId === null
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

async function waitFor(assertion: () => boolean, timeoutMs = 1000) {
  const started = Date.now()
  while (Date.now() - started < timeoutMs) {
    if (assertion()) return
    await sleep(10)
  }
  throw new Error('Timed out waiting for condition')
}

function turnOutcome(turnId: string): TurnOutcomeEvent {
  return {
    type: 'turn_outcome',
    forkId: null,
    turnId,
    chainId: 'chain-1',
    strategyId: 'native',
    outcome: {
      _tag: 'Completed',
      completion: {
        toolCallsCount: 0,
        finishReason: 'stop',
        feedback: [],
        yieldTarget: null,
      },
      requestId: null,
    },
    inputTokens: null,
    outputTokens: null,
    cacheReadTokens: null,
    cacheWriteTokens: null,
    cost: null,
    providerId: 'test',
    modelId: 'role/leader',
  }
}

function turnStarted(turnId: string): AppEvent {
  return {
    type: 'turn_started',
    forkId: null,
    turnId,
    chainId: `chain-${turnId}`,
  }
}

function userMessage(messageId: string): AppEvent {
  return {
    type: 'user_message',
    forkId: null,
    messageId,
    timestamp: Date.now(),
    text: 'continue',
    mentions: [],
    attachments: [],
    mode: 'text',
    synthetic: false,
    taskMode: false,
  }
}

function userMessageReady(messageId: string): AppEvent {
  return {
    type: 'user_message_ready',
    forkId: null,
    messageId,
    mentionResolutions: [],
  }
}

function observerReportEvents(args: {
  readonly toolCallId: ToolCallId
  readonly providerToolCallId: ProviderToolCallId
  readonly tools: readonly ToolDefinition[]
  readonly escalate: boolean
}): readonly ResponseStreamEvent[] {
  const toolName = args.escalate ? 'escalate' : 'pass'
  const toolDef = args.tools.find((t) => t.name === toolName)
  if (!toolDef) throw new Error(`Observer test expected ${toolName} tool`)

  const input = args.escalate
    ? { justification: 'churn' }
    : {}

  const parser = createStreamingFieldParser(toolDef.inputSchema)
  parser.push(JSON.stringify(input))
  parser.end()

  return [
    {
      _tag: 'tool_call_start',
      toolCallId: args.toolCallId,
      providerToolCallId: args.providerToolCallId,
      toolName,
    },
    {
      _tag: 'tool_call_ready',
      toolCallId: args.toolCallId,
      providerToolCallId: args.providerToolCallId,
    },
    {
      _tag: 'stream_end',
      terminal: ModelStreamTerminal.StreamCompleted({
        call: { provider: 'test', model: 'test', method: 'POST', url: 'http://test' },
        response: { status: 200, headers: [], requestId: null },
        finishReason: 'tool_calls',
        progress: { dataPayloadsDecoded: 1, modelEventsEmitted: 1 },
        usage: { _tag: 'UsageNotReported', reason: 'provider_does_not_report_usage' },
      }),
    },
  ] satisfies readonly ResponseStreamEvent[]
}

function makeModelResolver(escalateOnCalls: Set<number>) {
  let observerCalls = 0

  const observerModel: BoundModel<BaseCallOptions> = {
    stream: (_prompt: Prompt, tools: readonly ToolDefinition[], options) =>
      Effect.promise(async (): Promise<ModelStreamResult> => {
        observerCalls += 1
        await sleep(5)
        const toolCallId = options?.generateToolCallId?.() ?? (`observer-call-${observerCalls}` as ToolCallId)
        const providerToolCallId = `provider-${toolCallId}` as ProviderToolCallId
        const escalate = escalateOnCalls.has(observerCalls)
        const toolName = escalate ? 'escalate' : 'pass'

        return {
          events: Stream.fromIterable(observerReportEvents({
            toolCallId,
            providerToolCallId,
            tools,
            escalate,
          })),
          parsers: new Map([[toolCallId, (() => {
            const toolDef = tools.find((t) => t.name === toolName)
            if (!toolDef) throw new Error(`Observer test expected ${toolName} tool`)
            const parser = createStreamingFieldParser(toolDef.inputSchema)
            parser.push(JSON.stringify(escalate ? { justification: 'churn' } : {}))
            parser.end()
            return parser
          })()]]),
          logprobs: [],
          requestId: null,
        }
      }),
  }

  const observerResolved = makeAgentBoundModel({
    rawModel: observerModel,
    modelSource: { slotId: 'secondary' },
    modelId: 'util/observer',
    providerId: 'magnitude',
    profile,
    debug: false,
    agentId: 'observer',
  })

  return Layer.succeed(AgentModelResolver, {
    resolveSlotConfig: () => Effect.succeed(observerResolved),
    resolvePrimary: () => Effect.die('not used'),
    resolveSecondary: () => Effect.succeed(observerResolved),
  })
}

describe('Observer scheduler', () => {
  it('continues scheduling while advisor-required escalation is disabled', async () => {
    // First call escalates, subsequent calls do not
    const requirements = Layer.mergeAll(
      makeModelResolver(new Set([1])),
      ObserverStateLive,
      FetchHttpClient.layer,
      ToolUniverseSourceLive,
    ) as Parameters<typeof TestAgent.createClient>[0]

    const client = await TestAgent.createClient(requirements)
    const events: AppEvent[] = []
    const unsub = client.onEvent((event) => {
      events.push(event)
    })

    try {
      // t1 escalates, but disabled advisor-required routing does not create a
      // pending communication or block later observer work.
      await client.send(turnOutcome('t1'))
      await waitFor(() => {
        const count = events.filter(isRootObserverOutcome).length
        return count >= 1
      }, 1000)
      const firstOutcome = events.filter(isRootObserverOutcome)[0]!
      expect(firstOutcome.observedTurnId).toBe('t1')
      expect(firstOutcome.escalate).toBe(true)
      expect(firstOutcome.justification).toBe('churn')

      await client.send(turnOutcome('blocked-before-claim'))
      await waitFor(() => events.filter(isRootObserverOutcome).length >= 2, 1000)
      expect(events.filter(isRootObserverOutcome)[1]!.observedTurnId).toBe('blocked-before-claim')

      await client.send(userMessage('message-1'))
      await client.send(userMessageReady('message-1'))
      await client.send(turnStarted('t2'))
      await client.send(turnOutcome('t2'))
      await waitFor(() => events.filter(isRootObserverOutcome).length >= 3, 1000)
      const thirdOutcome = events.filter(isRootObserverOutcome)[2]!
      expect(thirdOutcome.observedTurnId).toBe('t2')
    } finally {
      unsub()
      await client.dispose()
    }
  })
})
