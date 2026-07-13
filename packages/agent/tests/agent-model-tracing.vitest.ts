import { beforeEach, describe, expect, it, vi } from 'vitest'
import { Effect, Stream } from 'effect'
import * as HttpClient from '@effect/platform/HttpClient'
import { Fork } from '@magnitudedev/event-core'
import {
  Prompt,
  TraceListener,
  type BoundModel,
  type ModelCallTrace,
  type ModelSpec,
  type ModelStreamResult,
  type StreamStartFailure,
  type ToolDefinition,
} from '@magnitudedev/ai'
import type { BaseCallOptions, ProviderRejection } from '@magnitudedev/sdk'
import type { ModelProfile } from '@magnitudedev/ai'

const traceMock = vi.hoisted(() => ({
  sessionId: 'session-1' as string | null,
  traces: [] as unknown[],
}))

vi.mock('@magnitudedev/tracing', () => ({
  getTraceSessionId: () => traceMock.sessionId,
  writeTrace: (trace: unknown) => {
    traceMock.traces.push(trace)
  },
}))

const { makeAgentBoundModel, AgentModelOperationContextTag } = await import('../src/model/agent-model')
const { TurnContextTag } = await import('../src/engine/turn-context')
const { ForkContext } = Fork

const profile: ModelProfile = {
  contextWindow: 100_000,
  maxOutputTokens: 4096,
  capabilities: { vision: true },
}

const testModelSpec: ModelSpec<BaseCallOptions> = {
  modelId: 'test',
  endpoint: 'http://test',
  bind: () => { throw new Error('not used') },
  _execute: () => { throw new Error('not used') },
}

const prompt = Prompt.from({
  messages: [{
    _tag: 'UserMessage',
    parts: [{ _tag: 'TextPart', text: 'hello' }],
  }],
})

const modelTrace: ModelCallTrace = {
  modelId: 'test',
  url: 'http://test',
  startedAt: 123,
  durationMs: 4,
  request: {
    model: 'test',
    messages: [{ role: 'user', content: 'hello' }],
    stream: true,
  },
  response: {
    reasoning: null,
    text: 'world',
    toolCalls: [],
    finishReason: 'stop',
    usage: null,
    logprobs: null,
  },
}

function makeRawModel() {
  const calls: Array<{
    readonly tools: readonly ToolDefinition[]
    readonly options: (BaseCallOptions & { readonly generateToolCallId?: unknown }) | undefined
    readonly hadTraceListener: boolean
  }> = []

  const model: BoundModel<BaseCallOptions> = {
    stream: (_prompt, tools, options) =>
      Effect.gen(function* () {
        const listener = yield* Effect.serviceOption(TraceListener)
        calls.push({ tools, options, hadTraceListener: listener._tag === 'Some' })
        if (listener._tag === 'Some') listener.value.onTrace(modelTrace)
        return {
          events: Stream.empty,
          parsers: new Map(),
          logprobs: [],
          requestId: null,
        } satisfies ModelStreamResult
      }) as Effect.Effect<ModelStreamResult, StreamStartFailure, HttpClient.HttpClient>,
  }

  return { model, calls }
}

function run<A>(effect: Effect.Effect<A, StreamStartFailure, HttpClient.HttpClient>) {
  return Effect.runPromise(
    effect.pipe(Effect.provideService(HttpClient.HttpClient, {} as HttpClient.HttpClient)),
  )
}

describe('makeAgentBoundModel tracing', () => {
  beforeEach(() => {
    traceMock.sessionId = 'session-1'
    traceMock.traces = []
  })

  it('derives turn trace metadata at the model wrapper boundary', async () => {
    const { model, calls } = makeRawModel()
    const wrapped = makeAgentBoundModel({
      rawModel: model,
      modelSource: { slotId: 'primary' },
      modelId: 'role/leader',
      providerId: 'magnitude',
      profile,
      debug: true,
      agentId: 'agent-1',
      roleId: 'leader',
    })

    await run(
      wrapped.model.stream(prompt, [], {
        maxTokens: 1200,
      }).pipe(
        Effect.provideService(TurnContextTag, {
          turnId: 'turn-1',
          chainId: 'chain-1',
          forkId: null,
        }),
      ),
    )

    expect(calls).toHaveLength(1)
    expect(calls[0]?.hadTraceListener).toBe(true)
    expect(traceMock.traces).toHaveLength(1)
    expect(traceMock.traces[0]).toMatchObject({
      sessionId: 'session-1',
      actor: { agentId: 'agent-1', forkId: null, roleId: 'leader' },
      callType: 'chat',
      scope: { kind: 'turn', turnId: 'turn-1', chainId: 'chain-1' },
      modelId: 'test',
    })
  })

  it('derives operation trace metadata for non-turn model work', async () => {
    const { model } = makeRawModel()
    const wrapped = makeAgentBoundModel({
      rawModel: model,
      modelSource: { slotId: 'secondary' },
      modelId: 'util/observer',
      providerId: 'magnitude',
      profile,
      debug: true,
      agentId: 'observer',
      roleId: null,
    })

    await run(
      wrapped.model.stream(prompt, [], { maxTokens: 800 }).pipe(
        Effect.provideService(AgentModelOperationContextTag, {
          operationKind: 'observer',
          operationId: 'observer-turn-1',
          relatedTurnId: 'turn-1',
          chainId: 'chain-1',
          forkId: 'fork-1',
        }),
      ),
    )

    expect(traceMock.traces).toHaveLength(1)
    expect(traceMock.traces[0]).toMatchObject({
      sessionId: 'session-1',
      actor: { agentId: 'observer', forkId: 'fork-1', roleId: null },
      callType: 'observer',
      scope: {
        kind: 'operation',
        operationId: 'observer-turn-1',
        operationKind: 'observer',
        relatedTurnId: 'turn-1',
        chainId: 'chain-1',
        forkId: 'fork-1',
      },
    })
  })

  it('does not install trace listener when tracing is inactive', async () => {
    traceMock.sessionId = null
    const { model, calls } = makeRawModel()
    const wrapped = makeAgentBoundModel({
      rawModel: model,
      modelSource: { slotId: 'secondary' },
      modelId: 'util/title',
      providerId: 'magnitude',
      profile,
      debug: true,
      agentId: 'title-gen',
    })

    await run(wrapped.model.stream(prompt, [], { maxTokens: 100 }))

    expect(calls).toHaveLength(1)
    expect(calls[0]?.hadTraceListener).toBe(false)
    expect(traceMock.traces).toEqual([])
  })
})
