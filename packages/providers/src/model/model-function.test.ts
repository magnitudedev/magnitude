import { describe, expect, it } from 'vitest'
import { Effect, Stream } from 'effect'
import type { BoundModel } from './bound-model'
import { Model, type ModelCosts } from './model'
import { ModelConnection } from './model-connection'
import { TraceEmitter } from '../resolver/tracing'
import {
  CodingAgentChat,
  SimpleChat,
  CodingAgentCompact,
  AutopilotContinuation,
} from './model-function'

function makeModel(providerId: string, id: string, authType: 'oauth' | 'api' | null): BoundModel {
  const model = new Model({
    id,
    providerId,
    name: id,
    contextWindow: 100_000,
    maxOutputTokens: 8192,
    costs: null as unknown as ModelCosts,
  })

  const calls: Array<{ kind: 'stream' | 'complete'; functionName: string; args: readonly unknown[]; options: any }> = []

  const bound: BoundModel = {
    model,
    connection: ModelConnection.Baml({
      auth: authType === null
        ? null
        : authType === 'oauth'
          ? {
            type: 'oauth',
            oauthMethod: providerId === 'github-copilot' ? 'oauth-device' : 'oauth-browser',
            accessToken: 'token',
            refreshToken: 'refresh',
            expiresAt: Date.now() + 60_000,
          }
          : { type: 'api', key: 'key' },
    }),
    invoke(fn: any, input: any) {
      return fn.execute(bound, input)
    },
    stream(functionName: any, args: readonly unknown[], options?: any) {
      calls.push({ kind: 'stream', functionName, args, options })
      return Effect.succeed({
        stream: Stream.empty,
        getUsage: () => ({
          inputTokens: null,
          outputTokens: null,
          cacheReadTokens: null,
          cacheWriteTokens: null,
          inputCost: null,
          outputCost: null,
          totalCost: null,
        }),
        getCollectorData: () => ({ _tag: 'Baml', rawRequestBody: null, rawResponseBody: null }),
      })
    },
    complete(functionName: any, args: readonly unknown[], options?: any) {
      calls.push({ kind: 'complete', functionName, args, options })
      return Effect.succeed({
        result: 'ok',
        usage: {
          inputTokens: null,
          outputTokens: null,
          cacheReadTokens: null,
          cacheWriteTokens: null,
          inputCost: null,
          outputCost: null,
          totalCost: null,
        },
      } as any)
    },
  }

  ;(bound as any).__calls = calls
  return bound
}

function getLastCall(bound: BoundModel) {
  const calls = (bound as any).__calls as Array<{ kind: 'stream' | 'complete'; functionName: string; args: readonly unknown[]; options: any }>
  return calls.at(-1)!
}

const mockTraceEmitter = { emit: () => Effect.void, debug: false }

describe('model-function codex request mapping', () => {
  it('maps OpenAI OAuth Codex systemPrompt into providerOptions.instructions + store false', async () => {
    const bound = makeModel('openai', 'gpt-5.3-codex', 'oauth')
    await Effect.runPromise(
      CodingAgentChat.execute(bound, {
        systemPrompt: 'REAL SYSTEM',
        messages: [{ role: 'user', content: ['hi'] }],
        ackTurns: [
          { role: 'user', content: '--- FEW-SHOT EXAMPLE START ---\n<system>\nRespond using the required turn format. The user reports a bug in the login redirect.\n</system>' },
          { role: 'assistant', content: '<lens name="skills">Bug report → activate the bug skill to load methodology.</lens>\n<skill name="bug" />\n<yield-invoke/>' },
          { role: 'user', content: '--- FEW-SHOT EXAMPLE END ---' },
        ],
      }),
    )
    const call = getLastCall(bound)
    expect(call.args[4]).toBe(false)
    expect(call.options.providerOptions.instructions).toBe('REAL SYSTEM')
    expect(call.options.providerOptions.store).toBe(false)
  })

  it('maps Copilot Codex systemPrompt into providerOptions.instructions without store false', async () => {
    const bound = makeModel('github-copilot', 'gpt-5-codex', 'oauth')
    await Effect.runPromise(
      SimpleChat.execute(bound, {
        systemPrompt: 'COPILOT SYSTEM',
        messages: [{ role: 'user', content: ['hi'] }],
      }).pipe(Effect.provideService(TraceEmitter, mockTraceEmitter)),
    )
    const call = getLastCall(bound)
    expect(call.args[2]).toBe(false)
    expect(call.options.providerOptions.instructions).toBe('COPILOT SYSTEM')
    expect(call.options.providerOptions.store).toBeUndefined()
  })

  it('keeps non-codex path unchanged for instructions and includes system prompt message', async () => {
    const bound = makeModel('anthropic', 'claude-sonnet-4-6', 'api')
    await Effect.runPromise(
      CodingAgentCompact.execute(bound, {
        systemPrompt: 'SYSTEM',
        messages: [{ role: 'user', content: ['hello'] }],
      }).pipe(Effect.provideService(TraceEmitter, mockTraceEmitter)),
    )
    const call = getLastCall(bound)
    expect(call.args[3]).toBe(true)
    expect(call.options).toBeUndefined()
  })

  it('applies includeSystemPromptMessage=false for codex autopilot path', async () => {
    const bound = makeModel('openai', 'gpt-5.3-codex', 'oauth')
    await Effect.runPromise(
      AutopilotContinuation.execute(bound, {
        systemPrompt: 'AUTO',
        messages: [{ role: 'user', content: ['go'] }],
      }).pipe(Effect.provideService(TraceEmitter, mockTraceEmitter)),
    )
    const call = getLastCall(bound)
    expect(call.args[3]).toBe(false)
    expect(call.options.providerOptions.instructions).toBe('AUTO')
  })
})
