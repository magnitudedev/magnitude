import { describe, expect, it } from 'vitest'
import { Effect, Schema, Stream } from 'effect'
import * as HttpClient from '@effect/platform/HttpClient'
import { Prompt, type BaseCallOptions, type BoundModel } from '@magnitudedev/ai'
import { makeAgentBoundModel } from '../src/model/agent-model'

describe('agent model tool-call prevention', () => {
  it('applies the resolved limit to an ordinary auto-choice provider request', async () => {
    let receivedOptions: BaseCallOptions | undefined
    const rawModel: BoundModel<BaseCallOptions> = {
      stream: (_prompt, _tools, options) => {
        receivedOptions = options
        return Effect.succeed({
          events: Stream.empty,
          parsers: new Map(),
          logprobs: [],
          requestId: null,
        })
      },
    }
    const model = makeAgentBoundModel({
      rawModel,
      modelId: 'test-model',
      modelDisplayName: 'Test Model',
      modelSource: { slotId: 'primary' },
      providerId: 'test-provider',
      profile: { contextWindow: 1000, maxOutputTokens: 100 },
      debug: false,
      agentId: 'agent-1',
      maxToolCalls: 2,
    })
    const prompt = Prompt.from({
      system: 'system',
      messages: [{
        _tag: 'UserMessage',
        parts: [{ _tag: 'TextPart', text: 'do work' }],
      }],
    })

    await Effect.runPromise(model.model.stream(prompt, [{
      name: 'read',
      description: 'Read a file',
      inputSchema: Schema.Struct({ path: Schema.String }),
      outputSchema: Schema.String,
    }]).pipe(
      Effect.provideService(HttpClient.HttpClient, {} as HttpClient.HttpClient),
    ))

    expect(receivedOptions?.toolChoice).toEqual({
      type: 'grammar',
      grammar: [
        '<turn> ::= | <call1>',
        '<call1> ::= <tool> | <tool> <call2>',
        '<call2> ::= <tool>',
        '<tool> ::= "read"',
      ].join('\n'),
    })
  })
})
