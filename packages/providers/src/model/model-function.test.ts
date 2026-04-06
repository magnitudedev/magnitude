import { describe, expect, it, vi } from 'vitest'
import type { BoundModel } from './bound-model'
import { Model } from './model'
import { ModelConnection } from './model-connection'
import { CodingAgentChat } from './model-function'

function makeBoundModel(providerId: string, modelId: string, authType: 'oauth' | 'api' | null): BoundModel {
  const model = new Model({
    id: modelId,
    providerId,
    name: modelId,
    contextWindow: 200000,
    maxOutputTokens: null,
    costs: null,
  })

  const connection = ModelConnection.Baml({
    auth: authType === null
      ? null
      : authType === 'oauth'
        ? { type: 'oauth', accessToken: 'token', refreshToken: 'refresh', expiresAt: Date.now() + 100000 }
        : { type: 'api', key: 'key' },
  })

  return {
    model,
    connection,
    invoke: vi.fn() as any,
    complete: vi.fn() as any,
    stream: vi.fn() as any,
  }
}

describe('CodingAgentChat provider instruction mapping', () => {
  it('maps system prompt to providerOptions.instructions for openai oauth codex path', () => {
    const bound = makeBoundModel('openai', 'gpt-5.4', 'oauth')
    const streamSpy = vi.spyOn(bound, 'stream').mockReturnValue({} as any)

    CodingAgentChat.execute(bound, {
      systemPrompt: 'SYSTEM PROMPT',
      messages: [{ role: 'user', content: ['hey'] }],
      ackTurn: 'ack',
      options: { stopSequences: ['</done>'] },
    })

    expect(streamSpy).toHaveBeenCalledTimes(1)
    const [, args, options] = streamSpy.mock.calls[0]
    expect(args[4]).toBe(false)
    expect(options?.providerOptions).toEqual({ instructions: 'SYSTEM PROMPT' })
    expect(options?.stopSequences).toEqual(['</done>'])
  })

  it('keeps system prompt message rendering for non-codex paths', () => {
    const bound = makeBoundModel('anthropic', 'claude-sonnet', 'api')
    const streamSpy = vi.spyOn(bound, 'stream').mockReturnValue({} as any)

    CodingAgentChat.execute(bound, {
      systemPrompt: 'SYSTEM PROMPT',
      messages: [{ role: 'user', content: ['hey'] }],
      ackTurn: 'ack',
    })

    const [, args, options] = streamSpy.mock.calls[0]
    expect(args[4]).toBe(true)
    expect(options?.providerOptions).toBeUndefined()
  })
})
