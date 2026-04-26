import { describe, expect, it, vi } from 'vitest'
import { Effect } from 'effect'
import { BamlDriver } from '../drivers/baml-driver'
import { Model } from '../model/model'
import { ModelConnection } from '../model/model-connection'
import type { DriverRequest } from '../drivers/types'
import * as dispatch from '../drivers/baml-dispatch'

describe('openai-responses complete path', () => {
  it('uses stream-and-collect for openai-responses-style requests', async () => {
    const model = new Model({
      id: 'gpt-5.4',
      providerId: 'openai',
      providerName: 'OpenAI',
      name: 'GPT 5.4',
      contextWindow: 128_000,
      maxOutputTokens: 8_192,
      costs: null,
      supportsVision: true,
    })

    const req: DriverRequest = {
      slot: 'lead',
      functionName: 'CodingAgentCompact',
      args: ['system prompt', [{ role: 'user', content: ['hello'] }], false, false],
      connection: ModelConnection.Baml({
        auth: {
          type: 'oauth',
          oauthMethod: 'oauth-browser',
          accessToken: 'token',
          refreshToken: 'refresh',
          expiresAt: Date.now() + 60_000,
          accountId: 'acct',
        },
      }),
      model,
      inference: {},
      providerOptions: {
        instructions: 'system prompt',
        store: false,
      },
    }

    const bamlStreamSpy = vi.spyOn(dispatch, 'bamlStream').mockReturnValue({
      async getFinalResponse() {
        return 'compacted result'
      },
      async *[Symbol.asyncIterator]() {
        yield 'compacted result'
      },
    })

    await expect(Effect.runPromise(BamlDriver.complete(req))).resolves.toMatchObject({
      result: 'compacted result',
    })

    expect(bamlStreamSpy).toHaveBeenCalled()
  })
})