import { describe, expect, test, mock } from 'bun:test'
import { Effect, Layer, Exit } from 'effect'
import { AppConfig } from '@magnitudedev/storage'
import { makeModelResolver } from './model-resolver-live'
import { ModelResolver } from './model-resolver'
import { ProviderAuth, ProviderCatalog, ProviderState } from '../runtime/contracts'
import { ProviderDisconnected } from '../errors/model-error'
import type { OAuthAuth } from '../types'

const expired: OAuthAuth = {
  type: 'oauth',
  oauthMethod: 'oauth-browser',
  accessToken: 'expired',
  refreshToken: 'refresh-old',
  expiresAt: Date.now() - 60_000,
}

describe('makeModelResolver auth remap', () => {
  test('maps preflight AuthFailed to ProviderDisconnected with reconnect guidance', async () => {
    const oldFetch = globalThis.fetch
    globalThis.fetch = mock(async () => new Response('boom', { status: 500 })) as any

    const providerCatalog = {
      listProviders: () => Effect.succeed([]),
      getProvider: () => Effect.succeed(null),
      getProviderName: () => Effect.succeed('OpenAI'),
      listModels: () => Effect.succeed([]),
      getModel: () => Effect.succeed(null),
      refresh: () => Effect.void,
    }

    const providerState = {
      peek: () =>
        Effect.succeed({
          model: {
            id: 'gpt-5',
            providerId: 'openai',
            providerName: 'OpenAI',
            name: 'GPT-5',
            contextWindow: 200_000,
            maxOutputTokens: null,
            costs: null,
          },
          auth: expired,
        }),
      getSlot: () => Effect.succeed({ providerId: 'openai', modelId: 'gpt-5', auth: expired, registry: undefined }),
      setSelection: () => Effect.succeed(true),
      clear: () => Effect.void,
      contextWindow: () => Effect.succeed(0),
      contextLimits: () => Effect.succeed({ hardCap: 0, softCap: 0 }),
      accumulateUsage: () => Effect.void,
      getUsage: () => Effect.die('unused'),
      resetUsage: () => Effect.void,
    }

    const providerAuth = {
      loadAuth: () => Effect.succeed({}),
      getAuth: () => Effect.succeed(expired),
      setAuth: () => Effect.void,
      removeAuth: () => Effect.void,
      refresh: () => Effect.die('unused'),
      detectProviders: () => Effect.succeed([]),
      detectDefaultProvider: () => Effect.succeed(null),
      detectProviderAuthMethods: () => Effect.succeed(null),
      connectedProviderIds: () => Effect.succeed(new Set(['openai'])),
    }

    const appConfig = {
      load: () => Effect.succeed({ roles: {}, presets: [], contextLimits: {}, setupComplete: true, telemetry: true }),
      save: () => Effect.void,
      update: () => Effect.succeed({ roles: {}, presets: [], contextLimits: {}, setupComplete: true, telemetry: true }),
      getContextLimitPolicy: () => Effect.succeed({ hardLimitRatio: 0.95, softLimitRatio: 0.8 }),
      setContextLimitPolicy: () => Effect.void,
      getSetupComplete: () => Effect.succeed(true),
      setSetupComplete: () => Effect.void,
      getTelemetryEnabled: () => Effect.succeed(true),
      setTelemetryEnabled: () => Effect.void,
      getRoleConfig: () => Effect.succeed(null),
      getRoleConfigs: () => Effect.succeed({}),
      getModelSelection: () => Effect.succeed(null),
      setModelSelection: () => Effect.void,
      getPresets: () => Effect.succeed([]),
      savePreset: () => Effect.void,
      deletePreset: () => Effect.void,
      getProviderOptions: () => Effect.succeed(undefined),
      setProviderOptions: () => Effect.void,
    }

    const layer = makeModelResolver<string>().pipe(
      Layer.provide(Layer.succeed(ProviderCatalog, providerCatalog as any)),
      Layer.provide(Layer.succeed(ProviderState, providerState as any)),
      Layer.provide(Layer.succeed(ProviderAuth, providerAuth as any)),
      Layer.provide(Layer.succeed(AppConfig, appConfig as any)),
    )

    const exit = await Effect.runPromiseExit(
      Effect.gen(function* () {
        const resolver = yield* ModelResolver
        return yield* resolver.resolve('primary')
      }).pipe(Effect.provide(layer)),
    )

    expect(Exit.isFailure(exit)).toBe(true)
    if (Exit.isFailure(exit)) {
      const causeText = String(exit.cause)
      expect(causeText).toContain('ProviderDisconnected')
      expect(causeText).toContain('OpenAI session expired or became invalid. Please reconnect in /settings.')
    }

    globalThis.fetch = oldFetch
  })
})
