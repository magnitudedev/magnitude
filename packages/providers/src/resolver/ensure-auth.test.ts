import { describe, expect, test, mock } from 'bun:test'
import { Effect, Exit } from 'effect'
import { ensureAuth } from './ensure-auth'
import { ProviderAuth, ProviderState } from '../runtime/contracts'
import type { OAuthAuth } from '../types'

const now = Date.now()
const expired: OAuthAuth = {
  type: 'oauth',
  oauthMethod: 'oauth-browser',
  accessToken: 'expired',
  refreshToken: 'refresh-old',
  expiresAt: now - 60_000,
}
const fresh: OAuthAuth = {
  type: 'oauth',
  oauthMethod: 'oauth-browser',
  accessToken: 'fresh',
  refreshToken: 'refresh-fresh',
  expiresAt: now + 60 * 60_000,
}

function makeState(auth: OAuthAuth) {
  let selectedAuth: OAuthAuth | null = auth
  return {
    getSlot: () => Effect.succeed({
      providerId: 'openai',
      modelId: 'gpt-5',
      auth: selectedAuth,
      registry: undefined,
    }),
    setSelection: (_slot: string, _providerId: string, _modelId: string, nextAuth: OAuthAuth | null) =>
      Effect.sync(() => {
        selectedAuth = nextAuth
        return true
      }),
    peek: () => Effect.die('unused'),
    clear: () => Effect.void,
    contextWindow: () => Effect.succeed(0),
    contextLimits: () => Effect.succeed({ hardCap: 0, softCap: 0 }),
    accumulateUsage: () => Effect.void,
    getUsage: () => Effect.die('unused'),
    resetUsage: () => Effect.void,
    getSelectedAuth: () => selectedAuth,
  }
}

describe('ensureAuth', () => {
  test('refreshes stale token and persists new auth', async () => {
    const oldFetch = globalThis.fetch
    globalThis.fetch = mock(async () =>
      new Response(JSON.stringify({
        id_token: 'x.y.z',
        access_token: 'new-token',
        refresh_token: 'new-refresh',
        expires_in: 3600,
      }), { status: 200 }),
    ) as any

    const setAuthCalls: OAuthAuth[] = []
    const auth = {
      getAuth: () => Effect.succeed(expired),
      setAuth: (_providerId: string, auth: OAuthAuth) => Effect.sync(() => { setAuthCalls.push(auth) }),
      removeAuth: () => Effect.void,
      loadAuth: () => Effect.succeed({}),
      refresh: () => Effect.die('unused'),
      detectProviders: () => Effect.succeed([]),
      detectDefaultProvider: () => Effect.succeed(null),
      detectProviderAuthMethods: () => Effect.succeed(null),
      connectedProviderIds: () => Effect.succeed(new Set(['openai'])),
    }
    const state = makeState(expired)

    await Effect.runPromise(
      ensureAuth('primary').pipe(
        Effect.provideService(ProviderAuth, auth as any),
        Effect.provideService(ProviderState, state as any),
      ),
    )

    expect(setAuthCalls.length).toBe(1)
    expect(setAuthCalls[0].accessToken).toBe('new-token')
    expect(state.getSelectedAuth()?.accessToken).toBe('new-token')
    globalThis.fetch = oldFetch
  })

  test('uses fresh disk token if refresh fails inside lock', async () => {
    const oldFetch = globalThis.fetch
    globalThis.fetch = mock(async () => new Response('boom', { status: 500 })) as any

    const diskSequence: Array<OAuthAuth> = [expired, expired, fresh]
    const auth = {
      getAuth: () => Effect.succeed(diskSequence.shift() ?? fresh),
      setAuth: () => Effect.void,
      removeAuth: () => Effect.void,
      loadAuth: () => Effect.succeed({}),
      refresh: () => Effect.die('unused'),
      detectProviders: () => Effect.succeed([]),
      detectDefaultProvider: () => Effect.succeed(null),
      detectProviderAuthMethods: () => Effect.succeed(null),
      connectedProviderIds: () => Effect.succeed(new Set(['openai'])),
    }
    const state = makeState(expired)

    await Effect.runPromise(
      ensureAuth('primary').pipe(
        Effect.provideService(ProviderAuth, auth as any),
        Effect.provideService(ProviderState, state as any),
      ),
    )

    expect(state.getSelectedAuth()?.accessToken).toBe('fresh')
    globalThis.fetch = oldFetch
  })

  test('fails with AuthFailed when refresh fails and disk has no fresh token', async () => {
    const oldFetch = globalThis.fetch
    globalThis.fetch = mock(async () => new Response('boom', { status: 500 })) as any

    const auth = {
      getAuth: () => Effect.succeed(expired),
      setAuth: () => Effect.void,
      removeAuth: () => Effect.void,
      loadAuth: () => Effect.succeed({}),
      refresh: () => Effect.die('unused'),
      detectProviders: () => Effect.succeed([]),
      detectDefaultProvider: () => Effect.succeed(null),
      detectProviderAuthMethods: () => Effect.succeed(null),
      connectedProviderIds: () => Effect.succeed(new Set(['openai'])),
    }
    const state = makeState(expired)

    const exit = await Effect.runPromiseExit(
      ensureAuth('primary').pipe(
        Effect.provideService(ProviderAuth, auth as any),
        Effect.provideService(ProviderState, state as any),
      ),
    )

    expect(Exit.isFailure(exit)).toBe(true)
    if (Exit.isFailure(exit)) {
      const failure = exit.cause as any
      expect(String(failure)).toContain('AuthFailed')
    }

    globalThis.fetch = oldFetch
  })
})
