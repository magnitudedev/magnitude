import { Effect, Layer } from 'effect'

import { AuthStorage } from './contracts'
import { getAuth, loadAuth, removeAuth, setAuth } from './storage'
import { GlobalStorage } from '../services'
import type { AuthInfo } from '../types'

export const AuthStorageLive = Layer.effect(
  AuthStorage,
  Effect.gen(function* () {
    const globalStorage = yield* GlobalStorage

    return AuthStorage.of({
      loadAll: () => Effect.promise(() => loadAuth(globalStorage.paths)),
      get: (providerId: string) =>
        Effect.promise(() => getAuth(globalStorage.paths, providerId)),
      set: (providerId: string, info: AuthInfo) =>
        Effect.promise(() => setAuth(globalStorage.paths, providerId, info)),
      remove: (providerId: string) =>
        Effect.promise(() => removeAuth(globalStorage.paths, providerId)),
    })
  })
)