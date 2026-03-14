import { Context, Effect } from 'effect'

import type { AuthInfo } from '../types'

export interface AuthStorageShape {
  readonly loadAll: () => Effect.Effect<Record<string, AuthInfo>>
  readonly get: (providerId: string) => Effect.Effect<AuthInfo | undefined>
  readonly set: (providerId: string, info: AuthInfo) => Effect.Effect<void>
  readonly remove: (providerId: string) => Effect.Effect<void>
}

export class AuthStorage extends Context.Tag('AuthStorage')<
  AuthStorage,
  AuthStorageShape
>() {}