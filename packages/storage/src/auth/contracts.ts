import type { PlatformError } from '@effect/platform/Error'
import { Context, Effect, Option } from 'effect'

import type { JsonError } from '../io/storage'
import type { AuthInfo, HuggingFaceCredential } from '../types'

export interface AuthStorageShape {
  readonly loadAll: () => Effect.Effect<Record<string, AuthInfo>, PlatformError | JsonError>
  readonly get: (providerId: string) => Effect.Effect<AuthInfo | undefined, PlatformError | JsonError>
  readonly set: (providerId: string, info: AuthInfo) => Effect.Effect<void, PlatformError | JsonError>
  readonly remove: (providerId: string) => Effect.Effect<void, PlatformError | JsonError>
  readonly getHuggingFace: Effect.Effect<Option.Option<HuggingFaceCredential>, PlatformError | JsonError>
  readonly setHuggingFace: (credential: HuggingFaceCredential) => Effect.Effect<void, PlatformError | JsonError>
  readonly removeHuggingFace: Effect.Effect<void, PlatformError | JsonError>
}

export class AuthStorage extends Context.Tag('AuthStorage')<
  AuthStorage,
  AuthStorageShape
>() {}
