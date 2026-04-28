import { Context, Effect } from "effect"

export interface StoredApiKey {
  readonly _tag: "api-key"
  readonly key: string
}

export interface StoredOAuth {
  readonly _tag: "oauth"
  readonly accessToken: string
  readonly refreshToken: string
  readonly expiresAt: number
}

export type StoredAuth = StoredApiKey | StoredOAuth

export class AuthStorage extends Context.Tag("@magnitudedev/ai/AuthStorage")<
  AuthStorage,
  {
    readonly getAuth: (providerId: string) => Effect.Effect<StoredAuth | null>
    readonly setAuth: (providerId: string, auth: StoredAuth | null) => Effect.Effect<void>
  }
>() {}
