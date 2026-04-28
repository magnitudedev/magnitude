export interface ApiKeyAuth {
  readonly _tag: "ApiKeyAuth"
  readonly apiKey: string
}

export interface OAuthAuth {
  readonly _tag: "OAuthAuth"
  readonly accessToken: string
}

export interface NoAuth {
  readonly _tag: "NoAuth"
}

export type ResolvedAuth = ApiKeyAuth | OAuthAuth | NoAuth

export interface AuthMethod {
  readonly type:
    | "api-key"
    | "none"
    | "oauth-pkce"
    | "oauth-browser"
    | "oauth-device"
  readonly envKeys?: readonly string[]
  readonly label?: string
}
