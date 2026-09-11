import type { ProviderModel } from "@magnitudedev/ai"

export type MiniMaxRegion = "global_en" | "cn_zh"

export interface MiniMaxEndpointConfig {
  readonly openaiBaseUrl: string
  readonly anthropicBaseUrl: string
}

export type MiniMaxModelInfo = ProviderModel

export type MiniMaxAuthentication =
  | { readonly _tag: "Configured" }
  | { readonly _tag: "NotConfigured" }
