import type { AuthApplicator } from "@magnitudedev/ai"
import type { ModelsDevClient } from "../catalog/models-dev"
import type { OpenAiCompatibleCallOptions, OpenAiCompatibleModelInfo } from "../openai-compatible"

export interface OpenRouterModelInfo extends OpenAiCompatibleModelInfo {
  readonly providerId: "openrouter"
}

export type OpenRouterCallOptions = OpenAiCompatibleCallOptions

export interface OpenRouterClientConfig {
  readonly apiKey?: string
  readonly endpoint?: string
  readonly auth?: AuthApplicator
  readonly modelsDev?: ModelsDevClient
}

