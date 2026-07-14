import type { AuthApplicator } from "@magnitudedev/ai"
import type { ModelsDevClient } from "../catalog/models-dev"
import type { OpenAiCompatibleCallOptions, OpenAiCompatibleModelInfo } from "../openai-compatible"

export interface KimiApiModelInfo extends OpenAiCompatibleModelInfo {
  readonly providerId: "kimi-api"
}

export type KimiApiCallOptions = OpenAiCompatibleCallOptions

export interface KimiApiClientConfig {
  readonly apiKey?: string
  readonly endpoint?: string
  readonly auth?: AuthApplicator
  readonly modelsDev?: ModelsDevClient
}

