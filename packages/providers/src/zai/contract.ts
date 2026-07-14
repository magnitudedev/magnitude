import type { AuthApplicator } from "@magnitudedev/ai"
import type { ModelsDevClient } from "../catalog/models-dev"
import type { OpenAiCompatibleCallOptions, OpenAiCompatibleModelInfo } from "../openai-compatible"

export interface ZaiModelInfo extends OpenAiCompatibleModelInfo {
  readonly providerId: "zai"
}

export type ZaiCallOptions = OpenAiCompatibleCallOptions

export interface ZaiClientConfig {
  readonly apiKey?: string
  readonly endpoint?: string
  readonly auth?: AuthApplicator
  readonly modelsDev?: ModelsDevClient
}

