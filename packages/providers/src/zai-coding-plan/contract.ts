import type { AuthApplicator } from "@magnitudedev/ai"
import type { ModelsDevClient } from "../catalog/models-dev"
import type { OpenAiCompatibleCallOptions, OpenAiCompatibleModelInfo } from "../openai-compatible"

export interface ZaiCodingPlanModelInfo extends OpenAiCompatibleModelInfo {
  readonly providerId: "zai-coding-plan"
}

export type ZaiCodingPlanCallOptions = OpenAiCompatibleCallOptions

export interface ZaiCodingPlanClientConfig {
  readonly apiKey?: string
  readonly endpoint?: string
  readonly auth?: AuthApplicator
  readonly modelsDev?: ModelsDevClient
}

