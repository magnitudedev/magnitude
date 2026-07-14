import type { AuthApplicator } from "@magnitudedev/ai"
import type { ModelsDevClient } from "../catalog/models-dev"
import type {
  OpenAiCompatibleCallOptions,
  OpenAiCompatibleModelInfo,
} from "../openai-compatible"

export interface DeepSeekModelInfo extends OpenAiCompatibleModelInfo {
  readonly providerId: "deepseek"
}

export type DeepSeekCallOptions = OpenAiCompatibleCallOptions

export interface DeepSeekClientConfig {
  readonly apiKey?: string
  readonly endpoint?: string
  readonly auth?: AuthApplicator
  readonly modelsDev?: ModelsDevClient
}

