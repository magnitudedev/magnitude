import type { AuthApplicator } from "@magnitudedev/ai"
import type { ModelsDevClient } from "../catalog/models-dev"
import type { OpenAiCompatibleCallOptions, OpenAiCompatibleModelInfo } from "../openai-compatible"

export interface VercelModelInfo extends OpenAiCompatibleModelInfo {
  readonly providerId: "vercel"
}

export type VercelCallOptions = OpenAiCompatibleCallOptions

export interface VercelClientConfig {
  readonly apiKey?: string
  readonly endpoint?: string
  readonly auth?: AuthApplicator
  readonly modelsDev?: ModelsDevClient
}

