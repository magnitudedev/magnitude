import type { AuthApplicator } from "@magnitudedev/ai"
import type { OpenAiCompatibleCallOptions, OpenAiCompatibleModelInfo } from "../openai-compatible"

export interface KimiForCodingModelInfo extends OpenAiCompatibleModelInfo {
  readonly providerId: "kimi-for-coding"
}

export type KimiForCodingCallOptions = OpenAiCompatibleCallOptions

export interface KimiForCodingClientConfig {
  readonly apiKey?: string
  readonly endpoint?: string
  readonly auth?: AuthApplicator
}

