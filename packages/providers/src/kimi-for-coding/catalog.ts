import { Effect } from "effect"
import { ModelCatalogError, type ModelCatalog } from "@magnitudedev/ai"
import type { KimiForCodingModelInfo } from "./contract"

export const KIMI_FOR_CODING_MODEL_ID = "kimi-for-coding"

const MODEL: KimiForCodingModelInfo = {
  providerId: "kimi-for-coding",
  providerModelId: KIMI_FOR_CODING_MODEL_ID,
  modelFamilyId: "kimi-k2",
  displayName: "Kimi Code",
  contextWindow: 262_144,
  maxOutputTokens: 32_768,
  capabilities: {
    vision: true,
    toolCalls: true,
    structuredOutput: true,
    grammar: false,
    toolChoiceModes: ["auto", "none"],
  },
  pricing: { input: 0, output: 0, cached_input: null },
  reasoningEfforts: ["none", "low", "medium", "high", "xhigh", "max"],
  openWeightStatus: "open",
  metadataSource: "official_fallback",
  upstreamFamily: "kimi-k2",
  modalities: { input: ["text", "image", "video"], output: ["text"] },
}

export function createKimiForCodingCatalog(): ModelCatalog<KimiForCodingModelInfo> {
  const list = Effect.succeed([MODEL] as const)
  return {
    list,
    refresh: list,
    get: (_providerId, providerModelId) => providerModelId === KIMI_FOR_CODING_MODEL_ID
      ? Effect.succeed(MODEL)
      : Effect.fail(new ModelCatalogError({
          message: `Model not found: kimi-for-coding/${providerModelId}`,
        })),
  }
}
