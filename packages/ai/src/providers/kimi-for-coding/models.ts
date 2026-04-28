import { defineModels } from "../shared"

export const models = defineModels("kimi-for-coding", "Kimi for Coding", [
  { id: "k2p6", name: "K2p6", releaseDate: "2025-01-01", supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 131072, contextWindow: 262144 },
  { id: "k2p5", name: "K2p5", releaseDate: "2025-01-01", supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 131072, contextWindow: 262144 },
] as const)
