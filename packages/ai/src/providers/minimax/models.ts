import { defineModels } from "../shared"

export const models = defineModels("minimax", "MiniMax", [
  { id: "MiniMax-M2.7", name: "MiniMax M2.7", releaseDate: "2025-03-01", supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 1000000 },
  { id: "MiniMax-M2.5", name: "MiniMax M2.5", releaseDate: "2025-01-01", supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 1000000 },
] as const)
