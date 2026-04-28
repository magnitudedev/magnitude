import { defineModels } from "../shared"

export const moonshotAiModels = defineModels("moonshotai", "Moonshot AI", [
  { id: "kimi-k2.6", name: "Kimi K2.6", releaseDate: "2025-01-01", supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 131072, contextWindow: 262144 },
  { id: "kimi-k2.5", name: "Kimi K2.5", releaseDate: "2025-01-01", supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 131072, contextWindow: 262144 },
] as const)
