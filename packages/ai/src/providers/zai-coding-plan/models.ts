import { defineModels } from "../shared"

export const models = defineModels("zai-coding-plan", "Z.AI Coding Plan", [
  { id: "glm-5.1", name: "GLM-5.1", releaseDate: "2025-01-01", supportsToolCalls: true, supportsReasoning: true, supportsVision: false, maxOutputTokens: 131072, contextWindow: 1000000 },
  { id: "glm-5", name: "GLM-5", releaseDate: "2025-01-01", supportsToolCalls: true, supportsReasoning: true, supportsVision: false, maxOutputTokens: 131072, contextWindow: 1000000 },
  { id: "glm-4.7", name: "GLM-4.7", releaseDate: "2024-01-01", supportsToolCalls: true, supportsReasoning: true, supportsVision: false, maxOutputTokens: 131072, contextWindow: 1000000 },
] as const)
