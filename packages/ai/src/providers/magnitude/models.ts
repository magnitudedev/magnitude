import { defineModels } from "../shared"

export const models = defineModels("magnitude", "Magnitude", [
  { id: "glm-4.7", name: "GLM-4.7", releaseDate: "2024-01-01", supportsToolCalls: true, supportsReasoning: true, supportsVision: false, maxOutputTokens: 202000, contextWindow: 202000, supportsGrammar: true },
  { id: "glm-5", name: "GLM-5", releaseDate: "2025-01-01", supportsToolCalls: true, supportsReasoning: true, supportsVision: false, maxOutputTokens: 202000, contextWindow: 202000, supportsGrammar: true },
  { id: "glm-5.1", name: "GLM-5.1", releaseDate: "2025-01-01", supportsToolCalls: true, supportsReasoning: true, supportsVision: false, maxOutputTokens: 202000, contextWindow: 202000, supportsGrammar: true },
  { id: "kimi-k2.5", name: "Kimi K2.5", releaseDate: "2025-01-01", supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 262000, contextWindow: 262000, supportsGrammar: true },
  { id: "kimi-k2.6", name: "Kimi K2.6", releaseDate: "2026-04-20", supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 262000, contextWindow: 262000, supportsGrammar: true },
  { id: "minimax-m2.5", name: "MiniMax M2.5", releaseDate: "2025-01-01", supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 196000, contextWindow: 196000, supportsGrammar: false },
  { id: "minimax-m2.7", name: "MiniMax M2.7", releaseDate: "2025-03-01", supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 196000, contextWindow: 196000, supportsGrammar: false },
] as const)
