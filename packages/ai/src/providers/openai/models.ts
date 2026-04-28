import { defineModels } from "../shared"

export const models = defineModels("openai", "OpenAI", [
  { id: "gpt-5.5", name: "GPT-5.5", releaseDate: "2026-06-01", supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 128000, contextWindow: 400000 },
  { id: "gpt-5.4", name: "GPT-5.4", releaseDate: "2026-06-01", supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 128000, contextWindow: 400000 },
  { id: "gpt-5.3-codex", name: "GPT-5.3 Codex", releaseDate: "2026-01-01", supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 128000, contextWindow: 400000 },
  { id: "gpt-5.2-codex", name: "GPT-5.2 Codex", releaseDate: "2025-01-01", supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 128000, contextWindow: 400000 },
  { id: "gpt-5.2", name: "GPT-5.2", releaseDate: "2025-01-01", supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 128000, contextWindow: 400000 },
] as const)
