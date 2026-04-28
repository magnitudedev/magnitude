import { defineModels } from "../shared"

export const models = defineModels("anthropic", "Anthropic", [
  { id: "claude-opus-4-7", name: "Claude Opus 4.7", releaseDate: "2026-01-01", supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 128000, contextWindow: 200000 },
  { id: "claude-opus-4-6", name: "Claude Opus 4.6", releaseDate: "2026-01-01", supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 128000, contextWindow: 200000 },
  { id: "claude-sonnet-4-6", name: "Claude Sonnet 4.6", releaseDate: "2026-01-01", supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 64000, contextWindow: 200000 },
  { id: "claude-haiku-4-5", name: "Claude Haiku 4.5", releaseDate: "2026-01-01", supportsToolCalls: true, supportsReasoning: false, supportsVision: true, maxOutputTokens: 16000, contextWindow: 200000 },
] as const)
