import { defineModels } from "../shared"

export const models = defineModels("vercel", "Vercel AI Gateway", [
  { id: "anthropic/claude-opus-4.7", name: "Claude Opus 4.7", releaseDate: "2026-01-01", supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 128000, contextWindow: 200000 },
  { id: "anthropic/claude-opus-4.6", name: "Claude Opus 4.6", releaseDate: "2026-01-01", supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 128000, contextWindow: 200000 },
  { id: "anthropic/claude-sonnet-4.6", name: "Claude Sonnet 4.6", releaseDate: "2026-01-01", supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 64000, contextWindow: 200000 },
  { id: "anthropic/claude-haiku-4.5", name: "Claude Haiku 4.5", releaseDate: "2026-01-01", supportsToolCalls: true, supportsReasoning: false, supportsVision: true, maxOutputTokens: 16000, contextWindow: 200000 },
  { id: "zai/glm-5.1", name: "GLM 5.1", releaseDate: "2025-01-01", supportsToolCalls: true, supportsReasoning: true, supportsVision: false, maxOutputTokens: 131072, contextWindow: 262144, supportsGrammar: true },
  { id: "moonshotai/kimi-k2.6", name: "Kimi K2.6", releaseDate: "2026-04-20", supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 262000, contextWindow: 262000, supportsGrammar: true },
  { id: "deepseek/deepseek-v4-pro", name: "DeepSeek V4 Pro", releaseDate: "2025-01-01", supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 262144, supportsGrammar: false },
] as const)
