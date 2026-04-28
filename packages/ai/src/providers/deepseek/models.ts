import { defineModels } from "../shared"

export const models = defineModels("deepseek", "DeepSeek", [
  { id: "deepseek-v4-flash", name: "DeepSeek V4 Flash", releaseDate: "2026-04-24", supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 384000, contextWindow: 1000000, supportsGrammar: false },
  { id: "deepseek-v4-pro", name: "DeepSeek V4 Pro", releaseDate: "2026-04-24", supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 384000, contextWindow: 1000000, supportsGrammar: false },
] as const)
