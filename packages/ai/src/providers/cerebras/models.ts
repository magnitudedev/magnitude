import { defineModels } from "../shared"

export const models = defineModels("cerebras", "Cerebras", [
  { id: "gpt-oss-120b", name: "GPT-OSS 120B", releaseDate: "2025-01-01", supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 32768, contextWindow: 131072 },
] as const)
