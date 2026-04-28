import type { ModelId } from "../../lib/model/canonical-model"
import { defineModels } from "../shared"

export const models = defineModels("fireworks-ai", "Fireworks AI", [
  {
    id: "accounts/fireworks/models/kimi-k2p6",
    name: "Kimi K2.6",
    releaseDate: "2025-01-01",
    supportsToolCalls: true,
    supportsReasoning: true,
    supportsVision: true,
    maxOutputTokens: 131072,
    contextWindow: 262144,
    supportsGrammar: true,
    paradigm: "native",
    canonicalModelId: "kimi-k2.6" as ModelId,
  },
  {
    id: "accounts/fireworks/models/glm-5p1",
    name: "GLM 5.1",
    releaseDate: "2025-01-01",
    supportsToolCalls: true,
    supportsReasoning: true,
    supportsVision: false,
    maxOutputTokens: 131072,
    contextWindow: 262144,
    supportsGrammar: true,
  },
] as const)
