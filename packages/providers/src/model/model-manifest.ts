/**
 * Model manifest — the hand-maintained source of truth for canonical models.
 *
 * Adding a new model: add an entry here, then re-run `bun run scripts/fetch-templates.ts`
 * to generate the template files and the registry.
 */

import type { ModelId } from './canonical-model'

export interface ModelManifestEntry {
  id: ModelId
  name: string
  family: string
  hfRepo: string
}

export const MODEL_MANIFEST: readonly ModelManifestEntry[] = [
  { id: "kimi-k2.5", name: "Kimi K2.5", family: "kimi", hfRepo: "moonshotai/Kimi-K2.5" },
  { id: "kimi-k2.6", name: "Kimi K2.6", family: "kimi", hfRepo: "moonshotai/Kimi-K2.6" },
  { id: "glm-4.7", name: "GLM-4.7", family: "glm", hfRepo: "zai-org/GLM-4.7" },
  { id: "glm-5", name: "GLM-5", family: "glm", hfRepo: "zai-org/GLM-5" },
  { id: "glm-5.1", name: "GLM-5.1", family: "glm", hfRepo: "zai-org/GLM-5.1" },
  { id: "minimax-m2.5", name: "MiniMax M2.5", family: "minimax", hfRepo: "MiniMaxAI/MiniMax-M2.5" },
  { id: "minimax-m2.7", name: "MiniMax M2.7", family: "minimax", hfRepo: "MiniMaxAI/MiniMax-M2.7" },
]
