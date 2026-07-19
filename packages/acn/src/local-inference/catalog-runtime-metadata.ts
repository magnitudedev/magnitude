import type { LocalModelRuntimeMetadata } from "./types"

/**
 * Runtime-relevant metadata read from the pinned GGUF revisions in the catalog.
 * Unlike display names and marketed parameter counts, these values come from
 * the artifacts' GGUF metadata and describe the tensors llama.cpp will load.
 */
export const CATALOG_RUNTIME_METADATA = {
  "qwen3.5-4b": {
    ggufArchitecture: "qwen35",
    parameterCount: 4_205_751_296,
    blockCount: 32,
    embeddingLength: 2_560,
    attentionHeadCount: 16,
    attentionCache: [{ layerCount: 8, keyHeads: 4, keyLength: 256, valueHeads: 4, valueLength: 256 }],
    recurrentState: { layerCount: 24, innerSize: 4_096, stateSize: 128, convolutionWidth: 4, bytesPerElement: 4 },
  },
  "qwen3.5-9b": {
    ggufArchitecture: "qwen35",
    parameterCount: 8_953_803_264,
    blockCount: 32,
    embeddingLength: 4_096,
    attentionHeadCount: 16,
    attentionCache: [{ layerCount: 8, keyHeads: 4, keyLength: 256, valueHeads: 4, valueLength: 256 }],
    recurrentState: { layerCount: 24, innerSize: 4_096, stateSize: 128, convolutionWidth: 4, bytesPerElement: 4 },
  },
  "gemma-4-e2b-it-qat": {
    ggufArchitecture: "gemma4",
    parameterCount: 4_628_569_635,
    blockCount: 35,
    embeddingLength: 1_536,
    attentionHeadCount: 8,
    attentionCache: [
      { layerCount: 7, keyHeads: 1, keyLength: 512, valueHeads: 1, valueLength: 512 },
      { layerCount: 28, keyHeads: 1, keyLength: 256, valueHeads: 1, valueLength: 256, contextLimitTokens: 512 },
    ],
  },
  "gemma-4-12b-it-qat": {
    ggufArchitecture: "gemma4",
    parameterCount: 11_907_350_576,
    blockCount: 48,
    embeddingLength: 3_840,
    attentionHeadCount: 16,
    attentionCache: [
      { layerCount: 8, keyHeads: 1, keyLength: 512, valueHeads: 1, valueLength: 512 },
      { layerCount: 40, keyHeads: 8, keyLength: 256, valueHeads: 8, valueLength: 256, contextLimitTokens: 1_024 },
    ],
  },
  "gemma-4-26b-a4b-it-qat": {
    ggufArchitecture: "gemma4",
    parameterCount: 25_233_142_046,
    blockCount: 30,
    embeddingLength: 2_816,
    attentionHeadCount: 16,
    attentionCache: [
      { layerCount: 5, keyHeads: 2, keyLength: 512, valueHeads: 2, valueLength: 512 },
      { layerCount: 25, keyHeads: 8, keyLength: 256, valueHeads: 8, valueLength: 256, contextLimitTokens: 1_024 },
    ],
  },
  "gemma-4-31b-it-qat": {
    ggufArchitecture: "gemma4",
    parameterCount: 30_697_345_596,
    blockCount: 60,
    embeddingLength: 5_376,
    attentionHeadCount: 32,
    attentionCache: [
      { layerCount: 10, keyHeads: 4, keyLength: 512, valueHeads: 4, valueLength: 512 },
      { layerCount: 50, keyHeads: 16, keyLength: 256, valueHeads: 16, valueLength: 256, contextLimitTokens: 1_024 },
    ],
  },
  "qwen3.6-27b": {
    ggufArchitecture: "qwen35",
    parameterCount: 26_895_998_464,
    blockCount: 64,
    embeddingLength: 5_120,
    attentionHeadCount: 24,
    attentionCache: [{ layerCount: 16, keyHeads: 4, keyLength: 256, valueHeads: 4, valueLength: 256 }],
    recurrentState: { layerCount: 48, innerSize: 6_144, stateSize: 128, convolutionWidth: 4, bytesPerElement: 4 },
  },
  "qwen3.6-35b-a3b": {
    ggufArchitecture: "qwen35moe",
    parameterCount: 34_660_610_688,
    blockCount: 40,
    embeddingLength: 2_048,
    attentionHeadCount: 16,
    attentionCache: [{ layerCount: 10, keyHeads: 2, keyLength: 256, valueHeads: 2, valueLength: 256 }],
    recurrentState: { layerCount: 30, innerSize: 4_096, stateSize: 128, convolutionWidth: 4, bytesPerElement: 4 },
  },
  "qwen3.5-122b-a10b": {
    ggufArchitecture: "qwen35moe",
    parameterCount: 122_111_526_912,
    blockCount: 48,
    embeddingLength: 3_072,
    attentionHeadCount: 32,
    attentionCache: [{ layerCount: 12, keyHeads: 2, keyLength: 256, valueHeads: 2, valueLength: 256 }],
    recurrentState: { layerCount: 36, innerSize: 8_192, stateSize: 128, convolutionWidth: 4, bytesPerElement: 4 },
  },
  "nemotron-3-super-120b-a12b": {
    ggufArchitecture: "nemotron_h_moe",
    parameterCount: 120_668_707_840,
    blockCount: 88,
    embeddingLength: 4_096,
    attentionHeadCount: 32,
    attentionCache: [{ layerCount: 8, keyHeads: 2, keyLength: 128, valueHeads: 2, valueLength: 128 }],
    recurrentState: { layerCount: 80, innerSize: 8_192, stateSize: 128, convolutionWidth: 4, bytesPerElement: 4 },
  },
  "deepseek-v4-flash": {
    ggufArchitecture: "deepseek4",
    parameterCount: 284_334_567_511,
    blockCount: 43,
    embeddingLength: 4_096,
    attentionHeadCount: 64,
    attentionCache: [{ layerCount: 43, keyHeads: 1, keyLength: 512, valueHeads: 1, valueLength: 512 }],
  },
  "nemotron-3-ultra-550b-a55b": {
    ggufArchitecture: "nemotron_h_moe",
    parameterCount: 549_308_993_536,
    blockCount: 108,
    embeddingLength: 8_192,
    attentionHeadCount: 64,
    attentionCache: [{ layerCount: 13, keyHeads: 2, keyLength: 128, valueHeads: 2, valueLength: 128 }],
    recurrentState: { layerCount: 95, innerSize: 16_384, stateSize: 128, convolutionWidth: 4, bytesPerElement: 4 },
  },
  "glm-5.2": {
    ggufArchitecture: "glm-dsa",
    parameterCount: 753_864_139_008,
    blockCount: 79,
    embeddingLength: 6_144,
    attentionHeadCount: 64,
    attentionCache: [{ layerCount: 79, keyHeads: 1, keyLength: 576, valueHeads: 1, valueLength: 512 }],
  },
} as const satisfies Record<string, LocalModelRuntimeMetadata>

export type CatalogRuntimeModelId = keyof typeof CATALOG_RUNTIME_METADATA

export const catalogRuntimeMetadata = (modelId: string): LocalModelRuntimeMetadata => {
  const metadata = CATALOG_RUNTIME_METADATA[modelId as CatalogRuntimeModelId]
  if (!metadata) throw new Error(`Missing GGUF runtime metadata for catalog model ${modelId}`)
  return metadata
}
