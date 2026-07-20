import type { LocalModelCatalogEntry } from "./types"

const APACHE = {
  id: "apache-2.0",
  url: "https://www.apache.org/licenses/LICENSE-2.0",
  acknowledgementRequired: false,
} as const
const QUANT_STUDY = "https://arxiv.org/abs/2606.19558"
const CONTEXTS = [100_000, 200_000] as const

const fidelity = (quantTag: string, quantAwareCheckpoint = false) => ({
  quantAwareCheckpoint,
  fidelityRank: quantTag.includes("Q8") ? 80 : quantTag.includes("Q6") ? 60 : quantTag.includes("Q5") ? 50 : quantAwareCheckpoint ? 58 : 40,
  fidelityLabel: quantAwareCheckpoint
    ? "Near-original fidelity from quantization-aware training"
    : quantTag.includes("Q8") ? "Near-original fidelity with the least quality loss"
    : quantTag.includes("Q6") ? "Very high fidelity with minimal quality loss"
    : quantTag.includes("Q5") ? "High fidelity with only minor quality loss"
    : "Good fidelity with some possible quality loss",
  fidelityEvidence: quantAwareCheckpoint
    ? "Curated checkpoint-level fidelity guidance; ICN supplies the artifact-specific model and fit facts."
    : "Curated cross-model fidelity guidance; ICN supplies the artifact-specific model and fit facts.",
  fidelitySourceUrl: QUANT_STUDY,
})

const entry = (input: {
  id: string
  family: string
  displayName: string
  repo: string
  revision: string
  primaryGguf: string
  quantTag: string
  quality: number
  contexts?: readonly number[]
  quantAware?: boolean
  license?: LocalModelCatalogEntry["license"]
}): LocalModelCatalogEntry => ({
  id: `${input.id}:${input.quantTag}`,
  modelId: input.id,
  family: input.family,
  displayName: input.displayName,
  repo: input.repo,
  revision: input.revision,
  primaryGguf: input.primaryGguf,
  additionalComponents: [],
  supportedContextTokens: input.contexts ?? CONTEXTS,
  quantTag: input.quantTag,
  quantization: fidelity(input.quantTag, input.quantAware),
  license: input.license ?? APACHE,
  modelQualityRank: input.quality,
})

const qwenVariants = (input: {
  id: string
  displayName: string
  repo: string
  revision: string
  quality: number
}) => (["UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "UD-Q8_K_XL"] as const).map((quantTag) => entry({
  ...input,
  family: input.id.startsWith("qwen3.6") ? "qwen3.6" : "qwen3.5",
  quantTag,
  primaryGguf: `${input.displayName.replaceAll(" ", "-")}-${quantTag}.gguf`,
}))

export const LOCAL_MODEL_CATALOG: readonly LocalModelCatalogEntry[] = [
  ...qwenVariants({ id: "qwen3.5-4b", displayName: "Qwen3.5 4B", repo: "unsloth/Qwen3.5-4B-GGUF", revision: "e87f176479d0855a907a41277aca2f8ee7a09523", quality: 10 }),
  ...qwenVariants({ id: "qwen3.5-9b", displayName: "Qwen3.5 9B", repo: "unsloth/Qwen3.5-9B-GGUF", revision: "3885219b6810b007914f3a7950a8d1b469d598a5", quality: 20 }),
  ...qwenVariants({ id: "qwen3.6-27b", displayName: "Qwen3.6 27B", repo: "unsloth/Qwen3.6-27B-GGUF", revision: "82d411acf4a06cfb8d9b073a5211bf410bfc29bf", quality: 45 }),
  ...qwenVariants({ id: "qwen3.6-35b-a3b", displayName: "Qwen3.6 35B-A3B", repo: "unsloth/Qwen3.6-35B-A3B-GGUF", revision: "a483e9e6cbd595906af30beda3187c2663a1118c", quality: 50 }),
  entry({ id: "gemma-4-e2b-it-qat", family: "gemma-4", displayName: "Gemma 4 E2B", repo: "unsloth/gemma-4-E2B-it-qat-GGUF", revision: "2ea637031baa8dc847d64b5dbb7011fd6a445849", primaryGguf: "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf", quantTag: "UD-Q4_K_XL", quality: 8, quantAware: true, contexts: [100_000] }),
  entry({ id: "gemma-4-12b-it-qat", family: "gemma-4", displayName: "Gemma 4 12B", repo: "unsloth/gemma-4-12B-it-qat-GGUF", revision: "f18012b8f690e563b7f872cb764b4cb3de90b14a", primaryGguf: "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf", quantTag: "UD-Q4_K_XL", quality: 30, quantAware: true }),
  entry({ id: "gemma-4-26b-a4b-it-qat", family: "gemma-4", displayName: "Gemma 4 26B-A4B", repo: "unsloth/gemma-4-26B-A4B-it-qat-GGUF", revision: "c1f25db7cf31985b52caa1db777eb72d17ca1c7c", primaryGguf: "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf", quantTag: "UD-Q4_K_XL", quality: 34, quantAware: true }),
  entry({ id: "gemma-4-31b-it-qat", family: "gemma-4", displayName: "Gemma 4 31B", repo: "unsloth/gemma-4-31B-it-qat-GGUF", revision: "1f1e54258d4a2cf7522856a5789045d9f2ea6d16", primaryGguf: "gemma-4-31B-it-qat-UD-Q4_K_XL.gguf", quantTag: "UD-Q4_K_XL", quality: 36, quantAware: true }),
  entry({ id: "qwen3.5-122b-a10b", family: "qwen3.5", displayName: "Qwen3.5 122B-A10B", repo: "unsloth/Qwen3.5-122B-A10B-GGUF", revision: "51eab4d59d53f573fb9206cb3ce613f1d0aa392b", primaryGguf: "UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf", quantTag: "UD-Q4_K_XL", quality: 64 }),
  entry({ id: "nemotron-3-super-120b-a12b", family: "nemotron-3", displayName: "NVIDIA Nemotron 3 Super 120B-A12B", repo: "unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF", revision: "036038fb30334a2d56a146c6f0d4871ab5edccbb", primaryGguf: "MXFP4_MOE/NVIDIA-Nemotron-3-Super-120B-A12B-MXFP4_MOE-00001-of-00003.gguf", quantTag: "MXFP4_MOE", quality: 65, contexts: [100_000] }),
  entry({ id: "deepseek-v4-flash", family: "deepseek-v4", displayName: "DeepSeek V4 Flash 284B-A13B", repo: "unsloth/DeepSeek-V4-Flash-GGUF", revision: "e3aa0d6a5fa4f820d9e132ac1fd1d01e1b2b49e0", primaryGguf: "UD-Q8_K_XL/DeepSeek-V4-Flash-UD-Q8_K_XL-00001-of-00005.gguf", quantTag: "UD-Q8_K_XL", quality: 72 }),
  entry({ id: "nemotron-3-ultra-550b-a55b", family: "nemotron-3", displayName: "NVIDIA Nemotron 3 Ultra 550B-A55B", repo: "unsloth/NVIDIA-Nemotron-3-Ultra-550B-A55B-GGUF", revision: "2fb7d5b3f4eae7aedb18b4839b6a6300111e46f6", primaryGguf: "MXFP4_MOE/NVIDIA-Nemotron-3-Ultra-550B-A55B-MXFP4_MOE-00001-of-00009.gguf", quantTag: "MXFP4_MOE", quality: 76 }),
  entry({ id: "glm-5.2", family: "glm-5.2", displayName: "GLM 5.2 753B-A40B", repo: "unsloth/GLM-5.2-GGUF", revision: "abc55e72527792c6e77069c99b4cb7de16fa9f23", primaryGguf: "UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf", quantTag: "UD-Q4_K_XL", quality: 80 }),
]

export const catalogSourcePageUrl = (model: LocalModelCatalogEntry): string =>
  `https://huggingface.co/${model.repo}/blob/${model.revision}/${model.primaryGguf.split("/").map(encodeURIComponent).join("/")}`
