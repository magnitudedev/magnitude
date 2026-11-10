// Throwaway prototype data for the models overhaul (specs/26-11-09/models-overhaul/spec.md).
// File sizes are real Hugging Face sizes from 2026-11-09. Speeds come from logic.ts, a rough model
// calibrated to engine assessments on an M4 Max 64 GB; every other machine is illustrative.

export type Country = "US" | "China"

export interface Machine {
  readonly id: string
  readonly name: string
  readonly detail: string
  readonly photo: string
  readonly kind: "unified" | "discrete"
  readonly memoryGB: number
  readonly vramGB: number
  readonly bandwidth: number
  readonly ssd: number
  readonly pcie: number
  readonly freeDiskGB: number
  readonly measured: boolean
  readonly gpus?: readonly { readonly name: string; readonly vramGB: number }[]
}

export const machines: readonly Machine[] = [
  { id: "m4max64", name: "MacBook Pro 16-inch", detail: "Apple M4 Max · 64 GB unified memory · 698 GB free", photo: "book16", kind: "unified", memoryGB: 64, vramGB: 0, bandwidth: 546, ssd: 7, pcie: 0, freeDiskGB: 698, measured: true },
  { id: "air16", name: "MacBook Air 15-inch", detail: "Apple M4 · 16 GB unified memory · 180 GB free", photo: "air15-m4", kind: "unified", memoryGB: 16, vramGB: 0, bandwidth: 120, ssd: 3, pcie: 0, freeDiskGB: 180, measured: false },
  { id: "studio256", name: "Mac Studio", detail: "Apple M3 Ultra · 256 GB unified memory · 3.1 TB free", photo: "studio", kind: "unified", memoryGB: 256, vramGB: 0, bandwidth: 819, ssd: 7, pcie: 0, freeDiskGB: 3100, measured: false },
  { id: "rtx4060", name: "Desktop PC", detail: "RTX 4060 8 GB · 32 GB RAM · 650 GB free", photo: "rtx5060", kind: "discrete", memoryGB: 32, vramGB: 8, bandwidth: 272, ssd: 5, pcie: 12, freeDiskGB: 650, measured: false },
  { id: "rtx4090", name: "Desktop PC", detail: "RTX 4090 24 GB · 64 GB RAM · 1.4 TB free", photo: "rtx4090", kind: "discrete", memoryGB: 64, vramGB: 24, bandwidth: 1008, ssd: 6, pcie: 25, freeDiskGB: 1400, measured: false },
  { id: "dual3090", name: "Workstation", detail: "2× RTX 3090 24 GB · 128 GB RAM · 2.6 TB free", photo: "rtx3090", kind: "discrete", memoryGB: 128, vramGB: 24, bandwidth: 936, ssd: 6, pcie: 25, freeDiskGB: 2600, measured: false, gpus: [{ name: "RTX 3090", vramGB: 24 }, { name: "RTX 3090", vramGB: 24 }] },
  { id: "mixed", name: "Desktop PC", detail: "RTX 4090 24 GB + RTX 3060 12 GB · 64 GB RAM · 1.1 TB free", photo: "rtx4090", kind: "discrete", memoryGB: 64, vramGB: 24, bandwidth: 1008, ssd: 6, pcie: 25, freeDiskGB: 1100, measured: false, gpus: [{ name: "RTX 4090", vramGB: 24 }, { name: "RTX 3060", vramGB: 12 }] },
  { id: "rtx5090", name: "Desktop PC", detail: "RTX 5090 32 GB · 128 GB RAM · 3.2 TB free", photo: "rtx5090", kind: "discrete", memoryGB: 128, vramGB: 32, bandwidth: 1792, ssd: 7, pcie: 50, freeDiskGB: 3200, measured: false },
]

export interface Quant {
  readonly publisher: string
  readonly repo: string
  readonly name: string
  readonly sizeGB: number
}

export interface Model {
  readonly id: string
  readonly name: string
  readonly lab: string
  readonly developer: string
  readonly country: Country
  readonly family: string
  readonly paramsB: number
  readonly activeB: number | null
  readonly denseShare: number
  readonly contextK: number
  readonly kvPerK: number
  readonly sideTable?: { readonly gb: number; readonly paramsB: number }
  readonly score: number | null
  readonly license: string
  readonly description: string
  readonly vision: boolean
  readonly comingSoon?: boolean
  readonly fineTuneOf?: string
  readonly drafter?: { readonly method: string; readonly repo: string; readonly sizeGB: number }
  readonly downloads?: string
  readonly quants: readonly Quant[]
}

const q = (publisher: string, repo: string, name: string, sizeGB: number): Quant => ({ publisher, repo: `${publisher}/${repo}`, name, sizeGB })

/** The curated pool behind "Best for this computer": models we've tested, each with a ladder that may mix publishers. */
export const pool: readonly Model[] = [
  {
    id: "qwen3.8-flash-next", contextK: 256, kvPerK: 0.027, drafter: { method: "MTP", repo: "unsloth/Qwen3.8-Flash-Next-GGUF", sizeGB: 2.8 }, name: "Qwen3.8 Flash Next", lab: "Qwen", developer: "Alibaba", country: "China", family: "Qwen",
    paramsB: 177, activeB: 6, denseShare: 0.08, sideTable: { gb: 28.8, paramsB: 51 }, score: 39.8, license: "Qwen license", vision: true,
    description: "The strongest open model that streams on a laptop. 512 experts, 6B active per token.",
    quants: [
      q("ISTA-DASLab", "Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "Q2_0", 66.4),
      q("ISTA-DASLab", "Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "IQ3_S", 83.6),
      q("unsloth", "Qwen3.8-Flash-Next-GGUF", "UD-Q4_K_XL", 111.3),
      q("unsloth", "Qwen3.8-Flash-Next-GGUF", "UD-Q6_K_XL", 169.2),
    ],
  },
  {
    id: "qwen3.8-27b", contextK: 256, kvPerK: 0.064, drafter: { method: "DFlash 2", repo: "magnitudedev/Qwen3.8-27B-DFlash2-GGUF", sizeGB: 2.1 }, name: "Qwen3.8 27B", lab: "Qwen", developer: "Alibaba", country: "China", family: "Qwen",
    paramsB: 27, activeB: null, denseShare: 1, score: 33.7, license: "Apache 2.0", vision: true,
    description: "Dense 27B. Very capable when it fits whole in fast memory.",
    quants: [
      q("unsloth", "Qwen3.8-27B-GGUF", "UD-IQ3_XXS", 10.9),
      q("unsloth", "Qwen3.8-27B-GGUF", "UD-Q3_K_XL", 13.1),
      q("unsloth", "Qwen3.8-27B-GGUF", "UD-Q4_K_XL", 17.6),
      q("unsloth", "Qwen3.8-27B-GGUF", "UD-Q6_K_XL", 25.3),
    ],
  },
  {
    id: "nemotron-3.5-super-vl", contextK: 256, kvPerK: 0.02, drafter: { method: "LiLiCorr", repo: "magnitudedev/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr-GGUF", sizeGB: 1.3 }, name: "Nemotron 3.5 Super VL", lab: "NVIDIA", developer: "NVIDIA", country: "US", family: "Nemotron",
    paramsB: 120, activeB: 12, denseShare: 0.08, score: 26.5, license: "OpenMDW 1.1", vision: true,
    description: "Hybrid MoE vision model for reasoning, coding and agents. 12B active.",
    quants: [
      q("unsloth", "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF", "UD-Q2_K_XL", 57.8),
      q("unsloth", "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF", "UD-Q3_K_XL", 65.8),
      q("unsloth", "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF", "UD-Q4_K_XL", 86.9),
      q("unsloth", "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF", "UD-Q6_K_XL", 121.0),
    ],
  },
  {
    id: "gemma-4-31b", contextK: 128, kvPerK: 0.12, drafter: { method: "MTP", repo: "magnitudedev/gemma-4-31B-it-qat-MTP-GGUF", sizeGB: 0.3 }, name: "Gemma 4 31B", lab: "Gemma", developer: "Google", country: "US", family: "Gemma",
    paramsB: 30.7, activeB: null, denseShare: 1, score: 19.0, license: "Gemma terms", vision: true,
    description: "Google's dense 31B, trained to hold up at 4-bit.",
    quants: [q("unsloth", "gemma-4-31B-it-qat-GGUF", "UD-Q4_K_XL", 17.3)],
  },
  {
    id: "qwen3.6-35b-a3b", contextK: 256, kvPerK: 0.021, drafter: { method: "DFlash", repo: "magnitudedev/Qwen3.6-35B-A3B-DFlash-GGUF", sizeGB: 0.4 }, name: "Qwen3.6 35B-A3B", lab: "Qwen", developer: "Alibaba", country: "China", family: "Qwen",
    paramsB: 35, activeB: 3, denseShare: 0.06, score: 18.2, license: "Apache 2.0", vision: true,
    description: "Fast everyday MoE. 3B active, so it stays quick even when streaming.",
    quants: [
      q("unsloth", "Qwen3.6-35B-A3B-MTP-GGUF", "UD-Q2_K_XL", 12.6),
      q("unsloth", "Qwen3.6-35B-A3B-MTP-GGUF", "UD-Q3_K_XL", 17.2),
      q("unsloth", "Qwen3.6-35B-A3B-MTP-GGUF", "UD-Q4_K_XL", 22.9),
      q("unsloth", "Qwen3.6-35B-A3B-MTP-GGUF", "UD-Q6_K_XL", 32.6),
    ],
  },
  {
    id: "gemma-4-26b-a4b", contextK: 128, kvPerK: 0.06, drafter: { method: "MTP", repo: "magnitudedev/gemma-4-26B-A4B-it-qat-MTP-GGUF", sizeGB: 0.3 }, name: "Gemma 4 26B-A4B", lab: "Gemma", developer: "Google", country: "US", family: "Gemma",
    paramsB: 25.2, activeB: 3.8, denseShare: 0.08, score: 16.7, license: "Gemma terms", vision: true,
    description: "Google's MoE, trained to hold up at 4-bit.",
    quants: [q("unsloth", "gemma-4-26B-A4B-it-qat-GGUF", "UD-Q4_K_XL", 14.2)],
  },
  {
    id: "lfm2.5-24b-a2b", contextK: 128, kvPerK: 0.02, name: "LFM2.5 24B-A2B", lab: "Liquid AI", developer: "Liquid AI", country: "US", family: "LFM",
    paramsB: 24, activeB: 2.45, denseShare: 0.06, score: 15.6, license: "LFM Open License", vision: false,
    description: "Liquid's MoE. 2B active and very fast.",
    quants: [
      q("LiquidAI", "LFM2.5-24B-A2B-GGUF", "Q4_K_M", 14.5),
      q("LiquidAI", "LFM2.5-24B-A2B-GGUF", "Q6_K", 19.7),
      q("LiquidAI", "LFM2.5-24B-A2B-GGUF", "Q8_0", 25.5),
    ],
  },
  {
    id: "lfm3-1.7b", contextK: 128, kvPerK: 0.012, name: "LFM3 1.7B", lab: "Liquid AI", developer: "Liquid AI", country: "US", family: "LFM",
    paramsB: 1.7, activeB: null, denseShare: 1, score: 14.0, license: "LFM Open License", vision: false,
    description: "Small and fast. Runs on anything.",
    quants: [q("LiquidAI", "LFM3-1.7B-GGUF", "QAD-Q4_0", 1.0), q("LiquidAI", "LFM3-1.7B-GGUF", "Q8_0", 1.8)],
  },
  {
    id: "qwen3.5-9b", contextK: 256, kvPerK: 0.032, drafter: { method: "DFlash", repo: "magnitudedev/Qwen3.5-9B-DFlash-GGUF", sizeGB: 1.4 }, name: "Qwen3.5 9B", lab: "Qwen", developer: "Alibaba", country: "China", family: "Qwen",
    paramsB: 9, activeB: null, denseShare: 1, score: 11.2, license: "Apache 2.0", vision: true,
    description: "Small dense Qwen with vision.",
    quants: [
      q("unsloth", "Qwen3.5-9B-MTP-GGUF", "UD-Q4_K_XL", 6.1),
      q("unsloth", "Qwen3.5-9B-MTP-GGUF", "UD-Q6_K_XL", 9.0),
      q("unsloth", "Qwen3.5-9B-MTP-GGUF", "Q8_0", 9.8),
    ],
  },
  {
    id: "deepseek-v4.1-flash", contextK: 1024, kvPerK: 0.006, drafter: { method: "DSpark", repo: "magnitudedev/DeepSeek-V4.1-Flash-1017-GGUF", sizeGB: 7.9 }, name: "DeepSeek V4.1 Flash", lab: "DeepSeek", developer: "DeepSeek", country: "China", family: "DeepSeek",
    paramsB: 552, activeB: 16, denseShare: 0.06, sideTable: { gb: 196.6, paramsB: 196 }, score: null, license: "MIT", vision: true, comingSoon: true,
    description: "552B MoE with a 1M-token context. Streams on a 64 GB Mac; support is still being validated.",
    quants: [q("magnitudedev", "DeepSeek-V4.1-Flash-1017-GGUF", "MXFP4_MOE", 501.4)],
  },
]

export interface Family {
  readonly name: string
  readonly developer: string
  readonly country: Country
  readonly status: "supported" | "comingSoon"
}

export const families: readonly Family[] = [
  { name: "Qwen", developer: "Alibaba", country: "China", status: "supported" },
  { name: "Gemma", developer: "Google", country: "US", status: "supported" },
  { name: "LFM", developer: "Liquid AI", country: "US", status: "supported" },
  { name: "Nemotron", developer: "NVIDIA", country: "US", status: "supported" },
  { name: "MiniCPM", developer: "OpenBMB", country: "China", status: "supported" },
  { name: "Muse", developer: "Meta", country: "US", status: "supported" },
  { name: "DeepSeek", developer: "DeepSeek", country: "China", status: "comingSoon" },
]

/** A Hugging Face repo as search shows it, grouped under its base model. */
export interface Repo {
  readonly repo: string
  readonly base: string
  readonly downloads: string
  readonly architecture: string
  readonly unsupportedFamily?: string
  readonly note?: string
}

/** A fine-tune on Hugging Face of a model Magnitude runs: same architecture, so it runs too, untested. */
export interface FineTune {
  readonly repo: string
  readonly name: string
  readonly baseId: string
  readonly downloads: string
}

export const fineTunes: readonly FineTune[] = [
  { repo: "orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF", name: "Qwen3.8 Flash Next Uncensored", baseId: "qwen3.8-flash-next", downloads: "864K" },
  { repo: "huihui-ai/Qwen3.6-35B-A3B-abliterated-MTP-GGUF", name: "Qwen3.6 35B-A3B Abliterated", baseId: "qwen3.6-35b-a3b", downloads: "4.9M" },
  { repo: "ISTA-DASLab/Qwen3.8-Flash-Next-Coder-GSQ-RCO-GGUF", name: "Qwen3.8 Flash Next Coder", baseId: "qwen3.8-flash-next", downloads: "756K" },
]

/** Fine-tunes as models: the base model's shape and lab, with no tested quants. */
export const fineTuneModels: readonly Model[] = fineTunes.map(fineTune => {
  const base = pool.find(model => model.id === fineTune.baseId)!
  const publisher = fineTune.repo.split("/")[0]!
  return { ...base, id: fineTune.repo, name: fineTune.name, score: null, fineTuneOf: base.id, quants: [], description: `${publisher}’s fine-tune of ${base.name}. It runs because it’s built on the same architecture.` }
})

export const searchIndex: readonly Repo[] = [
  { repo: "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", base: "qwen3.8-flash-next", downloads: "6.8M", architecture: "qwen4exp" },
  { repo: "unsloth/Qwen3.8-Flash-Next-GGUF", base: "qwen3.8-flash-next", downloads: "1.7M", architecture: "qwen4exp" },
  { repo: "AtomicChat/Qwen3.8-Flash-Next-GGUF", base: "qwen3.8-flash-next", downloads: "467K", architecture: "qwen4exp" },
  { repo: "agentionai/Qwen3.8-Flash-Next-Gyro-GGUF", base: "qwen3.8-flash-next", downloads: "1.2M", architecture: "qwen4exp", note: "Uses weight formats Magnitude can't run" },
  { repo: "unsloth/Qwen3.6-35B-A3B-MTP-GGUF", base: "qwen3.6-35b-a3b", downloads: "2.1M", architecture: "qwen35moe" },
  { repo: "unsloth/Qwen3.6-35B-A3B-GGUF", base: "qwen3.6-35b-a3b", downloads: "1.6M", architecture: "qwen35moe" },
  { repo: "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF", base: "qwen3.6-35b-a3b", downloads: "235K", architecture: "qwen35moe" },
  { repo: "mudler/Qwen3.6-35B-A3B-APEX-GGUF", base: "qwen3.6-35b-a3b", downloads: "72K", architecture: "qwen35moe" },
  { repo: "unsloth/Qwen3.8-27B-GGUF", base: "qwen3.8-27b", downloads: "910K", architecture: "qwen35" },
  { repo: "unsloth/Qwen3.5-9B-MTP-GGUF", base: "qwen3.5-9b", downloads: "640K", architecture: "qwen35" },
  { repo: "unsloth/gemma-4-26B-A4B-it-qat-GGUF", base: "gemma-4-26b-a4b", downloads: "2.0M", architecture: "gemma4" },
  { repo: "ggml-org/gemma-4-26B-A4B-it-GGUF", base: "gemma-4-26b-a4b", downloads: "343K", architecture: "gemma4" },
  { repo: "unsloth/gemma-4-31B-it-qat-GGUF", base: "gemma-4-31b", downloads: "405K", architecture: "gemma4" },
  { repo: "LiquidAI/LFM2.5-24B-A2B-GGUF", base: "lfm2.5-24b-a2b", downloads: "120K", architecture: "lfm2moe" },
  { repo: "LiquidAI/LFM3-1.7B-GGUF", base: "lfm3-1.7b", downloads: "380K", architecture: "lfm3" },
  { repo: "unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF", base: "nemotron-3.5-super-vl", downloads: "95K", architecture: "nemotron_h_moe" },
  { repo: "magnitudedev/DeepSeek-V4.1-Flash-1017-GGUF", base: "deepseek-v4.1-flash", downloads: "41K", architecture: "deepseek41" },
  { repo: "ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF", base: "deepseek-v4.1-flash", downloads: "3.3M", architecture: "deepseek41" },
  ...fineTunes.map(fineTune => ({ repo: fineTune.repo, base: fineTune.repo, downloads: fineTune.downloads, architecture: "qwen35moe" })),
  { repo: "unsloth/Qwen3-30B-A3B-GGUF", base: "Qwen3 30B-A3B", downloads: "1.1M", architecture: "qwen3moe", unsupportedFamily: "Qwen 3 (2025)" },
  { repo: "unsloth/Qwen2.5-Coder-32B-Instruct-GGUF", base: "Qwen2.5 Coder 32B", downloads: "520K", architecture: "qwen2", unsupportedFamily: "Qwen 2.5" },
  { repo: "ggml-org/gpt-oss-120b-GGUF", base: "gpt-oss 120B", downloads: "880K", architecture: "gpt-oss", unsupportedFamily: "gpt-oss" },
  { repo: "unsloth/GLM-5.3-Flash-GGUF", base: "GLM 5.3 Flash", downloads: "818K", architecture: "glm5", unsupportedFamily: "GLM 5" },
  { repo: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", base: "Llama 3.1 8B", downloads: "2.4M", architecture: "llama", unsupportedFamily: "Llama" },
]

/** A model file found in another app's folder: Hugging Face cache, LM Studio, Ollama, llama.cpp, a folder. */
export interface FoundModel {
  readonly source: "Hugging Face cache" | "LM Studio" | "Ollama" | "llama.cpp" | "Downloads folder"
  readonly fileName: string
  readonly match: "repo" | "model" | "none" | "unsupported"
  readonly modelId?: string
  readonly quant?: Quant
  readonly unsupportedFamily?: string
}

/** How a found file was matched: to a Hugging Face repo by its hash, to a model by its tensor shapes, or only to an architecture. */
export const foundModels: readonly FoundModel[] = [
  { source: "Hugging Face cache", fileName: "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", match: "repo", modelId: "qwen3.6-35b-a3b", quant: { publisher: "unsloth", repo: "unsloth/Qwen3.6-35B-A3B-MTP-GGUF", name: "UD-Q4_K_XL", sizeGB: 22.9 } },
  { source: "LM Studio", fileName: "gemma-4-26B-A4B-it-Q4_K_M.gguf", match: "repo", modelId: "gemma-4-26b-a4b", quant: { publisher: "lmstudio-community", repo: "lmstudio-community/gemma-4-26B-A4B-it-GGUF", name: "Q4_K_M", sizeGB: 15.6 } },
  { source: "llama.cpp", fileName: "qwen3.8-27b-q5_k_m.gguf", match: "model", modelId: "qwen3.8-27b", quant: { publisher: "", repo: "", name: "Q5_K_M", sizeGB: 19.2 } },
  { source: "Downloads folder", fileName: "coder-merge-v2.Q4_K_M.gguf", match: "none", modelId: "qwen3.6-35b-a3b", quant: { publisher: "", repo: "", name: "Q4_K_M", sizeGB: 21.2 } },
  { source: "Ollama", fileName: "qwen3:30b-a3b", match: "unsupported", unsupportedFamily: "Qwen 3 (2025)" },
  { source: "llama.cpp", fileName: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", match: "unsupported", unsupportedFamily: "Llama" },
]
