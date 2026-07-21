import type {
  CanonicalArtifactOverlay,
  CanonicalModelCatalogOverlay,
  CanonicalModelOverlay,
  CatalogBenchmarkEvidence,
  CatalogLicenseReview,
  CatalogQuantBitsClass,
  CatalogQuantizationEvidence,
  LocalModelCatalogEntry,
} from "./types"

const QUANT_STUDY = "https://arxiv.org/abs/2606.19558"
const APACHE: CatalogLicenseReview = {
  expectedId: "apache-2.0",
  name: "Apache License 2.0",
  url: "https://www.apache.org/licenses/LICENSE-2.0",
  acknowledgementRequired: false,
}
const GEMMA: CatalogLicenseReview = {
  expectedId: "apache-2.0",
  name: "Gemma terms",
  url: "https://ai.google.dev/gemma/docs/gemma_4_license",
  acknowledgementRequired: false,
}
const MIT: CatalogLicenseReview = {
  expectedId: "mit",
  name: "MIT License",
  url: "https://opensource.org/license/mit",
  acknowledgementRequired: false,
}
const NEMOTRON: CatalogLicenseReview = {
  expectedId: "nvidia-nemotron-open-model-license",
  name: "NVIDIA Nemotron Open Model License",
  url: "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/",
  acknowledgementRequired: false,
}
const OPEN_MDW: CatalogLicenseReview = {
  expectedId: "openmdw-1.1",
  name: "OpenMDW License 1.1",
  url: "https://openmdw.ai/license/1-1/",
  acknowledgementRequired: false,
}

const fidelityFor = (
  format: string,
  overrides: Partial<CatalogQuantizationEvidence> = {},
): CatalogQuantizationEvidence => {
  const bitsClass: CatalogQuantBitsClass = format.includes("Q8") ? "q8"
    : format.includes("Q6") ? "q6"
    : format.includes("Q5") ? "q5"
    : format.includes("Q4") ? "q4"
    : format.includes("MXFP4") ? "mxfp4"
    : "other"
  const fidelityRank = bitsClass === "q8" ? 80 : bitsClass === "q6" ? 60 : bitsClass === "q5" ? 50 : 40
  const fidelityLabel = bitsClass === "q8" ? "Near-original fidelity with the least quality loss"
    : bitsClass === "q6" ? "Very high fidelity with minimal quality loss"
    : bitsClass === "q5" ? "High fidelity with only minor quality loss"
    : "Good fidelity with some possible quality loss"
  return {
    format,
    bitsClass,
    quantAwareCheckpoint: false,
    fidelityRank,
    fidelityLabel,
    evidenceScope: "cross_model_quant_tier",
    summary: "Cross-model quantization guidance; no exact-artifact quality measurement is available.",
    sourceUrl: QUANT_STUDY,
    ...overrides,
  }
}

const artifact = (
  modelId: string,
  repository: string,
  quantization: CatalogQuantizationEvidence,
): CanonicalArtifactOverlay => ({
  id: `${modelId}:${quantization.format}`,
  repository,
  filenameIncludes: quantization.format,
  quantization,
})

const quantArtifacts = (
  modelId: string,
  repository: string,
  formats: readonly string[],
  overrides: Readonly<Record<string, Partial<CatalogQuantizationEvidence>>> = {},
): readonly CanonicalArtifactOverlay[] => formats.map((format) =>
  artifact(modelId, repository, fidelityFor(format, overrides[format])))

const benchmark = (
  benchmarkId: string,
  label: string,
  score: number,
  methodologyId: string,
  mode: string,
  sourceUrl: string,
  notes = "Publisher-reported checkpoint result; not an exact GGUF measurement.",
): CatalogBenchmarkEvidence => ({
  benchmarkId,
  label,
  score,
  unit: "percent",
  higherIsBetter: true,
  methodologyId,
  mode,
  evidenceScope: "publisher_checkpoint",
  sourceUrl,
  notes,
})

const source = (repository: string): string => `https://huggingface.co/${repository}`
const qwenFormats = ["UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "UD-Q8_K_XL"] as const

const qwenModels: readonly CanonicalModelOverlay[] = [
  {
    id: "qwen3.5-4b", family: "qwen3.5", displayName: "Qwen3.5 4B", developer: "Qwen",
    description: "Compact dense model for machines where responsiveness and footprint matter most.",
    modelRepository: "Qwen/Qwen3.5-4B", productContextTokens: [100_000, 200_000],
    performance: { summary: "The smallest Qwen catalog tier, optimized for low footprint.", benchmarks: [benchmark("livecodebench-v6", "LiveCodeBench v6", 55.8, "qwen3.5-small-2026", "thinking", source("Qwen/Qwen3.5-4B"))] },
    licenseReview: APACHE, legacyQualityRank: 10,
    artifacts: quantArtifacts("qwen3.5-4b", "unsloth/Qwen3.5-4B-GGUF", qwenFormats),
  },
  {
    id: "qwen3.5-9b", family: "qwen3.5", displayName: "Qwen3.5 9B", developer: "Qwen",
    description: "Small dense model that trades some speed and memory for a substantial quality gain over 4B.",
    modelRepository: "Qwen/Qwen3.5-9B", productContextTokens: [100_000, 200_000],
    performance: { summary: "Higher coding capability than the 4B checkpoint while remaining practical on common consumer machines.", benchmarks: [benchmark("livecodebench-v6", "LiveCodeBench v6", 65.6, "qwen3.5-small-2026", "thinking", source("Qwen/Qwen3.5-9B"))] },
    licenseReview: APACHE, legacyQualityRank: 20,
    artifacts: quantArtifacts("qwen3.5-9b", "unsloth/Qwen3.5-9B-GGUF", qwenFormats),
  },
  {
    id: "qwen3.6-27b", family: "qwen3.6", displayName: "Qwen3.6 27B", developer: "Qwen",
    description: "Large dense coding model with strong publisher-reported agent and coding results.",
    modelRepository: "Qwen/Qwen3.6-27B", productContextTokens: [100_000, 200_000],
    performance: { summary: "The strongest dense Qwen checkpoint in the initial consumer catalog.", benchmarks: [benchmark("livecodebench-v6", "LiveCodeBench v6", 83.9, "qwen3.6-medium-2026", "thinking", source("Qwen/Qwen3.6-27B"))] },
    licenseReview: APACHE, legacyQualityRank: 45,
    artifacts: quantArtifacts("qwen3.6-27b", "unsloth/Qwen3.6-27B-GGUF", qwenFormats),
  },
  {
    id: "qwen3.6-35b-a3b", family: "qwen3.6", displayName: "Qwen3.6 35B-A3B", developer: "Qwen",
    description: "Efficient MoE coding model with a large knowledge footprint and low active compute per token.",
    modelRepository: "Qwen/Qwen3.6-35B-A3B", productContextTokens: [100_000, 200_000],
    performance: { summary: "MoE execution reduces generation work relative to a similarly sized dense model.", benchmarks: [benchmark("livecodebench-v6", "LiveCodeBench v6", 80.4, "qwen3.6-medium-2026", "thinking", source("Qwen/Qwen3.6-35B-A3B"))] },
    licenseReview: APACHE, legacyQualityRank: 50,
    artifacts: quantArtifacts("qwen3.6-35b-a3b", "unsloth/Qwen3.6-35B-A3B-GGUF", qwenFormats, {
      "UD-Q4_K_XL": { evidenceScope: "exact_artifact", summary: "Model-specific testing places this artifact in the study's near-baseline region (KLD 0.0135).", sourceUrl: QUANT_STUDY },
      "UD-Q5_K_XL": { evidenceScope: "exact_artifact", summary: "Model-specific testing places this artifact in the study's near-baseline region (KLD 0.0082).", sourceUrl: QUANT_STUDY },
    }),
  },
]

const gemmaSource = source("google/gemma-4-12B")
const qatFidelity = fidelityFor("UD-Q4_K_XL", {
  quantAwareCheckpoint: true,
  fidelityRank: 58,
  fidelityLabel: "Near-original fidelity from quantization-aware training",
  evidenceScope: "checkpoint_quantization",
  summary: "The checkpoint was trained for low-precision deployment; this is checkpoint-level evidence.",
  sourceUrl: gemmaSource,
})

const gemma = (
  id: string,
  displayName: string,
  description: string,
  modelRepository: string,
  artifactRepository: string,
  score: number,
  rank: number,
  productContextTokens: readonly (100_000 | 200_000)[],
): CanonicalModelOverlay => ({
  id, family: "gemma-4", displayName, developer: "Google DeepMind", description,
  modelRepository, productContextTokens,
  performance: { summary: "Publisher benchmark from the common Gemma 4 family evaluation.", benchmarks: [benchmark("livecodebench-v6", "LiveCodeBench v6", score, "gemma4-family-2026", "thinking", gemmaSource)] },
  licenseReview: GEMMA, legacyQualityRank: rank,
  artifacts: [artifact(id, artifactRepository, qatFidelity)],
})

const gemmaModels: readonly CanonicalModelOverlay[] = [
  gemma("gemma-4-e2b-it-qat", "Gemma 4 E2B", "Very small dense model optimized for on-device use.", "google/gemma-4-E2B-it-qat-q4_0-unquantized", "unsloth/gemma-4-E2B-it-qat-GGUF", 44, 8, [100_000]),
  gemma("gemma-4-12b-it-qat", "Gemma 4 12B", "Mid-size dense model with native tool use, reasoning, vision, and audio capabilities.", "google/gemma-4-12B-it-qat-q4_0-unquantized", "unsloth/gemma-4-12B-it-qat-GGUF", 72, 30, [100_000, 200_000]),
  gemma("gemma-4-26b-a4b-it-qat", "Gemma 4 26B-A4B", "Mid-size MoE model balancing a substantial weight footprint with low active compute.", "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized", "unsloth/gemma-4-26B-A4B-it-qat-GGUF", 77.1, 34, [100_000, 200_000]),
  gemma("gemma-4-31b-it-qat", "Gemma 4 31B", "Large dense Gemma model with the strongest publisher-reported coding score in its family.", "google/gemma-4-31B-it-qat-q4_0-unquantized", "unsloth/gemma-4-31B-it-qat-GGUF", 80, 36, [100_000, 200_000]),
]

const largeModel = (input: {
  id: string
  family: string
  displayName: string
  developer: string
  description: string
  modelRepository: string
  artifactRepository: string
  format: string
  score: number
  benchmarkId: string
  benchmarkLabel: string
  methodologyId: string
  mode: string
  rank: number
  licenseReview: CatalogLicenseReview
  fidelity?: Partial<CatalogQuantizationEvidence>
}): CanonicalModelOverlay => ({
  id: input.id, family: input.family, displayName: input.displayName, developer: input.developer,
  description: input.description, modelRepository: input.modelRepository, productContextTokens: [100_000, 200_000],
  performance: {
    summary: input.description,
    benchmarks: [benchmark(input.benchmarkId, input.benchmarkLabel, input.score, input.methodologyId, input.mode, source(input.modelRepository))],
  },
  licenseReview: input.licenseReview,
  legacyQualityRank: input.rank,
  artifacts: [artifact(input.id, input.artifactRepository, fidelityFor(input.format, input.fidelity))],
})

const largeModels: readonly CanonicalModelOverlay[] = [
  largeModel({ id: "qwen3.5-122b-a10b", family: "qwen3.5", displayName: "Qwen3.5 122B-A10B", developer: "Qwen", description: "Workstation-class MoE model with a large knowledge footprint and moderate active compute.", modelRepository: "Qwen/Qwen3.5-122B-A10B", artifactRepository: "unsloth/Qwen3.5-122B-A10B-GGUF", format: "UD-Q4_K_XL", score: 78.9, benchmarkId: "livecodebench-v6", benchmarkLabel: "LiveCodeBench v6", methodologyId: "qwen3.5-large-2026", mode: "thinking", rank: 64, licenseReview: APACHE }),
  largeModel({ id: "nemotron-3-super-120b-a12b", family: "nemotron-3", displayName: "NVIDIA Nemotron 3 Super 120B-A12B", developer: "NVIDIA", description: "Workstation-class hybrid MoE model designed for agentic workflows and efficient low-precision execution.", modelRepository: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", artifactRepository: "unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF", format: "MXFP4_MOE", score: 78.44, benchmarkId: "livecodebench-v6-2024-08-2025-05", benchmarkLabel: "LiveCodeBench v6", methodologyId: "nemotron3-super-2026", mode: "reasoning", rank: 65, licenseReview: NEMOTRON, fidelity: { bitsClass: "mxfp4", quantAwareCheckpoint: true, fidelityRank: 58, fidelityLabel: "Near-original fidelity in benchmark comparisons", evidenceScope: "checkpoint_quantization", summary: "NVIDIA reports near-BF16 benchmark results for its low-precision checkpoint; the GGUF conversion itself is not directly benchmarked.", sourceUrl: source("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16") } }),
  largeModel({ id: "deepseek-v4-flash", family: "deepseek-v4", displayName: "DeepSeek V4 Flash 284B-A13B", developer: "DeepSeek", description: "Frontier MoE model with a very large weight footprint but low active compute relative to its size.", modelRepository: "deepseek-ai/DeepSeek-V4-Flash", artifactRepository: "unsloth/DeepSeek-V4-Flash-GGUF", format: "UD-Q8_K_XL", score: 91.6, benchmarkId: "livecodebench", benchmarkLabel: "LiveCodeBench", methodologyId: "deepseek-v4-2026", mode: "maximum reasoning", rank: 72, licenseReview: MIT }),
  largeModel({ id: "nemotron-3-ultra-550b-a55b", family: "nemotron-3", displayName: "NVIDIA Nemotron 3 Ultra 550B-A55B", developer: "NVIDIA", description: "Frontier workstation/server MoE model intended for exceptionally high-memory systems.", modelRepository: "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", artifactRepository: "unsloth/NVIDIA-Nemotron-3-Ultra-550B-A55B-GGUF", format: "MXFP4_MOE", score: 89, benchmarkId: "livecodebench-v6", benchmarkLabel: "LiveCodeBench v6", methodologyId: "nemotron3-ultra-2026", mode: "reasoning", rank: 76, licenseReview: OPEN_MDW, fidelity: { bitsClass: "mxfp4", quantAwareCheckpoint: true, fidelityRank: 58, fidelityLabel: "Near-original fidelity in benchmark comparisons", evidenceScope: "checkpoint_quantization", summary: "NVIDIA reports near-BF16 benchmark results for its low-precision checkpoint; the GGUF conversion itself is not directly benchmarked.", sourceUrl: source("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16") } }),
  largeModel({ id: "glm-5.2", family: "glm-5.2", displayName: "GLM 5.2 753B-A40B", developer: "Z.ai", description: "Largest catalog tier, intended only for exceptionally high-memory systems.", modelRepository: "zai-org/GLM-5.2", artifactRepository: "unsloth/GLM-5.2-GGUF", format: "UD-Q4_K_XL", score: 81, benchmarkId: "terminal-bench-2.1-terminus-2", benchmarkLabel: "Terminal Bench 2.1", methodologyId: "glm5.2-2026", mode: "reasoning", rank: 80, licenseReview: MIT, fidelity: { fidelityRank: 70, fidelityLabel: "Near-original fidelity in quantization tests", evidenceScope: "exact_artifact", summary: "Unsloth reports approximately 97.5% top-token agreement against Q8_0 for this artifact family; this is not coding accuracy.", sourceUrl: "https://huggingface.co/unsloth/GLM-5.2-GGUF/discussions/3" } }),
]

export const LOCAL_MODEL_CATALOG_OVERLAY: CanonicalModelCatalogOverlay = {
  schemaVersion: 2,
  catalogVersion: "2026-07-20",
  reviewedAt: "2026-07-20",
  models: [...qwenModels, ...gemmaModels, ...largeModels],
}

const REPOSITORY = /^[^/\s]+\/[^/\s]+$/
const DATE = /^\d{4}-\d{2}-\d{2}$/
const isHttpsUrl = (value: string): boolean => {
  try { return new URL(value).protocol === "https:" } catch { return false }
}

/** Offline validation covers only checked-in Magnitude metadata. */
export const validateCanonicalModelCatalog = (catalog: CanonicalModelCatalogOverlay): readonly string[] => {
  const issues: string[] = []
  if (catalog.schemaVersion !== 2) issues.push(`unsupported catalog schema version ${catalog.schemaVersion}`)
  if (!DATE.test(catalog.catalogVersion)) issues.push("catalog version must be an ISO date")
  if (!DATE.test(catalog.reviewedAt)) issues.push("catalog review date must be an ISO date")
  const modelIds = new Set<string>()
  const artifactIds = new Set<string>()
  for (const model of catalog.models) {
    if (modelIds.has(model.id)) issues.push(`duplicate model id ${model.id}`)
    modelIds.add(model.id)
    if (!REPOSITORY.test(model.modelRepository)) issues.push(`${model.id} has an invalid model repository`)
    if (model.productContextTokens.length === 0) issues.push(`${model.id} has no product contexts`)
    if (new Set(model.productContextTokens).size !== model.productContextTokens.length) issues.push(`${model.id} has duplicate product contexts`)
    if (model.performance.benchmarks.length === 0) issues.push(`${model.id} has no performance evidence`)
    for (const evidence of model.performance.benchmarks) {
      if (!evidence.benchmarkId || !evidence.methodologyId || !evidence.mode) issues.push(`${model.id} has incomplete benchmark evidence`)
      if (!Number.isFinite(evidence.score) || !isHttpsUrl(evidence.sourceUrl)) issues.push(`${model.id} has invalid benchmark evidence`)
    }
    if (!model.licenseReview.expectedId || !isHttpsUrl(model.licenseReview.url)) issues.push(`${model.id} has an incomplete license review`)
    if (model.artifacts.length === 0) issues.push(`${model.id} has no artifact selectors`)
    const formats = new Set<string>()
    for (const candidate of model.artifacts) {
      if (artifactIds.has(candidate.id)) issues.push(`duplicate artifact id ${candidate.id}`)
      artifactIds.add(candidate.id)
      if (!REPOSITORY.test(candidate.repository)) issues.push(`${candidate.id} has an invalid artifact repository`)
      if (!candidate.filenameIncludes || candidate.filenameIncludes.toLowerCase().endsWith(".gguf")) issues.push(`${candidate.id} selector must be a filename fragment, not a pinned path`)
      if (candidate.id !== `${model.id}:${candidate.quantization.format}`) issues.push(`${candidate.id} does not match its model and quantization`)
      if (formats.has(candidate.quantization.format)) issues.push(`${model.id} contains duplicate quantization ${candidate.quantization.format}`)
      formats.add(candidate.quantization.format)
      if (!candidate.quantization.summary || !isHttpsUrl(candidate.quantization.sourceUrl)) issues.push(`${candidate.id} has incomplete fidelity evidence`)
    }
  }
  return issues
}

const validationIssues = validateCanonicalModelCatalog(LOCAL_MODEL_CATALOG_OVERLAY)
if (validationIssues.length > 0) throw new Error(`Invalid local model catalog overlay:\n${validationIssues.join("\n")}`)

export interface ResolvedHubSnapshot {
  readonly repository: string
  readonly commit: string
  readonly license?: string | null
  readonly license_url?: string | null
  readonly gguf_files: readonly { readonly path: string; readonly size_bytes?: number }[]
}

const firstShard = /-00001-of-\d{5}\.gguf$/i
const laterShard = /-\d{5}-of-\d{5}\.gguf$/i

const selectedWeightBytes = (
  primary: string,
  files: ResolvedHubSnapshot["gguf_files"],
): number | undefined => {
  const split = primary.match(/^(.*)-00001-of-(\d{5})\.gguf$/i)
  const shardCount = split ? Number(split[2]) : 1
  if (!Number.isSafeInteger(shardCount) || shardCount < 1 || shardCount > 256) return undefined
  const selected = split
    ? Array.from({ length: shardCount }, (_, index) =>
      `${split[1]}-${String(index + 1).padStart(5, "0")}-of-${split[2]}.gguf`)
        .map((path) => files.find((file) => file.path === path))
    : [files.find((file) => file.path === primary)]
  if (selected.some((file) => !file || !Number.isSafeInteger(file.size_bytes) || file.size_bytes! <= 0)) return undefined
  return selected.reduce((total, file) => total + file!.size_bytes!, 0)
}

/** Resolve a stable quant selector against one immutable live Hub snapshot. */
export const resolveCatalogArtifact = (
  model: CanonicalModelOverlay,
  candidate: CanonicalArtifactOverlay,
  snapshot: ResolvedHubSnapshot,
): LocalModelCatalogEntry | undefined => {
  const selector = candidate.filenameIncludes.toLowerCase()
  const matches = snapshot.gguf_files.filter(({ path }) => {
    const lower = path.toLowerCase()
    const basename = lower.slice(lower.lastIndexOf("/") + 1)
    return lower.includes(selector)
      && !basename.startsWith("mmproj-")
      && !basename.includes("imatrix")
      && (!laterShard.test(lower) || firstShard.test(lower))
  })
  if (matches.length !== 1) return undefined
  const primaryGguf = matches[0]!.path
  const publishedWeightBytes = selectedWeightBytes(primaryGguf, snapshot.gguf_files)
  if (publishedWeightBytes === undefined) return undefined
  return {
    id: candidate.id,
    modelId: model.id,
    family: model.family,
    displayName: model.displayName,
    repo: snapshot.repository,
    revision: snapshot.commit,
    primaryGguf,
    publishedWeightBytes,
    additionalComponents: [],
    supportedContextTokens: model.productContextTokens,
    quantTag: candidate.quantization.format,
    quantization: {
      quantAwareCheckpoint: candidate.quantization.quantAwareCheckpoint,
      fidelityRank: candidate.quantization.fidelityRank,
      fidelityLabel: candidate.quantization.fidelityLabel,
      fidelityEvidence: candidate.quantization.summary,
      fidelitySourceUrl: candidate.quantization.sourceUrl,
    },
    license: {
      id: !snapshot.license || snapshot.license === "other"
        ? model.licenseReview.expectedId
        : snapshot.license,
      url: snapshot.license && snapshot.license !== "other" && snapshot.license_url
        ? snapshot.license_url
        : model.licenseReview.url,
      acknowledgementRequired: model.licenseReview.acknowledgementRequired,
    },
    modelQualityRank: model.legacyQualityRank,
  }
}

export const catalogSourcePageUrl = (model: LocalModelCatalogEntry): string => {
  const path = model.primaryGguf.split("/").map(encodeURIComponent).join("/")
  return `https://huggingface.co/${model.repo}/blob/${model.revision}/${path}`
}
