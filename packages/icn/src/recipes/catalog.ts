import { Option } from "effect"
import type {
  ModelRecipeArtifact,
  ModelRecipeRegistry,
  ModelRecipe,
  RecipeBenchmarkEvidence,
  RecipeLicenseReview,
  RecipeQuantBitsClass,
  RecipeQuantizationEvidence,
  ResolvedModelRecipe,
} from "./types"

const QUANT_STUDY = "https://arxiv.org/abs/2606.19558"
const TERMINAL_BENCH_SOURCE = "https://artificialanalysis.ai/evaluations/terminalbench-v2-1"
const TERMINAL_BENCH_ID = "terminal-bench-v2.1"
const TERMINAL_BENCH_METHODOLOGY = "terminal-bench-v2.1-percent-success"
const APACHE: RecipeLicenseReview = {
  expectedId: "apache-2.0",
  name: "Apache License 2.0",
  url: "https://www.apache.org/licenses/LICENSE-2.0",
  acknowledgementRequired: false,
}
const GEMMA: RecipeLicenseReview = {
  expectedId: "apache-2.0",
  name: "Gemma terms",
  url: "https://ai.google.dev/gemma/docs/gemma_4_license",
  acknowledgementRequired: false,
}
const MIT: RecipeLicenseReview = {
  expectedId: "mit",
  name: "MIT License",
  url: "https://opensource.org/license/mit",
  acknowledgementRequired: false,
}
const NEMOTRON: RecipeLicenseReview = {
  expectedId: "nvidia-nemotron-open-model-license",
  name: "NVIDIA Nemotron Open Model License",
  url: "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/",
  acknowledgementRequired: false,
}
const OPEN_MDW: RecipeLicenseReview = {
  expectedId: "openmdw-1.1",
  name: "OpenMDW License 1.1",
  url: "https://openmdw.ai/license/1-1/",
  acknowledgementRequired: false,
}

const fidelityFor = (
  format: string,
  overrides: Partial<RecipeQuantizationEvidence> = {},
): RecipeQuantizationEvidence => {
  const bitsClass: RecipeQuantBitsClass = format.includes("Q8") ? "q8"
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
  quantization: RecipeQuantizationEvidence,
): ModelRecipeArtifact => ({
  id: `${modelId}:${quantization.format}`,
  repository,
  filenameIncludes: quantization.format,
  quantization,
})

const quantArtifacts = (
  modelId: string,
  repository: string,
  formats: readonly string[],
  overrides: Readonly<Record<string, Partial<RecipeQuantizationEvidence>>> = {},
): readonly ModelRecipeArtifact[] => formats.map((format) =>
  artifact(modelId, repository, fidelityFor(format, overrides[format])))

const terminalBench = (
  score: number,
  provenance: RecipeBenchmarkEvidence["provenance"] = "measured_terminal_bench_2.1",
  basis = "Independently measured by Artificial Analysis using Terminus 2 on E2B; checkpoint-level evidence, not an exact GGUF quantization measurement.",
  sourceUrl = TERMINAL_BENCH_SOURCE,
): RecipeBenchmarkEvidence => ({
  benchmarkId: TERMINAL_BENCH_ID,
  label: "Terminal-Bench v2.1",
  score,
  unit: "percent",
  higherIsBetter: true,
  methodologyId: TERMINAL_BENCH_METHODOLOGY,
  mode: "agentic-coding",
  evidenceScope: "independent_checkpoint",
  provenance,
  sourceUrl,
  basis,
  notes: provenance === "measured_terminal_bench_2.1"
    ? "Independent checkpoint result; not an exact GGUF quantization measurement."
    : "Magnitude estimate used only where no measured Terminal-Bench v2.1 result is available.",
})

const source = (repository: string): string => `https://huggingface.co/${repository}`
const publisherTerminalBench = (
  score: number,
  repository: string,
): RecipeBenchmarkEvidence => ({
  ...terminalBench(
    score,
    "measured_terminal_bench_2.1",
    "Measured Terminal-Bench v2.1 result published by the checkpoint developer; checkpoint-level evidence, not an exact GGUF quantization measurement.",
    source(repository),
  ),
  evidenceScope: "publisher_checkpoint",
  notes: "Publisher-measured checkpoint result; not an exact GGUF quantization measurement.",
})
const qwenFormats = ["UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "UD-Q8_K_XL"] as const

const qwenModels: readonly ModelRecipe[] = [
  {
    id: "qwen3.5-4b", family: "qwen3.5", displayName: "Qwen3.5 4B", developer: "Qwen",
    description: "Compact dense model for machines where responsiveness and footprint matter most.",
    modelRepository: "Qwen/Qwen3.5-4B", productContextTokens: [100_000, 200_000],
    performance: { summary: "The smallest Qwen catalog tier, optimized for low footprint.", benchmarks: [terminalBench(25.8)] },
    licenseReview: APACHE,
    artifacts: quantArtifacts("qwen3.5-4b", "unsloth/Qwen3.5-4B-GGUF", qwenFormats),
  },
  {
    id: "qwen3.5-9b", family: "qwen3.5", displayName: "Qwen3.5 9B", developer: "Qwen",
    description: "Small dense model that trades some speed and memory for a substantial quality gain over 4B.",
    modelRepository: "Qwen/Qwen3.5-9B", productContextTokens: [100_000, 200_000],
    performance: { summary: "Higher coding capability than the 4B checkpoint while remaining practical on common consumer machines.", benchmarks: [terminalBench(29.2)] },
    licenseReview: APACHE,
    artifacts: quantArtifacts("qwen3.5-9b", "unsloth/Qwen3.5-9B-GGUF", qwenFormats),
  },
  {
    id: "qwen3.6-27b", family: "qwen3.6", displayName: "Qwen3.6 27B", developer: "Qwen",
    description: "Large dense coding model with strong publisher-reported agent and coding results.",
    modelRepository: "Qwen/Qwen3.6-27B", productContextTokens: [100_000, 200_000],
    performance: { summary: "The strongest dense Qwen checkpoint in the initial consumer catalog.", benchmarks: [terminalBench(60.7)] },
    licenseReview: APACHE,
    artifacts: quantArtifacts("qwen3.6-27b", "unsloth/Qwen3.6-27B-GGUF", qwenFormats),
  },
  {
    id: "qwen3.6-35b-a3b", family: "qwen3.6", displayName: "Qwen3.6 35B-A3B", developer: "Qwen",
    description: "Efficient MoE coding model with a large knowledge footprint and low active compute per token.",
    modelRepository: "Qwen/Qwen3.6-35B-A3B", productContextTokens: [100_000, 200_000],
    performance: { summary: "MoE execution reduces generation work relative to a similarly sized dense model.", benchmarks: [terminalBench(44.9)] },
    licenseReview: APACHE,
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
  capability: RecipeBenchmarkEvidence,
  productContextTokens: readonly (100_000 | 200_000)[],
): ModelRecipe => ({
  id, family: "gemma-4", displayName, developer: "Google DeepMind", description,
  modelRepository, productContextTokens,
  performance: { summary: "Terminal-Bench v2.1 capability evidence for this checkpoint.", benchmarks: [capability] },
  licenseReview: GEMMA,
  artifacts: [artifact(id, artifactRepository, qatFidelity)],
})

const gemmaModels: readonly ModelRecipe[] = [
  gemma("gemma-4-e2b-it-qat", "Gemma 4 E2B", "Very small dense model optimized for on-device use.", "google/gemma-4-E2B-it-qat-q4_0-unquantized", "unsloth/gemma-4-E2B-it-qat-GGUF", terminalBench(15, "estimated_terminal_bench_2.1", "Magnitude estimate based on relative Gemma family capability; no measured Terminal-Bench v2.1 result is available.", source("google/gemma-4-E2B-it-qat-q4_0-unquantized")), [100_000]),
  gemma("gemma-4-12b-it-qat", "Gemma 4 12B", "Mid-size dense model with native tool use, reasoning, vision, and audio capabilities.", "google/gemma-4-12B-it-qat-q4_0-unquantized", "unsloth/gemma-4-12B-it-qat-GGUF", terminalBench(21, "estimated_terminal_bench_2.1", "Magnitude estimate based on relative Gemma family capability; no measured Terminal-Bench v2.1 result is available.", source("google/gemma-4-12B-it-qat-q4_0-unquantized")), [100_000, 200_000]),
  gemma("gemma-4-26b-a4b-it-qat", "Gemma 4 26B-A4B", "Mid-size MoE model balancing a substantial weight footprint with low active compute.", "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized", "unsloth/gemma-4-26B-A4B-it-qat-GGUF", terminalBench(39.0), [100_000, 200_000]),
  gemma("gemma-4-31b-it-qat", "Gemma 4 31B", "Large dense Gemma model with the strongest measured coding score in its family.", "google/gemma-4-31B-it-qat-q4_0-unquantized", "unsloth/gemma-4-31B-it-qat-GGUF", terminalBench(43.4), [100_000, 200_000]),
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
  licenseReview: RecipeLicenseReview
  fidelity?: Partial<RecipeQuantizationEvidence>
}): ModelRecipe => ({
  id: input.id, family: input.family, displayName: input.displayName, developer: input.developer,
  description: input.description, modelRepository: input.modelRepository, productContextTokens: [100_000, 200_000],
  performance: {
    summary: input.description,
    benchmarks: [terminalBench(input.score)],
  },
  licenseReview: input.licenseReview,
  artifacts: [artifact(input.id, input.artifactRepository, fidelityFor(input.format, input.fidelity))],
})

const largeModels: readonly ModelRecipe[] = [
  {
    id: "laguna-s-2.1",
    family: "laguna",
    displayName: "Laguna S 2.1 118B-A8B",
    developer: "Poolside",
    description: "High-capability MoE model designed for agentic coding and long-horizon software work.",
    modelRepository: "poolside/Laguna-S-2.1",
    productContextTokens: [100_000, 200_000],
    performance: {
      summary: "Poolside's high-capability Laguna tier, with 8B active parameters per token.",
      benchmarks: [publisherTerminalBench(70.2, "poolside/Laguna-S-2.1")],
    },
    licenseReview: OPEN_MDW,
    artifacts: quantArtifacts(
      "laguna-s-2.1",
      "poolside/Laguna-S-2.1-GGUF",
      ["Q4_K_M", "Q8_0"],
    ),
  },
  largeModel({ id: "qwen3.5-122b-a10b", family: "qwen3.5", displayName: "Qwen3.5 122B-A10B", developer: "Qwen", description: "Workstation-class MoE model with a large knowledge footprint and moderate active compute.", modelRepository: "Qwen/Qwen3.5-122B-A10B", artifactRepository: "unsloth/Qwen3.5-122B-A10B-GGUF", format: "UD-Q4_K_XL", score: 47.6, licenseReview: APACHE }),
  largeModel({ id: "nemotron-3-super-120b-a12b", family: "nemotron-3", displayName: "NVIDIA Nemotron 3 Super 120B-A12B", developer: "NVIDIA", description: "Workstation-class hybrid MoE model designed for agentic workflows and efficient low-precision execution.", modelRepository: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", artifactRepository: "unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF", format: "MXFP4_MOE", score: 38.6, licenseReview: NEMOTRON, fidelity: { bitsClass: "mxfp4", quantAwareCheckpoint: true, fidelityRank: 58, fidelityLabel: "Near-original fidelity in benchmark comparisons", evidenceScope: "checkpoint_quantization", summary: "NVIDIA reports near-BF16 benchmark results for its low-precision checkpoint; the GGUF conversion itself is not directly benchmarked.", sourceUrl: source("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16") } }),
  largeModel({ id: "deepseek-v4-flash", family: "deepseek-v4", displayName: "DeepSeek V4 Flash 284B-A13B", developer: "DeepSeek", description: "Frontier MoE model with a very large weight footprint but low active compute relative to its size.", modelRepository: "deepseek-ai/DeepSeek-V4-Flash", artifactRepository: "unsloth/DeepSeek-V4-Flash-GGUF", format: "UD-Q8_K_XL", score: 61.8, licenseReview: MIT }),
  largeModel({ id: "nemotron-3-ultra-550b-a55b", family: "nemotron-3", displayName: "NVIDIA Nemotron 3 Ultra 550B-A55B", developer: "NVIDIA", description: "Frontier workstation/server MoE model intended for exceptionally high-memory systems.", modelRepository: "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", artifactRepository: "unsloth/NVIDIA-Nemotron-3-Ultra-550B-A55B-GGUF", format: "MXFP4_MOE", score: 53.9, licenseReview: OPEN_MDW, fidelity: { bitsClass: "mxfp4", quantAwareCheckpoint: true, fidelityRank: 58, fidelityLabel: "Near-original fidelity in benchmark comparisons", evidenceScope: "checkpoint_quantization", summary: "NVIDIA reports near-BF16 benchmark results for its low-precision checkpoint; the GGUF conversion itself is not directly benchmarked.", sourceUrl: source("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16") } }),
  largeModel({ id: "glm-5.2", family: "glm-5.2", displayName: "GLM 5.2 753B-A40B", developer: "Z.ai", description: "Largest catalog tier, intended only for exceptionally high-memory systems.", modelRepository: "zai-org/GLM-5.2", artifactRepository: "unsloth/GLM-5.2-GGUF", format: "UD-Q4_K_XL", score: 77.9, licenseReview: MIT, fidelity: { fidelityRank: 70, fidelityLabel: "Near-original fidelity in quantization tests", evidenceScope: "exact_artifact", summary: "Unsloth reports approximately 97.5% top-token agreement against Q8_0 for this artifact family; this is not coding accuracy.", sourceUrl: "https://huggingface.co/unsloth/GLM-5.2-GGUF/discussions/3" } }),
]

export const MODEL_RECIPE_REGISTRY: ModelRecipeRegistry = {
  reviewedAt: "2026-07-23",
  models: [...qwenModels, ...gemmaModels, ...largeModels],
}

const REPOSITORY = /^[^/\s]+\/[^/\s]+$/
const DATE = /^\d{4}-\d{2}-\d{2}$/
const isHttpsUrl = (value: string): boolean => {
  try { return new URL(value).protocol === "https:" } catch { return false }
}

/** Offline validation covers only checked-in Magnitude metadata. */
export const validateModelRecipeRegistry = (catalog: ModelRecipeRegistry): readonly string[] => {
  const issues: string[] = []
  if (!DATE.test(catalog.reviewedAt)) issues.push("catalog review date must be an ISO date")
  const modelIds = new Set<string>()
  const artifactIds = new Set<string>()
  for (const model of catalog.models) {
    if (modelIds.has(model.id)) issues.push(`duplicate model id ${model.id}`)
    modelIds.add(model.id)
    if (!REPOSITORY.test(model.modelRepository)) issues.push(`${model.id} has an invalid model repository`)
    if (model.productContextTokens.length === 0) issues.push(`${model.id} has no product contexts`)
    if (new Set(model.productContextTokens).size !== model.productContextTokens.length) issues.push(`${model.id} has duplicate product contexts`)
    const terminalBenchEvidence = model.performance.benchmarks.filter(({ benchmarkId }) =>
      benchmarkId === TERMINAL_BENCH_ID)
    if (terminalBenchEvidence.length !== 1) {
      issues.push(`${model.id} must have exactly one Terminal-Bench v2.1 capability score`)
    }
    for (const evidence of model.performance.benchmarks) {
      if (!evidence.benchmarkId || !evidence.methodologyId || !evidence.mode) issues.push(`${model.id} has incomplete benchmark evidence`)
      if (!Number.isFinite(evidence.score) || evidence.score < 0 || evidence.score > 100 || !isHttpsUrl(evidence.sourceUrl)) issues.push(`${model.id} has invalid benchmark evidence`)
      if (!evidence.basis.trim()) issues.push(`${model.id} has benchmark evidence without a stated basis`)
      if (evidence.benchmarkId === TERMINAL_BENCH_ID
        && evidence.methodologyId !== TERMINAL_BENCH_METHODOLOGY) {
        issues.push(`${model.id} has inconsistent Terminal-Bench v2.1 methodology`)
      }
      if (evidence.provenance === "measured_terminal_bench_2.1"
        && evidence.evidenceScope === "independent_checkpoint"
        && evidence.sourceUrl !== TERMINAL_BENCH_SOURCE) {
        issues.push(`${model.id} has measured Terminal-Bench evidence without the canonical source`)
      }
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

const validationIssues = validateModelRecipeRegistry(MODEL_RECIPE_REGISTRY)
if (validationIssues.length > 0) throw new Error(`Invalid local model catalog overlay:\n${validationIssues.join("\n")}`)

export interface ResolvedHubSnapshot {
  readonly repository: string
  readonly commit: string
  readonly license: Option.Option<string>
  readonly licenseUrl: Option.Option<string>
  readonly ggufFiles: readonly {
    readonly path: string
    readonly sizeBytes: Option.Option<number>
  }[]
}

const firstShard = /-00001-of-\d{5}\.gguf$/i
const laterShard = /-\d{5}-of-\d{5}\.gguf$/i

const selectedWeightBytes = (
  primary: string,
  files: ResolvedHubSnapshot["ggufFiles"],
): Option.Option<number> => {
  const split = Option.fromNullable(primary.match(/^(.*)-00001-of-(\d{5})\.gguf$/i))
  const shardCount = Option.match(split, {
    onNone: () => 1,
    onSome: (match) => Option.match(Option.fromNullable(match.at(2)), {
      onNone: () => Number.NaN,
      onSome: Number,
    }),
  })
  if (!Number.isSafeInteger(shardCount) || shardCount < 1 || shardCount > 256) return Option.none()
  const selected = Option.match(split, {
    onNone: () => [Option.fromNullable(files.find((file) => file.path === primary))],
    onSome: (match) => Option.match(Option.all({
      prefix: Option.fromNullable(match.at(1)),
      count: Option.fromNullable(match.at(2)),
    }), {
      onNone: (): readonly Option.Option<ResolvedHubSnapshot["ggufFiles"][number]>[] => [],
      onSome: ({ prefix, count }) => Array.from({ length: shardCount }, (_, index) =>
        Option.fromNullable(files.find((file) => file.path ===
          `${prefix}-${String(index + 1).padStart(5, "0")}-of-${count}.gguf`))),
    }),
  })
  if (selected.length !== shardCount) return Option.none()
  let total = 0
  for (const file of selected) {
    if (Option.isNone(file) || Option.isNone(file.value.sizeBytes)) return Option.none()
    if (!Number.isSafeInteger(file.value.sizeBytes.value) || file.value.sizeBytes.value <= 0) return Option.none()
    total += file.value.sizeBytes.value
  }
  return Option.some(total)
}

/** Resolve a stable quant selector against one immutable live Hub snapshot. */
export const resolveModelRecipeArtifact = (
  model: ModelRecipe,
  candidate: ModelRecipeArtifact,
  snapshot: ResolvedHubSnapshot,
): Option.Option<ResolvedModelRecipe> => {
  const capability = model.performance.benchmarks.find(({ benchmarkId }) =>
    benchmarkId === TERMINAL_BENCH_ID)
  if (!capability) return Option.none()
  const selector = candidate.filenameIncludes.toLowerCase()
  const matches = snapshot.ggufFiles.filter(({ path }) => {
    const lower = path.toLowerCase()
    const basename = lower.slice(lower.lastIndexOf("/") + 1)
    return lower.includes(selector)
      && !basename.startsWith("mmproj-")
      && !basename.includes("imatrix")
      && (!laterShard.test(lower) || firstShard.test(lower))
  })
  if (matches.length !== 1) return Option.none()
  return Option.flatMap(Option.fromNullable(matches.at(0)), (primary) =>
    Option.map(selectedWeightBytes(primary.path, snapshot.ggufFiles), (publishedWeightBytes) => ({
    id: candidate.id,
    modelId: model.id,
    family: model.family,
    displayName: model.displayName,
    repo: snapshot.repository,
    revision: snapshot.commit,
    primaryGguf: primary.path,
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
      id: Option.getOrElse(
        Option.filter(snapshot.license, (license) => license !== "other"),
        () => model.licenseReview.expectedId,
      ),
      url: Option.getOrElse(
        Option.flatMap(
          Option.filter(snapshot.license, (license) => license !== "other"),
          () => snapshot.licenseUrl,
        ),
        () => model.licenseReview.url,
      ),
      acknowledgementRequired: model.licenseReview.acknowledgementRequired,
    },
    capability,
    })))
}

export const recipeSourcePageUrl = (model: ResolvedModelRecipe): string => {
  const path = model.primaryGguf.split("/").map(encodeURIComponent).join("/")
  return `https://huggingface.co/${model.repo}/blob/${model.revision}/${path}`
}
