import { Option } from "effect"
import {
  AssessmentEnvironmentIdSchema, CatalogFormModelIdSchema, CustomFormModelIdSchema, HuggingFaceArtifactSelectorSchema,
  HuggingFaceRepositoryIdSchema, ModelAssessmentIdSchema, ModelReleaseDateSchema, ModelVariantLabelSchema,
  type CandidateInspection, type CandidateVersion, type CatalogLocalModel, type CustomLocalModel, type CustomModelOrigin,
  type AttachedDrafter, type AttachedDrafterAvailability, type DrafterCandidate, type LocalModelAssessment, type UnsupportedLocalModel, type UnsupportedReason,
} from "@magnitudedev/sdk"

const GB = 1_000_000_000
export const catalogId = CatalogFormModelIdSchema.make("qwen3.6-35b-a3b:gguf:q4")
export const repository = HuggingFaceRepositoryIdSchema.make("HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive")
const environmentId = AssessmentEnvironmentIdSchema.make("environment")
const assessmentId = ModelAssessmentIdSchema.make("assessment")

const capabilities = { vision: false, tools: true, structuredOutput: true, reasoning: { supported: false, efforts: [], defaultEffort: Option.none() } }
const metadata = { format: "gguf" as const, quantization: "Q4_K", quantizationName: "Q4_K_P", architecture: "qwen35moe", maximumContextLength: Option.some(32_768), storageBytes: 23 * GB }

export const fits: Extract<LocalModelAssessment, { readonly _tag: "Fits" }> = {
  _tag: "Fits", assessmentId, environmentId, profile: { contextLength: 32_768 },
  memory: { domains: [], totalRequiredBytes: 24 * GB, requiredSystemMemoryBytes: 0 },
  performance: { _tag: "Estimated", samples: [{ contextTokens: 32_768, estimatedTokensPerSecond: 80 }] },
}
export const doesNotFit: LocalModelAssessment = {
  _tag: "DoesNotFit", assessmentId, environmentId, profile: { contextLength: 32_768 }, memoryDomains: [], totalRequiredBytes: 0, deficitBytes: 4 * GB, limitingResource: "system",
}
export const unsupported = (...reasons: [UnsupportedReason, ...UnsupportedReason[]]): LocalModelAssessment => ({
  _tag: "Unsupported", environmentId, profile: Option.some({ contextLength: 32_768 }), reasons, detail: "engine detail",
})

export const unreadable: LocalModelAssessment = {
  _tag: "Unsupported", environmentId, profile: Option.none(), reasons: [{ _tag: "Unreadable" }], detail: "file is shorter than its header declares",
}

export const catalogModel = (): CatalogLocalModel => ({
  _tag: "Catalog",
  modelId: catalogId,
  storageBytes: 24 * GB,
  presentation: { displayName: "Qwen3.6 35B-A3B", variantLabel: ModelVariantLabelSchema.make("Q4"), description: "", license: Option.some("apache-2.0"), sourceUrls: [] },
  catalogData: { releaseDate: ModelReleaseDateSchema.make("2026-01-01"), parameterization: { architecture: "dense", totalParameters: 1 }, intelligence: 1, support: { _tag: "Supported" }, fidelityRank: 1, quantizationAware: false, aliases: [] },
  acquisitionState: { _tag: "Installed", installation: { _tag: "Resolved", installedBytes: 24 * GB, primaryPath: "/models/qwen.gguf", ownership: "Magnitude" }, residencyState: { _tag: "Unloaded" } },
  servingState: { _tag: "Assessed", capabilities, speculativeMethod: Option.none(), metadata, assessment: fits, rankingScores: { speed: Option.some(1), intelligence: 1, fidelity: 1 } },
  missingComponents: [],
  storedComponents: [],
  drafterWithheld: false,
})

export const drafterCandidate = (overrides: Partial<DrafterCandidate> = {}): DrafterCandidate => ({
  category: "CatalogDrafter",
  reference: { _tag: "Catalog", modelId: catalogId },
  name: "Magnitude DFlash for Qwen3.6 35B-A3B",
  label: "Trained for Qwen3.6 35B-A3B",
  sizeBytes: 0.9 * GB,
  lineage: "Lineage",
  verdict: { _tag: "Bound", method: { _tag: "DFlash" }, memoryBytes: GB },
  onThisComputer: false,
  method: Option.some({ _tag: "DFlash" }),
  automatic: (overrides.category ?? "CatalogDrafter") !== "Other" && (overrides.verdict?._tag ?? "Bound") === "Bound",
  ...overrides,
})

export const attachedDrafter = (availability: AttachedDrafterAvailability = { _tag: "Present" }, overrides: Partial<AttachedDrafter> = {}): AttachedDrafter => ({
  reference: { _tag: "Catalog", modelId: catalogId },
  name: "Magnitude DFlash for Qwen3.6 35B-A3B",
  label: "Trained for Qwen3.6 35B-A3B",
  sizeBytes: 0.9 * GB,
  verdict: { _tag: "Bound", method: { _tag: "DFlash" }, memoryBytes: GB },
  availability,
  updateAvailable: false,
  ...overrides,
})

const sourceLabel = (origin: CustomModelOrigin) => origin._tag === "Added" ? "Hugging Face" : origin.source._tag === "Ollama" ? "Ollama" : origin.source._tag === "LmStudio" ? "LM Studio" : origin.source._tag === "Folder" ? "models" : "Hugging Face cache"

export const customModel = (origin: CustomModelOrigin, overrides: Partial<CustomLocalModel> = {}): CustomLocalModel => ({
  _tag: "Custom",
  modelId: CustomFormModelIdSchema.make(origin._tag === "Added" || origin._tag === "Found" && origin.source._tag === "HuggingFaceCache"
    ? `hf:${repository}/Qwen-Q4_K_P.gguf` : "local:qwen3.6-35b-a3b-uncensored-q4_k_p:aaaaaaaaaaaa"),
  origin,
  presentation: { displayName: "Qwen3.6 35B A3B Uncensored", variantLabel: ModelVariantLabelSchema.make("Q4_K_P"), sourceLabel: sourceLabel(origin), variantOf: Option.some(catalogId), license: Option.none(), sourceUrls: [] },
  drafter: Option.none(),
  drafterPending: false,
  drafterChoice: { _tag: "Auto" },
  recommendedDrafter: Option.none(),
  acquisitionState: origin._tag === "Added"
    ? { _tag: "Installed", installation: { _tag: "Resolved", installedBytes: 23 * GB, primaryPath: "/models/qwen.gguf", ownership: "Magnitude" }, residencyState: { _tag: "Unloaded" } }
    : { _tag: "External", installation: { _tag: "Resolved", installedBytes: 23 * GB, primaryPath: "/Users/me/.ollama/models/blobs/sha256-1", ownership: "External" }, residencyState: { _tag: "Unloaded" } },
  servingState: { _tag: "Assessed", capabilities, speculativeMethod: Option.none(), metadata, assessment: fits },
  storageBytes: 23 * GB,
  ...overrides,
})

export const added: CustomModelOrigin = { _tag: "Added", repository, commit: "4f1c2e9abcdef" }
/** A model found in a folder the user added to Magnitude. */
export const inFolder: CustomModelOrigin = { _tag: "Found", source: { _tag: "Folder", path: "/Users/me/models" } }
export const foundIn = (source: Extract<CustomModelOrigin, { _tag: "Found" }>["source"]): CustomModelOrigin => ({ _tag: "Found", source })

export const version = (label: string, overrides: Partial<CandidateVersion> = {}): CandidateVersion => ({
  label,
  modelId: CustomFormModelIdSchema.make(`hf:${repository}/${label}.gguf`),
  files: [`${label}.gguf`],
  sizeBytes: 23.4 * GB,
  downloadBytes: overrides.sizeBytes ?? 23.4 * GB,
  assessment: Option.some(fits),
  recommended: false,
  catalogModel: Option.none(),
  onThisComputer: Option.none(),
  ...overrides,
})

export const inspection = (overrides: Partial<CandidateInspection> = {}): CandidateInspection => ({
  source: { _tag: "HuggingFace", repository, commit: "4f1c2e9abcdef" },
  displayName: "Qwen3.6 35B A3B Uncensored HauhauCS Aggressive",
  license: Option.some("apache-2.0"),
  variantOf: Option.some(catalogId),
  versions: [
    version("Q8_K_P", { sizeBytes: 43.6 * GB, assessment: Option.some(doesNotFit) }),
    version("Q4_K_P", { recommended: true }),
    version("Q2_K_P", { sizeBytes: 15 * GB, assessment: Option.some(unsupported({ _tag: "WeightTypes", types: [{ ggmlType: "Q2_K", tensorCount: 2 }] })) }),
  ],
  selectedVersion: Option.none(),
  projector: Option.none(),
  unsupportedArchitecture: Option.none(),
  ...overrides,
})

export const unsupportedLocal = (displayName: string, origin: CustomModelOrigin, reason: UnsupportedReason): UnsupportedLocalModel => ({
  modelId: Option.none(), displayName, origin, path: `/models/${displayName}.gguf`, reasons: [reason], detail: "engine detail",
})

export const selector = HuggingFaceArtifactSelectorSchema.make("Q4_K_P.gguf")
