import { Option } from "effect"
import {
  AssessmentEnvironmentIdSchema,
  CatalogFormModelIdSchema,
  CustomFormModelIdSchema,
  HuggingFaceRepositoryIdSchema,
  HuggingFaceArtifactSelectorSchema,
  HttpsUrlSchema,
  LocalInferenceAcceleratorIdSchema,
  LocalInferenceDeviceIdSchema,
  LocalInferenceMemoryDomainIdSchema,
  ModelAssessmentIdSchema,
  ModelVariantLabelSchema,
  type CandidateInspection,
  type CandidateVersion,
  type CatalogLocalModel,
  type CustomLocalModel,
  type DrafterCandidate,
  type HuggingFaceSearchResponse,
  type LocalInferenceHardware,
  type LocalModel,
  type LocalModelAcquisitionState,
  type LocalModelAssessment,
  type ModelCatalogState,
  type ModelLocation,
  type ModelReleaseDate,
  type UnsupportedLocalModel,
} from "@magnitudedev/sdk"

const GIB = 1024 ** 3
const GB = 1_000_000_000
const TEST_CATALOG_MODEL_ID = CatalogFormModelIdSchema.make("qwen-test:gguf:q4")
const TEST_MEMORY_DOMAIN_ID = LocalInferenceMemoryDomainIdSchema.make("memory")
const ENVIRONMENT_ID = AssessmentEnvironmentIdSchema.make("environment_test")
export const TEST_REPOSITORY = HuggingFaceRepositoryIdSchema.make("HauhauCS/Qwen-Test-Uncensored-GGUF")
export const TEST_COMMIT = "4f1c2e9d0b7a6c5e4f3a2b1c0d9e8f7a6b5c4d3e"
export const FOUND_MODEL_ID = CustomFormModelIdSchema.make("hf:bartowski/Qwen-Test-GGUF/Qwen-Test-Q4_K_M.gguf")
export const ADDED_MODEL_ID = CustomFormModelIdSchema.make(`hf:${TEST_REPOSITORY}/Qwen-Test-Q4_K_P.gguf`)
export const IMPORTED_MODEL_ID = CustomFormModelIdSchema.make(`local:${"a".repeat(64)}`)
export const CATALOG_DRAFTER_ID = CatalogFormModelIdSchema.make("qwen-test:gguf:q4")

export const makeHardware = (
  overrides: Partial<LocalInferenceHardware> = {},
): LocalInferenceHardware => ({
  platform: "Linux",
  architecture: "X64",
  productName: Option.none(),
  processor: Option.some("Test CPU"),
  physicalCores: Option.none(),
  logicalCores: 16,
  totalSystemMemoryBytes: 64 * GIB,
  accelerators: [{
    acceleratorId: LocalInferenceAcceleratorIdSchema.make("gpu"),
    name: "Test GPU",
    executors: [{ deviceId: LocalInferenceDeviceIdSchema.make("cuda:test"), backend: "cuda", memoryDomainIds: [TEST_MEMORY_DOMAIN_ID] }],
  }],
  selectedDeviceId: Option.some(LocalInferenceDeviceIdSchema.make("cuda:test")),
  memoryDomains: [{
    memoryDomainId: TEST_MEMORY_DOMAIN_ID,
    kind: "PhysicalDevice",
    totalBytes: 24 * GIB,
    availableBytes: Option.some(6 * GIB),
  }],
  ...overrides,
})

const capabilities = {
  vision: false,
  tools: true,
  structuredOutput: true,
  reasoning: { supported: false as const, efforts: [], defaultEffort: Option.none() },
}

const CONTEXT_LENGTH = 32_768

const performance = (contextLength: number) => [...new Set([
  ...[25_000, 50_000, 75_000].filter((context) => context <= contextLength),
  contextLength,
])].sort((left, right) => left - right).map((contextTokens) => ({
  contextTokens,
  estimatedTokensPerSecond: contextTokens === contextLength ? 24 : 28,
}))

const metadata = {
  format: "gguf",
  quantization: "Q4_K_M",
  quantizationName: "4-bit",
  architecture: "test",
  storageBytes: 16 * GIB,
  maximumContextLength: Option.some(CONTEXT_LENGTH),
}

export const fitsAssessment = (): Extract<LocalModelAssessment, { readonly _tag: "Fits" }> => ({
  _tag: "Fits",
  profile: { contextLength: CONTEXT_LENGTH },
  assessmentId: ModelAssessmentIdSchema.make("assessment_test"),
  environmentId: ENVIRONMENT_ID,
  memory: { domains: [], totalRequiredBytes: 0, requiredSystemMemoryBytes: 0 },
  performance: performance(CONTEXT_LENGTH),
})

export const doesNotFitAssessment = (): LocalModelAssessment => ({
  _tag: "DoesNotFit",
  profile: { contextLength: CONTEXT_LENGTH },
  assessmentId: ModelAssessmentIdSchema.make("assessment_large"),
  environmentId: ENVIRONMENT_ID,
  memoryDomains: [],
  totalRequiredBytes: 0,
  deficitBytes: 8 * GIB,
  limitingResource: "memory",
})

export const unsupportedAssessment = (ggmlType: string): LocalModelAssessment => ({
  _tag: "Unsupported",
  profile: Option.some({ contextLength: CONTEXT_LENGTH }),
  environmentId: ENVIRONMENT_ID,
  reasons: [{ _tag: "WeightTypes", types: [{ ggmlType, tensorCount: 2 }] }],
  detail: `source_element has no ${ggmlType} case`,
})

export const unreadableAssessment = (): LocalModelAssessment => ({
  _tag: "Unsupported",
  profile: Option.none(),
  environmentId: ENVIRONMENT_ID,
  reasons: [{ _tag: "Unreadable" }],
  detail: "file is shorter than its header declares",
})

const catalogPresentation = {
  displayName: "Qwen Test",
  variantLabel: ModelVariantLabelSchema.make("Q4"),
  description: "Test model",
  license: Option.some("Apache-2.0"),
  sourceUrls: [HttpsUrlSchema.make("https://huggingface.co/test/model")],
}

const makeCatalogOnlyModel = (
  overrides: Partial<CatalogLocalModel> = {},
  modelId = TEST_CATALOG_MODEL_ID,
): CatalogLocalModel => ({
  _tag: "Catalog",
  modelId,
  presentation: catalogPresentation,
  storageBytes: metadata.storageBytes,
  catalogData: {
    releaseDate: "2026-01-01" as ModelReleaseDate,
    parameterization: { architecture: "dense", totalParameters: 8_000_000_000 },
    intelligence: 75,
    support: { _tag: "Supported" },
    fidelityRank: 75,
    quantizationAware: false,
    aliases: [],
  },
  acquisitionState: { _tag: "NotInstalled" },
  missingComponents: [],
  storedComponents: [],
  drafterWithheld: false,
  servingState: {
    _tag: "Assessed",
    metadata,
    capabilities,
    speculativeMethod: Option.some({ _tag: "DFlash" }),
    assessment: fitsAssessment(),
    rankingScores: Option.some({ intelligence: 0.75, speed: 0.65, fidelity: 0.75 }),
  },
  ...overrides,
})

export const makeCatalogModel = (overrides: Partial<CatalogLocalModel> = {}): CatalogLocalModel =>
  makeCatalogOnlyModel(overrides)

const installation = (primaryPath: string, ownership: "Magnitude" | "External" = "Magnitude") => ({
  _tag: "Resolved" as const,
  installedBytes: metadata.storageBytes,
  primaryPath,
  ownership,
})

export const makeInstalledCatalogModel = (
  overrides: Partial<CatalogLocalModel> = {},
): CatalogLocalModel => makeCatalogOnlyModel({
  acquisitionState: {
    _tag: "Installed",
    installation: installation("/models/catalog-model.gguf"),
    residencyState: { _tag: "Unloaded" },
  },
  ...overrides,
})

export const makeAcquiringModel = (
  acquisitionState: LocalModelAcquisitionState,
  overrides: Partial<CatalogLocalModel> = {},
): CatalogLocalModel => makeCatalogOnlyModel({ acquisitionState, ...overrides })

export const makeDrafterCandidate = (overrides: Partial<DrafterCandidate> = {}): DrafterCandidate => ({
  category: "CatalogDrafter",
  reference: { _tag: "Catalog", modelId: CATALOG_DRAFTER_ID },
  name: "Magnitude DFlash for Qwen Test",
  label: "Trained for Qwen Test",
  sizeBytes: 900_000_000,
  lineage: "Lineage",
  verdict: { _tag: "Bound", method: { _tag: "DFlash" }, memoryBytes: GIB },
  onThisComputer: false,
  method: Option.some({ _tag: "DFlash" }),
  automatic: (overrides.category ?? "CatalogDrafter") !== "Other" && (overrides.verdict?._tag ?? "Bound") === "Bound",
  ...overrides,
})

export const makeDrafterCandidates = (): readonly DrafterCandidate[] => [
  makeDrafterCandidate({
    category: "NextToModel",
    reference: {
      _tag: "HuggingFace",
      repository: TEST_REPOSITORY,
      commit: TEST_COMMIT,
      path: HuggingFaceArtifactSelectorSchema.make("drafters/dflash.gguf"),
    },
    name: `${TEST_REPOSITORY}/dflash.gguf`,
    label: "In this repository",
    sizeBytes: 1_200_000_000,
    lineage: "Lineage",
    verdict: {
      _tag: "Rejected",
      reasons: [{ _tag: "TargetMismatch", detail: "hidden size 2048 does not match 4096" }],
      detail: "drafter hidden size mismatch",
    },
    method: Option.none(),
  }),
  makeDrafterCandidate(),
  makeDrafterCandidate({
    category: "BuiltIn",
    reference: { _tag: "Embedded" },
    name: "MTP heads",
    label: "Built into this model",
    sizeBytes: 0,
    lineage: "Exact",
    verdict: { _tag: "Bound", method: { _tag: "Mtp" }, memoryBytes: GIB / 2 },
    method: Option.some({ _tag: "Mtp" }),
  }),
]

const customPresentation = (sourceLabel: string) => ({
  displayName: "Qwen Test Uncensored",
  variantLabel: ModelVariantLabelSchema.make("Q4_K_M"),
  sourceLabel,
  variantOf: Option.some(TEST_CATALOG_MODEL_ID),
  license: Option.some("apache-2.0"),
  sourceUrls: [],
})

const customServing = (assessment: LocalModelAssessment = fitsAssessment()): CustomLocalModel["servingState"] => ({
  _tag: "Assessed",
  metadata,
  capabilities,
  speculativeMethod: Option.none(),
  assessment,
})

export const makeFoundModel = (overrides: Partial<CustomLocalModel> = {}): CustomLocalModel => ({
  _tag: "Custom",
  modelId: FOUND_MODEL_ID,
  origin: { _tag: "Found", source: { _tag: "HuggingFaceCache" } },
  presentation: customPresentation("Hugging Face cache"),
  drafter: Option.none(),
  drafterPending: false,
  drafterChoice: { _tag: "Auto" },
  recommendedDrafter: Option.some(makeDrafterCandidate()),
  acquisitionState: {
    _tag: "External",
    installation: installation("/cache/Qwen-Test-Q4_K_M.gguf", "External"),
    residencyState: { _tag: "Unloaded" },
  },
  servingState: customServing(),
  storageBytes: metadata.storageBytes,
  ...overrides,
})

export const makeAddedModel = (overrides: Partial<CustomLocalModel> = {}): CustomLocalModel => makeFoundModel({
  modelId: ADDED_MODEL_ID,
  origin: { _tag: "Added", repository: TEST_REPOSITORY, commit: TEST_COMMIT },
  presentation: { ...customPresentation("Hugging Face"), variantLabel: ModelVariantLabelSchema.make("Q4_K_P") },
  drafter: Option.some({
    reference: { _tag: "Catalog", modelId: CATALOG_DRAFTER_ID },
    name: "Magnitude DFlash for Qwen Test",
    label: "Trained for Qwen Test",
    sizeBytes: 900_000_000,
    verdict: { _tag: "Bound", method: { _tag: "DFlash" }, memoryBytes: GIB },
    availability: { _tag: "Present" },
    updateAvailable: false,
  }),
  recommendedDrafter: Option.none(),
  acquisitionState: {
    _tag: "Installed",
    installation: installation("/magnitude/models/Qwen-Test-Q4_K_P.gguf"),
    residencyState: { _tag: "Unloaded" },
  },
  ...overrides,
})

export const makeImportedModel = (overrides: Partial<CustomLocalModel> = {}): CustomLocalModel => makeFoundModel({
  modelId: IMPORTED_MODEL_ID,
  origin: { _tag: "Imported", path: "/Users/test/models/qwen-test.gguf" },
  presentation: { ...customPresentation("GGUF file"), variantOf: Option.none() },
  recommendedDrafter: Option.none(),
  acquisitionState: {
    _tag: "Unavailable",
    installation: installation("/Users/test/models/qwen-test.gguf", "External"),
    failure: { code: "missing", message: "The file was moved or deleted", retryable: false },
  },
  ...overrides,
})

export const makeUnsupportedLocalModel = (overrides: Partial<UnsupportedLocalModel> = {}): UnsupportedLocalModel => ({
  modelId: Option.none(),
  displayName: "DeepSeek V4 Flash",
  origin: { _tag: "Found", source: { _tag: "Ollama" } },
  path: "/Users/test/.ollama/models/blobs/sha256-abc",
  reasons: [{ _tag: "Architecture", architecture: "deepseek4" }],
  detail: "unknown general.architecture deepseek4",
  ...overrides,
})

type CatalogSnapshotState = Exclude<ModelCatalogState, { readonly _tag: "Initializing" }>

export const catalogState = (
  models: readonly LocalModel[] = [],
  unsupported: readonly UnsupportedLocalModel[] = [],
  hidden: readonly LocalModel[] = [],
): CatalogSnapshotState => ({
  _tag: "Ready",
  providers: [],
  models: models.map((product) => ({ _tag: "Local", product, offering: Option.none() })),
  failures: [],
  localModelPreparation: {
    discovery: { complete: true, modelsFound: models.length },
    assessment: { complete: true, settledModels: models.length, totalModels: models.length },
  },
  unsupportedLocalModels: unsupported,
  hiddenLocalModels: hidden,
})

export const makeSearchResponse = (overrides: Partial<HuggingFaceSearchResponse> = {}): HuggingFaceSearchResponse => ({
  results: [
    {
      repository: TEST_REPOSITORY,
      displayName: "Qwen Test Uncensored",
      uploader: "HauhauCS",
      variantOf: Option.some(TEST_CATALOG_MODEL_ID),
      downloads: 48_213,
      architecture: Option.some("qwen35moe"),
      unsupported: Option.none(),
      gated: false,
      versionCount: Option.none(),
    },
    {
      repository: HuggingFaceRepositoryIdSchema.make("huihui-ai/Qwen-Test-abliterated-GGUF"),
      displayName: "Qwen Test abliterated",
      uploader: "huihui-ai",
      variantOf: Option.none(),
      downloads: 912,
      architecture: Option.some("qwen35moe"),
      unsupported: Option.none(),
      gated: false,
      versionCount: Option.none(),
    },
  ],
  ...overrides,
})

export const makeUnsupportedSearchResult = (): HuggingFaceSearchResponse["results"][number] => ({
  repository: HuggingFaceRepositoryIdSchema.make("unsloth/DeepSeek-V4-Flash-GGUF"),
  displayName: "DeepSeek V4 Flash",
  uploader: "unsloth",
  variantOf: Option.none(),
  downloads: 2_400_000,
  architecture: Option.some("deepseek4"),
  unsupported: Option.some([{ _tag: "Architecture", architecture: "deepseek4" }]),
  gated: false,
  versionCount: Option.none(),
})

const version = (
  label: string,
  sizeGb: number,
  assessment: LocalModelAssessment,
  overrides: Partial<CandidateVersion> = {},
): CandidateVersion => ({
  label,
  modelId: CustomFormModelIdSchema.make(`hf:${TEST_REPOSITORY}/Qwen-Test-${label}.gguf`),
  files: [`Qwen-Test-${label}.gguf`],
  sizeBytes: Math.round(sizeGb * GB),
  downloadBytes: Math.round(sizeGb * GB),
  assessment: Option.some(assessment),
  recommended: false,
  catalogModel: Option.none(),
  onThisComputer: Option.none(),
  ...overrides,
})

export const makeInspection = (overrides: Partial<CandidateInspection> = {}): CandidateInspection => ({
  source: { _tag: "HuggingFace", repository: TEST_REPOSITORY, commit: TEST_COMMIT },
  displayName: "Qwen Test Uncensored",
  license: Option.some("apache-2.0"),
  variantOf: Option.some(TEST_CATALOG_MODEL_ID),
  versions: [
    version("Q8_K_P", 43.6, doesNotFitAssessment()),
    version("Q6_K_P", 30.6, fitsAssessment(), { onThisComputer: Option.some(FOUND_MODEL_ID) }),
    version("Q4_K_P", 23.4, fitsAssessment(), { recommended: true }),
    version("Q2_K_P", 15.0, unsupportedAssessment("Q2_K")),
  ],
  selectedVersion: Option.none(),
  projector: Option.none(),
  unsupportedArchitecture: Option.none(),
  ...overrides,
})

export const makeLocations = (): readonly ModelLocation[] => [
  { kind: "HuggingFaceCache", paths: [
    { path: "/Volumes/hf/hub", state: "Found" },
    { path: "/Users/test/.cache/huggingface/hub", state: "Found" },
  ], enabled: true, automatic: true, state: "Found" },
  { kind: "LmStudio", paths: [
    { path: "/Users/test/.lmstudio/models", state: "NotFound" },
    { path: "/Users/test/.cache/lm-studio/models", state: "NotFound" },
  ], enabled: true, automatic: true, state: "NotFound" },
  { kind: "Ollama", paths: [{ path: "/Users/test/.ollama/models", state: "Found" }], enabled: false, automatic: true, state: "Found" },
  { kind: "Folder", paths: [{ path: "/Volumes/External/models", state: "Found" }], enabled: true, automatic: false, state: "Found" },
]
