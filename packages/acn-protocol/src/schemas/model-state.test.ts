import { describe, expect, it } from "vitest"
import { Option, Schema } from "effect"
import {
  CatalogBaseIdSchema,
  CatalogVariantIdSchema,
  CatalogLocalModelServingStateSchema,
  CatalogSupportSchema,
  DiscoveredLocalModelServingStateSchema,
  IntelligenceScoreSchema,
  LocalModelMemorySchema,
  LocalModelPreparationSchema,
  LocalModelSchema,
  LocalModelServingStateSchema,
  ModelIdSchema,
  ModelParameterizationSchema,
  ModelReleaseDateSchema,
  parseModelId,
} from "./model-state"

const catalogModel = {
  _tag: "Catalog",
  modelId: "model:gguf:q4",
  storageBytes: 1,
  presentation: { displayName: "Model", variantLabel: "Q4", description: "", sourceUrls: [] },
  catalogData: {
    releaseDate: "2026-08-29",
    parameterization: { architecture: "dense", totalParameters: 1 },
    intelligence: 1,
    support: { _tag: "Supported" },
    fidelityRank: 1,
    quantizationAware: false,
  },
  acquisitionState: { _tag: "NotInstalled" },
  servingState: {
    _tag: "Failed",
    profile: { contextLength: 4096 },
    failure: { code: "unavailable", message: "Unavailable", retryable: true },
  },
} as const

describe("CatalogSupportSchema", () => {
  it("requires a disabled reason and a deprecated replacement", () => {
    expect(() => Schema.decodeUnknownSync(CatalogSupportSchema)({ _tag: "Disabled", reason: "" })).toThrow()
    expect(() => Schema.decodeUnknownSync(CatalogSupportSchema)({
      _tag: "Deprecated", since: "2026-09-27", reason: "superseded",
    })).toThrow()
    expect(() => Schema.decodeUnknownSync(CatalogSupportSchema)({
      _tag: "Deprecated", since: "2026-09-27", reason: "superseded", replacement: "model:gguf:q4",
    })).not.toThrow()
  })
})

describe("ModelIdSchema", () => {
  it("accepts canonical catalog and Hugging Face callable identities", () => {
    expect(Schema.decodeUnknownSync(CatalogBaseIdSchema)("qwen3.5-4b")).toBe("qwen3.5-4b")
    expect(Schema.decodeUnknownSync(CatalogVariantIdSchema)("gguf:q4")).toBe("gguf:q4")
    for (const id of [
      "qwen3.5-4b:gguf:q4",
      "hf:owner/repository/model-q4.gguf",
      "hf:owner/repository/subdirectory/model-00001-of-00002.gguf",
    ]) expect(Schema.decodeUnknownSync(ModelIdSchema)(id)).toBe(id)
  })

  it("parses canonical identity into its semantic components and round-trips them", () => {
    const catalog = Schema.decodeUnknownSync(ModelIdSchema)("qwen3.5-4b:gguf:q4")
    const huggingFace = Schema.decodeUnknownSync(ModelIdSchema)("hf:owner/repository/sub/model-q4.gguf")
    expect(parseModelId(catalog)).toEqual({
      _tag: "Catalog",
      baseId: "qwen3.5-4b",
      variantId: "gguf:q4",
    })
    expect(parseModelId(huggingFace)).toEqual({
      _tag: "HuggingFace",
      repositoryId: "owner/repository",
      artifactSelector: "sub/model-q4.gguf",
    })
  })

  it("rejects aliases, missing selectors, traversal, backslashes, and non-GGUF artifacts", () => {
    for (const id of [
      "qwen3.5-4b",
      "hf:owner/repository",
      "hf:owner/repository/../model.gguf",
      "hf:owner/repository/path\\model.gguf",
      "hf:owner/repository/model.safetensors",
      "hf:owner/repository/model\n.gguf",
      "owner/repository/model.gguf",
    ]) expect(() => Schema.decodeUnknownSync(ModelIdSchema)(id)).toThrow()
  })
})

describe("LocalModelSchema invariants", () => {
  it("preserves structured acquisition failure facts", () => {
    const decoded = Schema.decodeUnknownSync(LocalModelSchema)({
      ...catalogModel,
      acquisitionState: {
        _tag: "InstallFailed",
        failure: { _tag: "InsufficientDiskSpace", requiredBytes: 40, availableBytes: 30 },
      },
    })
    expect(decoded._tag === "Catalog" && decoded.acquisitionState).toEqual({
      _tag: "InstallFailed",
      failure: { _tag: "InsufficientDiskSpace", requiredBytes: 40, availableBytes: 30 },
    })
  })

  it("allows a catalog failure before a serving profile can be resolved", () => {
    expect(() => Schema.decodeUnknownSync(LocalModelSchema)({
      ...catalogModel,
      servingState: {
        _tag: "Failed",
        failure: { code: "assessment_failed", message: "Assessment failed", retryable: true },
      },
    })).not.toThrow()
  })

  it("requires domain-specific catalog and discovery facts", () => {
    const { catalogData: _catalogData, ...catalogWithoutData } = catalogModel
    expect(() => Schema.decodeUnknownSync(LocalModelSchema)({
      ...catalogWithoutData,
    })).toThrow()
    expect(() => Schema.decodeUnknownSync(LocalModelSchema)({
      ...catalogModel,
      modelId: "hf:owner/repository/model.gguf",
    })).toThrow()
    expect(() => Schema.decodeUnknownSync(LocalModelSchema)({
      _tag: "Discovered",
      modelId: "hf:owner/repository/model.gguf",
      presentation: catalogModel.presentation,
      state: {
        _tag: "Ready",
        installation: { _tag: "Resolved", installedBytes: 1, primaryPath: "/model.gguf", ownership: "ExternalHuggingFace" },
        residencyState: { _tag: "Unloaded" },
        catalogAttribution: { _tag: "NotInCatalog" },
      },
    })).toThrow()
    const unavailable = Schema.decodeUnknownSync(LocalModelSchema)({
      _tag: "Discovered",
      modelId: "hf:owner/repository/model.gguf",
      presentation: catalogModel.presentation,
      state: {
        _tag: "Unavailable",
        installation: { _tag: "Resolved", installedBytes: 1, primaryPath: "/model.gguf", ownership: "ExternalHuggingFace" },
        failure: { code: "invalid", message: "Invalid", retryable: false },
      },
    })
    expect(unavailable._tag).toBe("Discovered")
    expect(() => Schema.decodeUnknownSync(LocalModelSchema)({
      ...unavailable,
      modelId: "model:gguf:q4",
    })).toThrow()
  })

  it("keeps failed serving profiles only on ready discoveries", () => {
    const discovered = {
      _tag: "Discovered",
      modelId: "hf:owner/repository/model.gguf",
      presentation: catalogModel.presentation,
    } as const
    const failure = { code: "failed", message: "Failed", retryable: true } as const
    const ready = {
      _tag: "Ready",
      installation: {
        _tag: "Resolved", installedBytes: 1, primaryPath: "/model.gguf", ownership: "ExternalHuggingFace",
      },
      residencyState: { _tag: "Unloaded" },
      catalogAttribution: { _tag: "NotInCatalog" },
    } as const
    expect(() => Schema.decodeUnknownSync(LocalModelSchema)({
      ...discovered,
      state: { ...ready, servingState: { _tag: "Failed", failure } },
    })).toThrow()
    expect(() => Schema.decodeUnknownSync(LocalModelSchema)({
      ...discovered,
      state: {
        ...ready,
        servingState: { _tag: "Failed", profile: { contextLength: 4096 }, failure },
      },
    })).not.toThrow()
    const unavailable = Schema.decodeUnknownSync(LocalModelSchema)({
      ...discovered,
      state: {
        _tag: "Unavailable",
        installation: ready.installation,
        failure,
        servingState: { _tag: "Failed", profile: { contextLength: 4096 }, failure },
      },
    })
    expect(unavailable._tag === "Discovered" && unavailable.state).toEqual({
      _tag: "Unavailable",
      installation: ready.installation,
      failure,
    })
  })

  it("stores an assessed profile exactly once, in the assessment", () => {
    const assessed = {
      _tag: "Assessed",
      metadata: {
        format: "gguf", architecture: "test", quantization: "q4", quantizationName: "Q4",
        storageBytes: 1,
      },
      capabilities: {
        vision: false, tools: false, structuredOutput: false,
        reasoning: { supported: false, efforts: [] },
      },
      assessment: {
        _tag: "Fits",
        assessmentId: "assessment",
        environmentId: "environment",
        profile: { contextLength: 4096 },
        memory: {
          domains: [], totalRequiredBytes: 0, requiredSystemMemoryBytes: 0,
        },
        performance: [{ contextTokens: 4096, estimatedTokensPerSecond: 2 }],
      },
    } as const
    expect(() => Schema.decodeUnknownSync(LocalModelServingStateSchema)(assessed)).not.toThrow()
  })

  it("requires performance for fitting models and admits Unsupported only for discovered models", () => {
    const failure = { code: "unsupported_family", message: "Unrecognized family", retryable: false }
    const assessed = {
      _tag: "Assessed",
      metadata: {
        format: "gguf", architecture: "test", quantization: "q4", quantizationName: "Q4",
        storageBytes: 1,
      },
      capabilities: {
        vision: false, tools: false, structuredOutput: false,
        reasoning: { supported: false, efforts: [] },
      },
    } as const
    const fits = {
      ...assessed,
      assessment: {
        _tag: "Fits",
        assessmentId: "assessment",
        environmentId: "environment",
        profile: { contextLength: 4096 },
        memory: {
          domains: [], totalRequiredBytes: 0, requiredSystemMemoryBytes: 0,
        },
        performance: [{ contextTokens: 4096, estimatedTokensPerSecond: 2 }],
      },
    }
    const withoutPerformance = { ...fits, assessment: { ...fits.assessment, performance: [] } }
    const unsupported = {
      ...assessed,
      assessment: {
        _tag: "Unsupported", environmentId: "environment", profile: { contextLength: 4096 }, failure,
      },
    }
    expect(() => Schema.decodeUnknownSync(CatalogLocalModelServingStateSchema)(fits)).not.toThrow()
    expect(() => Schema.decodeUnknownSync(CatalogLocalModelServingStateSchema)(withoutPerformance)).toThrow()
    expect(() => Schema.decodeUnknownSync(CatalogLocalModelServingStateSchema)(unsupported)).toThrow()
    expect(() => Schema.decodeUnknownSync(DiscoveredLocalModelServingStateSchema)(unsupported)).not.toThrow()
  })

  it("rejects memory totals that disagree with domain evidence", () => {
    expect(() => Schema.decodeUnknownSync(LocalModelMemorySchema)({
      domains: [{
        memoryDomainId: "system",
        capacityBytes: 10,
        requiredBytes: 4,
        compatibilityReserveBytes: 1,
        remainingBytes: 5,
      }],
      totalRequiredBytes: 5,
      requiredSystemMemoryBytes: 4,
    })).toThrow()
  })
})

describe("LocalModelPreparationSchema", () => {
  it("accepts growing live counts and rejects settled counts above the current total", () => {
    expect(Schema.decodeUnknownSync(LocalModelPreparationSchema)({
      discovery: { complete: false, modelsFound: 5 },
      assessment: { complete: false, settledModels: 12, totalModels: 18 },
    })).toEqual({
      discovery: { complete: false, modelsFound: 5 },
      assessment: { complete: false, settledModels: 12, totalModels: 18 },
    })
    expect(() => Schema.decodeUnknownSync(LocalModelPreparationSchema)({
      discovery: { complete: false, modelsFound: 5 },
      assessment: { complete: false, settledModels: 19, totalModels: 18 },
    })).toThrow()
    expect(() => Schema.decodeUnknownSync(LocalModelPreparationSchema)({
      discovery: { complete: true, modelsFound: 18 },
      assessment: { complete: true, settledModels: 17, totalModels: 18 },
    })).toThrow()
  })
})

describe("ModelReleaseDateSchema", () => {
  it("accepts real ISO calendar dates", () => {
    expect(Schema.decodeUnknownSync(ModelReleaseDateSchema)("2024-02-29")).toBe("2024-02-29")
  })

  it("rejects malformed and impossible dates", () => {
    for (const value of ["2026-8-13", "2026-02-29", "2026-13-01", "0000-01-01"]) {
      expect(() => Schema.decodeUnknownSync(ModelReleaseDateSchema)(value)).toThrow()
    }
  })
})

describe("ModelParameterizationSchema", () => {
  it("accepts valid dense and mixture-of-experts parameterization", () => {
    expect(Schema.decodeUnknownSync(ModelParameterizationSchema)({
      architecture: "dense",
      totalParameters: 8_000_000_000,
    })).toEqual({ architecture: "dense", totalParameters: 8_000_000_000 })
    expect(Schema.decodeUnknownSync(ModelParameterizationSchema)({
      architecture: "mixtureOfExperts",
      totalParameters: 35_000_000_000,
      activeParameters: 3_000_000_000,
    })).toEqual({
      architecture: "mixtureOfExperts",
      totalParameters: 35_000_000_000,
      activeParameters: 3_000_000_000,
    })
  })

  it("rejects nonpositive counts and active counts at or above the total", () => {
    expect(() => Schema.decodeUnknownSync(ModelParameterizationSchema)({
      architecture: "dense",
      totalParameters: 0,
    })).toThrow()
    expect(() => Schema.decodeUnknownSync(ModelParameterizationSchema)({
      architecture: "mixtureOfExperts",
      totalParameters: 3_000_000_000,
      activeParameters: 3_000_000_000,
    })).toThrow()
  })
})

describe("IntelligenceScoreSchema", () => {
  it("accepts whole percentages of the frontier and rejects fractions or values outside 0 to 100", () => {
    for (const valid of [0, 59, 100]) {
      expect(Schema.decodeUnknownSync(IntelligenceScoreSchema)(valid)).toBe(valid)
    }
    for (const invalid of [-1, 101, 58.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => Schema.decodeUnknownSync(IntelligenceScoreSchema)(invalid)).toThrow()
    }
  })
})
