import { describe, expect, it } from "vitest"
import { Option, Schema } from "effect"
import {
  CatalogIntelligenceSchema,
  CatalogBaseIdSchema,
  CatalogVariantIdSchema,
  LocalModelMemorySchema,
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
    intelligence: { score: 1, provenance: {
      kind: "artificialAnalysisIntelligenceIndex",
      methodologyVersion: "test",
      asOfDate: "2026-08-29",
      url: "https://example.com/model",
    } },
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
    const ambiguous = Schema.decodeUnknownSync(LocalModelSchema)({
      _tag: "Discovered",
      modelId: "hf:owner/repository/model.gguf",
      presentation: catalogModel.presentation,
      state: {
        _tag: "Ambiguous",
        failure: { code: "ambiguous", message: "Ambiguous", retryable: false },
      },
    })
    expect(ambiguous._tag).toBe("Discovered")
    expect(() => Schema.decodeUnknownSync(LocalModelSchema)({
      ...ambiguous,
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
    const ambiguous = Schema.decodeUnknownSync(LocalModelSchema)({
      ...discovered,
      state: {
        _tag: "Ambiguous",
        failure,
        servingState: { _tag: "Failed", profile: { contextLength: 4096 }, failure },
      },
    })
    expect(ambiguous._tag === "Discovered" && ambiguous.state).toEqual({
      _tag: "Ambiguous",
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
          systemUseState: { _tag: "NotObserved" }, currentHeadroomState: { _tag: "NotObserved" },
        },
        performance: [{
          contextTokens: 4096, lowerTokensPerSecond: 1, estimatedTokensPerSecond: 2,
          upperTokensPerSecond: 3, confidence: "high",
        }],
      },
    } as const
    expect(() => Schema.decodeUnknownSync(LocalModelServingStateSchema)(assessed)).not.toThrow()
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
      systemUseState: { _tag: "NotObserved" },
      currentHeadroomState: { _tag: "NotObserved" },
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

describe("CatalogIntelligenceSchema", () => {
  it("preserves direct and estimated provenance as distinct variants", () => {
    const direct = {
      score: 20.4,
      provenance: {
        kind: "artificialAnalysisIntelligenceIndex",
        methodologyVersion: "4.1.1",
        asOfDate: "2026-08-26",
        url: "https://artificialanalysis.ai/models/qwen3-5-4b",
      },
    }
    const estimate = {
      score: 7.3,
      provenance: {
        kind: "estimate",
        target: "artificialAnalysisIntelligenceIndex",
        methodologyVersion: "4.1.1",
        asOfDate: "2026-08-26",
        confidence: "moderate",
        methodology: "Compared with the exact parent model.",
        evidenceUrls: ["https://example.com/evidence"],
      },
    }
    expect(Schema.decodeUnknownSync(CatalogIntelligenceSchema)(direct)).toEqual(direct)
    expect(Schema.decodeUnknownSync(CatalogIntelligenceSchema)(estimate)).toEqual(estimate)
  })

  it("rejects malformed dates, non-HTTPS evidence, and empty estimate evidence", () => {
    const provenance = {
      kind: "estimate",
      target: "artificialAnalysisIntelligenceIndex",
      methodologyVersion: "4.1.1",
      asOfDate: "2026-08-26",
      confidence: "low",
      methodology: "Peer comparison.",
      evidenceUrls: ["https://example.com/evidence"],
    }
    for (const invalid of [
      { ...provenance, asOfDate: "2026-02-29" },
      { ...provenance, evidenceUrls: ["http://example.com/evidence"] },
      { ...provenance, evidenceUrls: [] },
    ]) {
      expect(() => Schema.decodeUnknownSync(CatalogIntelligenceSchema)({
        score: 1,
        provenance: invalid,
      })).toThrow()
    }
  })
})
