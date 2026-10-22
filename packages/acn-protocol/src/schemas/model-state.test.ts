import { describe, expect, it } from "vitest"
import { Option, Schema } from "effect"
import {
  CatalogBaseIdSchema,
  CatalogVariantIdSchema,
  CatalogLocalModelServingStateSchema,
  CatalogSupportSchema,
  CandidateInspectionSchema,
  CustomLocalModelServingStateSchema,
  LocalModelsStateSchema,
  UnsupportedReasonSchema,
  IntelligenceScoreSchema,
  LocalModelMemorySchema,
  LocalModelPreparationSchema,
  LocalModelSchema,
  LocalModelServingStateSchema,
  ModelCapabilitiesSchema,
  ModelIdSchema,
  ModelLocationSchema,
  modelLocationState,
  ModelParameterizationSchema,
  ModelReleaseDateSchema,
  parseModelId,
  sameModelId,
} from "./model-state"

const localId = "local:gemma-4-12b-instruct-q8_0:41ad09c2e7f3"

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
    aliases: [],
  },
  acquisitionState: { _tag: "NotInstalled" },
  missingComponents: [],
  storedComponents: [],
  drafterWithheld: false,
  servingState: {
    _tag: "Failed",
    profile: { contextLength: 4096 },
    failure: { code: "unavailable", message: "Unavailable", retryable: true },
  },
} as const

const foundModel = {
  _tag: "Custom",
  modelId: "hf:owner/repository/model.gguf",
  origin: { _tag: "Found", source: { _tag: "HuggingFaceCache" } },
  drafterPending: false,
  drafterChoice: { _tag: "Auto" },
  presentation: {
    displayName: "Model",
    variantLabel: "Q4_K_M",
    sourceLabel: "Hugging Face cache",
    sourceUrls: [],
  },
  acquisitionState: {
    _tag: "External",
    installation: { _tag: "Resolved", installedBytes: 1, primaryPath: "/model.gguf", ownership: "External" },
    residencyState: { _tag: "Unloaded" },
  },
  servingState: { _tag: "Assessing", profile: { contextLength: 4096 } },
  storageBytes: 1,
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
  it("accepts canonical catalog, Hugging Face, and local identities", () => {
    expect(Schema.decodeUnknownSync(CatalogBaseIdSchema)("qwen3.5-4b")).toBe("qwen3.5-4b")
    expect(Schema.decodeUnknownSync(CatalogVariantIdSchema)("gguf:q4")).toBe("gguf:q4")
    for (const id of [
      "qwen3.5-4b:gguf:q4",
      "hf:owner/repository/model-q4.gguf",
      "hf:owner/repository/subdirectory/model-00001-of-00002.gguf",
      localId,
      "local:my.finetune_v2:9f2c1a7b3e04",
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
    expect(parseModelId(Schema.decodeUnknownSync(ModelIdSchema)(localId))).toEqual({
      _tag: "Local",
      label: "gemma-4-12b-instruct-q8_0",
      hash: "41ad09c2e7f3",
    })
  })

  it("names a local model by its hash alone, so an ID stored under an older label is the same model", () => {
    expect(sameModelId(localId, "local:gemma-4-12b-it-q8_0:41ad09c2e7f3")).toBe(true)
    expect(sameModelId(localId, "local:gemma-4-12b-instruct-q8_0:41ad09c2e7f4")).toBe(false)
    expect(sameModelId("hf:owner/repository/model.gguf", "hf:owner/repository/model.gguf")).toBe(true)
    expect(sameModelId("hf:owner/repository/model.gguf", "hf:owner/repository/other.gguf")).toBe(false)
    expect(sameModelId("model:gguf:q4", "model:gguf:q4")).toBe(true)
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
      "local:abc",
      // The pre-release form: a whole content digest with no label.
      `local:${"a".repeat(32)}${"0123456789abcdef".repeat(2)}`,
      "local:gemma:41AD09C2E7F3",
      "local:gemma:41ad09c2e7f",
      "local:gemma:41ad09c2e7f30",
      "local:Gemma-Q8_0:41ad09c2e7f3",
      "local:gemma q8:41ad09c2e7f3",
      "local::41ad09c2e7f3",
    ]) expect(() => Schema.decodeUnknownSync(ModelIdSchema)(id)).toThrow()
  })
})

describe("LocalModelSchema invariants", () => {
  it("carries the components a catalog model lacks apart from its acquisition state", () => {
    const decoded = Schema.decodeUnknownSync(LocalModelSchema)({
      ...catalogModel,
      missingComponents: [{ component: "Projector", downloadBytes: 1_034_945_421 }],
    })
    expect(decoded._tag === "Catalog" && decoded.missingComponents)
      .toEqual([{ component: "Projector", downloadBytes: 1_034_945_421 }])
    expect(() => Schema.decodeUnknownSync(LocalModelSchema)({
      ...catalogModel,
      missingComponents: [{ component: "Weights", downloadBytes: 1 }],
    })).toThrow()
  })
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

  it("requires domain-specific catalog and custom-model facts", () => {
    const { catalogData: _catalogData, ...catalogWithoutData } = catalogModel
    expect(() => Schema.decodeUnknownSync(LocalModelSchema)({
      ...catalogWithoutData,
    })).toThrow()
    expect(() => Schema.decodeUnknownSync(LocalModelSchema)({
      ...catalogModel,
      modelId: "hf:owner/repository/model.gguf",
    })).toThrow()
    const found = Schema.decodeUnknownSync(LocalModelSchema)(foundModel)
    expect(found._tag).toBe("Custom")
    expect(() => Schema.decodeUnknownSync(LocalModelSchema)({
      ...foundModel,
      modelId: "model:gguf:q4",
    })).toThrow()
  })

  it("ties origin to identity and to who owns the files", () => {
    for (const source of [{ _tag: "Ollama" }, { _tag: "LmStudio" }, { _tag: "Folder", path: "/models" }]) {
      expect(() => Schema.decodeUnknownSync(LocalModelSchema)({
        ...foundModel,
        modelId: localId,
        origin: { _tag: "Found", source },
      })).not.toThrow()
      // Only a location that records a file's Hub origin gives it a Hugging Face ID.
      expect(() => Schema.decodeUnknownSync(LocalModelSchema)({
        ...foundModel,
        origin: { _tag: "Found", source },
      })).toThrow()
    }
    for (const modelId of [localId, foundModel.modelId]) {
      expect(() => Schema.decodeUnknownSync(LocalModelSchema)({
        ...foundModel,
        modelId,
        origin: { _tag: "Found", source: { _tag: "LlamaCpp" } },
      })).not.toThrow()
    }
    expect(() => Schema.decodeUnknownSync(LocalModelSchema)({
      ...foundModel,
      modelId: localId,
    })).toThrow()
    expect(() => Schema.decodeUnknownSync(LocalModelSchema)({
      ...foundModel,
      origin: { _tag: "Added", repository: "owner/repository", commit: "abc" },
    })).toThrow()
    expect(() => Schema.decodeUnknownSync(LocalModelSchema)({
      ...foundModel,
      origin: { _tag: "Added", repository: "owner/repository", commit: "abc" },
      acquisitionState: { _tag: "Installing", progress: { stage: "downloading", completedBytes: 1, totalBytes: 2 } },
    })).not.toThrow()
  })

  it("keeps a custom model's failed serving state tied to a profile", () => {
    const failure = { code: "failed", message: "Failed", retryable: true } as const
    expect(() => Schema.decodeUnknownSync(LocalModelSchema)({
      ...foundModel,
      servingState: { _tag: "Failed", failure },
    })).toThrow()
    expect(() => Schema.decodeUnknownSync(LocalModelSchema)({
      ...foundModel,
      servingState: { _tag: "Failed", profile: { contextLength: 4096 }, failure },
    })).not.toThrow()
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
        performance: {
          _tag: "Estimated",
          samples: [{ contextTokens: 4096, estimatedTokensPerSecond: 2 }],
        },
      },
    } as const
    expect(() => Schema.decodeUnknownSync(LocalModelServingStateSchema)(assessed)).not.toThrow()
  })

  it("requires estimates for estimated performance and admits Unsupported only for custom models", () => {
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
        performance: {
          _tag: "Estimated",
          samples: [{ contextTokens: 4096, estimatedTokensPerSecond: 2 }],
        },
      },
      rankingScores: { intelligence: 0.5, speed: 0.1, fidelity: 0.5 },
    }
    const withoutPerformance = {
      ...fits,
      assessment: { ...fits.assessment, performance: { _tag: "Estimated", samples: [] } },
    }
    const pending = {
      ...fits,
      assessment: {
        ...fits.assessment,
        performance: { _tag: "Pending", measurement: { _tag: "Failed", retryAt: 1_000 } },
      },
      rankingScores: { intelligence: 0.5, fidelity: 0.5 },
    }
    const unavailable = {
      ...fits,
      assessment: {
        ...fits.assessment,
        performance: {
          _tag: "Unavailable",
          failure: { code: "performance_unavailable", message: "hung", retryable: false },
        },
      },
    }
    const unsupported = {
      ...assessed,
      assessment: {
        _tag: "Unsupported",
        environmentId: "environment",
        profile: { contextLength: 4096 },
        reasons: [{ _tag: "Architecture", architecture: "deepseek4" }],
        detail: "unrecognized architecture deepseek4",
      },
    }
    expect(() => Schema.decodeUnknownSync(CatalogLocalModelServingStateSchema)(fits)).not.toThrow()
    expect(() => Schema.decodeUnknownSync(CatalogLocalModelServingStateSchema)(withoutPerformance)).toThrow()
    expect(() => Schema.decodeUnknownSync(CatalogLocalModelServingStateSchema)(pending)).not.toThrow()
    expect(() => Schema.decodeUnknownSync(CatalogLocalModelServingStateSchema)(unavailable)).not.toThrow()
    expect(() => Schema.decodeUnknownSync(CatalogLocalModelServingStateSchema)(unsupported)).toThrow()
    expect(() => Schema.decodeUnknownSync(CustomLocalModelServingStateSchema)(unsupported)).not.toThrow()
    const { profile: _profile, ...unreadable } = unsupported.assessment
    expect(Schema.decodeUnknownSync(CustomLocalModelServingStateSchema)({
      ...unsupported,
      assessment: { ...unreadable, reasons: [{ _tag: "Unreadable" }] },
    })).toMatchObject({ assessment: { _tag: "Unsupported", profile: Option.none() } })
    expect(() => Schema.decodeUnknownSync(CustomLocalModelServingStateSchema)({
      ...unsupported,
      assessment: { ...unsupported.assessment, reasons: [] },
    })).toThrow()
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

describe("UnsupportedReasonSchema", () => {
  it("decodes every known case and round-trips it", () => {
    for (const reason of [
      { _tag: "WeightTypes", types: [{ ggmlType: "Q2_K", tensorCount: 2 }, { ggmlType: "IQ1_M", tensorCount: 40 }] },
      { _tag: "Tokenizer", model: "gpt2", pre: "qwen2" },
      { _tag: "KernelDomain", backend: "metal", entry: "matmul", statics: [["head_dim", 72]] },
      { _tag: "MissingMetadata", key: "dflash.attention.causal" },
      { _tag: "TargetMismatch", detail: "drafter hidden size 2048 does not match 4096" },
      { _tag: "TiedOutputHead" },
      { _tag: "DraftArchitecture", architecture: "nemotron_h_moe" },
      { _tag: "Unreadable" },
      { _tag: "UnsupportedFormat" },
    ]) {
      const decoded = Schema.decodeUnknownSync(UnsupportedReasonSchema)(reason)
      expect(decoded._tag).toBe(reason._tag)
      expect(Schema.decodeUnknownSync(UnsupportedReasonSchema)(Schema.encodeSync(UnsupportedReasonSchema)(decoded)))
        .toEqual(decoded)
    }
    expect(Schema.decodeUnknownSync(UnsupportedReasonSchema)({ _tag: "Tokenizer", model: "gpt2" }))
      .toEqual({ _tag: "Tokenizer", model: "gpt2", pre: Option.none() })
  })

  it("decodes a case added by a newer engine as Unknown with its detail", () => {
    const decoded = Schema.decodeUnknownSync(UnsupportedReasonSchema)({
      _tag: "SlidingWindow", window: 4096, detail: "sliding window attention",
    })
    expect(decoded).toEqual({ _tag: "Unknown", kind: "SlidingWindow", detail: "sliding window attention" })
    expect(Schema.decodeUnknownSync(UnsupportedReasonSchema)(Schema.encodeSync(UnsupportedReasonSchema)(decoded)))
      .toEqual(decoded)
    expect(Schema.decodeUnknownSync(UnsupportedReasonSchema)({ _tag: "Other" }))
      .toEqual({ _tag: "Unknown", kind: "Other", detail: "" })
  })

  it("still rejects a malformed known case", () => {
    expect(() => Schema.decodeUnknownSync(UnsupportedReasonSchema)({ _tag: "Architecture" })).toThrow()
    expect(() => Schema.decodeUnknownSync(UnsupportedReasonSchema)({ _tag: "WeightTypes", types: [] })).toThrow()
  })
})

describe("LocalModelsStateSchema", () => {
  it("lists unsupported local models apart from callable rows", () => {
    const state = Schema.decodeUnknownSync(LocalModelsStateSchema)({
      preparation: {
        discovery: { complete: true, modelsFound: 1 },
        assessment: { complete: true, settledModels: 1, totalModels: 1 },
      },
      models: [foundModel],
      unsupported: [{
        displayName: "DeepSeek V4 Flash",
        origin: { _tag: "Found", source: { _tag: "Folder", path: "/Volumes/models" } },
        path: "/Volumes/models/deepseek.gguf",
        reasons: [{ _tag: "Architecture", architecture: "deepseek4" }],
        detail: "",
      }],
      hidden: [],
    })
    expect(state.unsupported).toHaveLength(1)
  })

  it("lists hidden custom models apart from callable rows, never twice", () => {
    const base = {
      preparation: {
        discovery: { complete: true, modelsFound: 1 },
        assessment: { complete: true, settledModels: 1, totalModels: 1 },
      },
      unsupported: [],
    }
    const state = Schema.decodeUnknownSync(LocalModelsStateSchema)({ ...base, models: [], hidden: [foundModel] })
    expect(state.hidden).toHaveLength(1)
    expect(() => Schema.decodeUnknownSync(LocalModelsStateSchema)({ ...base, models: [foundModel], hidden: [foundModel] }))
      .toThrow()
  })
})

describe("CandidateInspectionSchema", () => {
  const version = (label: string, recommended: boolean) => ({
    label,
    modelId: `hf:owner/repository/model-${label}.gguf`,
    files: [`model-${label}.gguf`],
    sizeBytes: 10,
    downloadBytes: 10,
    recommended,
  })
  const inspection = {
    source: { _tag: "HuggingFace", repository: "owner/repository", commit: "abc" },
    displayName: "Model",
    versions: [version("Q4_K_M", true), version("Q8_0", false)],
  }

  it("accepts an unreadable version whose assessment has no serving profile", () => {
    const unreadable = {
      _tag: "Unsupported",
      environmentId: "environment",
      reasons: [{ _tag: "Unreadable" }],
      detail: "file is shorter than its header declares",
    }
    const decoded = Schema.decodeUnknownSync(CandidateInspectionSchema)({
      ...inspection,
      versions: [version("Q4_K_M", true), { ...version("Q8_0", false), assessment: unreadable }],
    })
    const assessment = Option.getOrThrow(decoded.versions[1]!.assessment)
    expect(assessment._tag === "Unsupported" && assessment.profile).toEqual(Option.none())
    const encoded = Schema.encodeSync(CandidateInspectionSchema)(decoded)
    expect(encoded.versions[1]!.assessment).toEqual(unreadable)
    expect(() => Schema.decodeUnknownSync(CandidateInspectionSchema)({
      ...inspection,
      versions: [version("Q4_K_M", true), { ...version("Q8_0", false), assessment: { ...unreadable, profile: { contextLength: 0 } } }],
    })).toThrow()
  })

  it("accepts a repository step with one recommendation and a listed selection", () => {
    expect(() => Schema.decodeUnknownSync(CandidateInspectionSchema)({
      ...inspection,
      selectedVersion: "Q8_0",
    })).not.toThrow()
  })

  it("rejects an unlisted selection and two recommendations", () => {
    expect(() => Schema.decodeUnknownSync(CandidateInspectionSchema)({
      ...inspection,
      selectedVersion: "Q2_K",
    })).toThrow()
    expect(() => Schema.decodeUnknownSync(CandidateInspectionSchema)({
      ...inspection,
      versions: [version("Q4_K_M", true), version("Q8_0", true)],
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

describe("ModelLocationSchema", () => {
  const decode = Schema.decodeUnknownEither(ModelLocationSchema)
  const ollama = {
    kind: "Ollama",
    paths: [
      { path: "/home/ada/.ollama/models", state: "NotFound" },
      { path: "/var/lib/ollama", state: "Found" },
      { path: "/usr/share/ollama/.ollama/models", state: "Unreadable" },
    ],
    enabled: true,
    automatic: true,
    state: "Found",
  }

  it("summarizes a location's directories: found if any is found, else unreadable if any is", () => {
    expect(modelLocationState([{ path: "/a", state: "NotFound" }, { path: "/b", state: "Found" }])).toBe("Found")
    expect(modelLocationState([{ path: "/a", state: "NotFound" }, { path: "/b", state: "Unreadable" }])).toBe("Unreadable")
    expect(modelLocationState([{ path: "/a", state: "NotFound" }])).toBe("NotFound")
  })

  it("accepts an automatic location with several directories and their own states", () => {
    expect(decode(ollama)._tag).toBe("Right")
  })

  it("rejects a summary that disagrees with the directories, no directories, and a folder with several", () => {
    expect(decode({ ...ollama, state: "Unreadable" })._tag).toBe("Left")
    expect(decode({ ...ollama, paths: [] })._tag).toBe("Left")
    expect(decode({
      kind: "Folder",
      paths: [{ path: "/a", state: "Found" }, { path: "/b", state: "Found" }],
      enabled: true,
      automatic: false,
      state: "Found",
    })._tag).toBe("Left")
  })
})

describe("ModelCapabilitiesSchema reasoning", () => {
  const capabilities = (reasoning: unknown) => Schema.decodeUnknownEither(ModelCapabilitiesSchema)({
    vision: false, tools: false, structuredOutput: false, reasoning,
  })

  it("accepts a non-reasoning model that takes an explicit none", () => {
    expect(capabilities({ supported: false, efforts: ["none"], defaultEffort: "none" })._tag).toBe("Right")
    expect(capabilities({ supported: false, efforts: [] })._tag).toBe("Right")
  })

  it("ties supported to an enabled effort and the default to the effort set", () => {
    expect(capabilities({ supported: true, efforts: ["none", "high"], defaultEffort: "high" })._tag).toBe("Right")
    expect(capabilities({ supported: true, efforts: ["high"], defaultEffort: "high" })._tag).toBe("Right")
    expect(capabilities({ supported: true, efforts: ["none"], defaultEffort: "none" })._tag).toBe("Left")
    expect(capabilities({ supported: false, efforts: ["none", "high"], defaultEffort: "high" })._tag).toBe("Left")
    expect(capabilities({ supported: false, efforts: ["none"] })._tag).toBe("Left")
    expect(capabilities({ supported: false, efforts: [], defaultEffort: "none" })._tag).toBe("Left")
    expect(capabilities({ supported: true, efforts: ["high", "high"], defaultEffort: "high" })._tag).toBe("Left")
    expect(capabilities({ supported: true, efforts: ["none", "high"], defaultEffort: "low" })._tag).toBe("Left")
  })
})
