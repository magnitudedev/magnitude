import { describe, expect, it } from "vitest"
import { Option } from "effect"
import {
  LocalInferenceMemoryDomainIdSchema,
  ModelFileIdSchema,
  ModelDownloadIdSchema,
  ModelPackageIdSchema,
  ModelServingConfigurationIdSchema,
  ModelVariantLabelSchema,
  CatalogModelIdSchema,
  CatalogVariantIdSchema,
  servableModelBundlePackageIds,
  type LocalInferenceHardware,
  type MemoryAssessment,
  type ModelBundleDownload,
  type ModelPackageEntry,
  type ModelServingConfiguration,
  type ServableModelBundle,
  type ModelPackageSource,
  type RecommendableModel,
} from "@magnitudedev/acn-protocol"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import type * as Generated from "@magnitudedev/icn-protocol/schemas"
import {
  availabilityFromProviderProjection,
  aggregateAcquisitionState,
  deriveCatalogUpgradeState,
  projectLocalModelMemory,
  resolveBundlePresentation,
} from "./local-models"
import {
  configuredModelPackageIds,
  isStandalonePackageCandidate,
  resolveLocalModelConfigurations,
} from "./local-model-configuration-resolver"

const GIB = 1024 ** 3

const standaloneBundle = (
  source: ModelPackageSource,
  path = "model.gguf",
  quantization = "Q4_K - Medium",
  quantizationName = "4-bit",
): Extract<ServableModelBundle, { readonly _tag: "Standalone" }> => ({
  _tag: "Standalone",
  package: {
    id: ModelPackageIdSchema.make("package-test"),
    source,
    files: [{
      id: ModelFileIdSchema.make("file-test"),
      path,
      role: "weights",
      sizeBytes: 1,
      tensorStorageBytes: Option.none(),
      sha256: "a".repeat(64),
    }],
    relationships: [],
    properties: {
      format: "gguf",
      quantization,
      quantizationName,
      architecture: "test",
      maximumContextLength: 50_000,
      intrinsicModelId: Option.none(),
      intrinsicQualityId: Option.none(),
    },
  },
})

const configuration = (
  id: string,
  bundle: ServableModelBundle,
  contextLength: number,
): ModelServingConfiguration => ({
  id: ModelServingConfigurationIdSchema.make(id),
  bundle,
  profile: { contextLength },
})

const catalogModel = (configuration: ModelServingConfiguration): RecommendableModel => ({
  modelId: CatalogModelIdSchema.make(configuration.id),
  variantId: CatalogVariantIdSchema.make("gguf:q4"),
  configuration,
  displayName: configuration.id,
  variantLabel: ModelVariantLabelSchema.make("Q4"),
  description: "test",
  license: "test",
  capabilities: {
    vision: false,
    tools: false,
    structuredOutput: false,
    reasoning: { supported: false, efforts: [], defaultEffort: Option.none() },
  },
  qualityScore: 1,
  qualityScoreProvenance: "test",
  fidelityRank: 1,
  quantizationAware: false,
  qualityEvidence: [],
})

describe("local model configuration resolution", () => {
  it("reserves every package member of a configured speculative bundle", () => {
    const target = standaloneBundle({ _tag: "Local", path: "/models/target.gguf" }).package
    const draft = {
      ...standaloneBundle({ _tag: "Local", path: "/models/draft.gguf" }).package,
      id: ModelPackageIdSchema.make("package-draft"),
    }
    const speculative = configuration("configuration-speculative", {
      _tag: "SpeculativeDecoding",
      target,
      draftSource: { _tag: "Separate", draft },
      method: { _tag: "DSpark" },
    }, 50_000)

    expect(configuredModelPackageIds([speculative])).toEqual(new Set([
      target.id,
      draft.id,
    ]))
    const configuredPackages = configuredModelPackageIds([speculative])
    expect(isStandalonePackageCandidate(target, configuredPackages)).toBe(false)
    expect(isStandalonePackageCandidate(draft, configuredPackages)).toBe(false)
    expect(isStandalonePackageCandidate({
      ...draft,
      id: ModelPackageIdSchema.make("package-draft-role"),
      files: draft.files.map((file) => ({ ...file, role: "draft" as const })),
    }, new Set())).toBe(false)
  })

  it("resolves one configuration per target with catalog over standard precedence", () => {
    const bundle = standaloneBundle({ _tag: "Local", path: "/models" })
    const standard = configuration("configuration-standard", bundle, 50_000)
    const catalog = configuration("configuration-catalog", bundle, 32_000)
    const curated = catalogModel(catalog)
    expect([...resolveLocalModelConfigurations({
      catalog: [],
      effectiveCatalogConfigurations: [],
      installedPackageIds: new Set(servableModelBundlePackageIds(bundle)),
      assessed: new Map([[standard.id, {
        configuration: standard,
        origin: "Standard",
        assessment: { _tag: "Assessing" },
      }]]),
    }).values()].map(({ servingConfiguration }) => servingConfiguration)).toEqual([standard])
    expect([...resolveLocalModelConfigurations({
      catalog: [curated],
      effectiveCatalogConfigurations: [{
        identity: { modelId: curated.modelId, variantId: curated.variantId },
        configuration: catalog,
      }],
      catalogAttributionByPackageId: new Map([[bundle.package.id, {
        _tag: "Attributed",
        modelId: curated.modelId,
        variantId: curated.variantId,
      }]]),
      installedPackageIds: new Set(servableModelBundlePackageIds(bundle)),
      assessed: new Map([
        [standard.id, {
          configuration: standard,
          origin: "Standard",
          assessment: { _tag: "Assessing" },
        }],
        [catalog.id, {
          configuration: catalog,
          origin: "Authored",
          assessment: { _tag: "Assessing" },
        }],
      ]),
    }).values()].map(({ servingConfiguration }) => servingConfiguration)).toEqual([catalog])
  })

  it("replaces generated standalone serving with the current catalog configuration for its target", () => {
    const standalone = standaloneBundle({ _tag: "Local", path: "/models/target.gguf" })
    const standard = configuration("configuration-standard", standalone, 50_000)
    const embedded = configuration("configuration-catalog-embedded", {
      _tag: "SpeculativeDecoding",
      target: standalone.package,
      draftSource: { _tag: "Embedded" },
      method: { _tag: "Mtp" },
    }, 50_000)
    const draft = {
      ...standaloneBundle({ _tag: "Local", path: "/models/draft.gguf" }).package,
      id: ModelPackageIdSchema.make("package-draft"),
    }
    const separate = configuration("configuration-catalog-separate", {
      _tag: "SpeculativeDecoding",
      target: standalone.package,
      draftSource: { _tag: "Separate", draft },
      method: { _tag: "DFlash" },
    }, 50_000)
    const assessed = new Map([[standard.id, {
      configuration: standard,
      origin: "Standard" as const,
      assessment: { _tag: "Assessing" as const },
    }]])

    const embeddedCatalog = catalogModel(embedded)
    const embeddedResolution = resolveLocalModelConfigurations({
      catalog: [embeddedCatalog],
      effectiveCatalogConfigurations: [{
        identity: { modelId: embeddedCatalog.modelId, variantId: embeddedCatalog.variantId },
        configuration: embedded,
      }],
      catalogAttributionByPackageId: new Map([[standalone.package.id, {
        _tag: "Attributed",
        modelId: embeddedCatalog.modelId,
        variantId: embeddedCatalog.variantId,
      }]]),
      installedPackageIds: new Set([standalone.package.id]),
      assessed,
    })
    expect(embeddedResolution.size).toBe(1)
    expect([...embeddedResolution.values()][0]?.servingConfiguration).toEqual(embedded)

    const separateCatalog = catalogModel(separate)
    const separateResolution = resolveLocalModelConfigurations({
      catalog: [separateCatalog],
      effectiveCatalogConfigurations: [{
        identity: { modelId: separateCatalog.modelId, variantId: separateCatalog.variantId },
        configuration: standard,
      }],
      catalogAttributionByPackageId: new Map([[standalone.package.id, {
        _tag: "Attributed",
        modelId: separateCatalog.modelId,
        variantId: separateCatalog.variantId,
      }]]),
      installedPackageIds: new Set([standalone.package.id]),
      assessed,
    })
    expect(separateResolution.size).toBe(1)
    expect([...separateResolution.values()][0]?.servingConfiguration).toEqual(standard)
  })

  it("does not reinterpret removed authored configurations as standard resolutions", () => {
    const bundle = standaloneBundle({ _tag: "Local", path: "/models" })
    const staleCatalog = configuration("configuration-catalog", bundle, 32_000)

    expect(resolveLocalModelConfigurations({
      catalog: [],
      effectiveCatalogConfigurations: [],
      installedPackageIds: new Set(servableModelBundlePackageIds(bundle)),
      assessed: new Map([[staleCatalog.id, {
        configuration: staleCatalog,
        origin: "Authored",
        assessment: { _tag: "Assessing" },
      }]]),
    }).size).toBe(0)
  })

  it("keeps an attributed prior target active until the current catalog bundle is complete", () => {
    const priorBundle = standaloneBundle({ _tag: "Local", path: "/models/prior.gguf" })
    const desiredBase = standaloneBundle({ _tag: "Local", path: "/models/current.gguf" })
    const desiredBundle = {
      ...desiredBase,
      package: { ...desiredBase.package, id: ModelPackageIdSchema.make("package-current") },
    }
    const prior = configuration("configuration-prior", priorBundle, 50_000)
    const desired = configuration("configuration-current", desiredBundle, 50_000)
    const curated = catalogModel(desired)
    const assessed = new Map([
      [prior.id, {
        configuration: prior,
        origin: "Standard" as const,
        assessment: { _tag: "Assessing" as const },
      }],
      [desired.id, {
        configuration: desired,
        origin: "Authored" as const,
        assessment: { _tag: "Assessing" as const },
      }],
    ])
    const attribution = new Map([[priorBundle.package.id, {
      _tag: "Attributed" as const,
      modelId: curated.modelId,
      variantId: curated.variantId,
    }]])

    const before = [...resolveLocalModelConfigurations({
      catalog: [curated],
      effectiveCatalogConfigurations: [{
        identity: { modelId: curated.modelId, variantId: curated.variantId },
        configuration: prior,
      }],
      assessed,
      installedPackageIds: new Set([priorBundle.package.id]),
      catalogAttributionByPackageId: attribution,
    }).values()][0]
    expect(before?.servingConfiguration).toEqual(prior)
    expect(before?.catalogModel).toEqual(Option.some(curated))

    const after = [...resolveLocalModelConfigurations({
      catalog: [curated],
      effectiveCatalogConfigurations: [{
        identity: { modelId: curated.modelId, variantId: curated.variantId },
        configuration: desired,
      }],
      assessed,
      installedPackageIds: new Set([priorBundle.package.id, desiredBundle.package.id]),
      catalogAttributionByPackageId: attribution,
    }).values()][0]
    expect(after?.servingConfiguration).toEqual(desired)
  })

  it("drops a generated resolution when its installed package disappears", () => {
    const bundle = standaloneBundle({ _tag: "Local", path: "/models" })
    const standard = configuration("configuration-standard", bundle, 50_000)

    expect(resolveLocalModelConfigurations({
      catalog: [],
      effectiveCatalogConfigurations: [],
      installedPackageIds: new Set(),
      assessed: new Map([[standard.id, {
        configuration: standard,
        origin: "Standard",
        assessment: { _tag: "Assessing" },
      }]]),
    }).size).toBe(0)
  })
})

describe("bundle download projection", () => {
  it("derives upgrade state only for an installed superseded catalog target", () => {
    const notInstalled = { _tag: "NotInstalled" as const, completedBytes: 0, totalBytes: 10 }
    expect(deriveCatalogUpgradeState({
      inCatalog: true,
      nativeUpdateAvailable: true,
      currentAcquisitionState: notInstalled,
      desiredAcquisitionState: notInstalled,
      hasPriorCatalogTarget: true,
    })).toEqual({ _tag: "Available" })
    const downloading = {
      _tag: "Downloading" as const,
      downloadId: ModelDownloadIdSchema.make("upgrade"),
      stage: "downloading" as const,
      completedBytes: 1,
      totalBytes: 10,
      bytesPerSecond: Option.none(),
    }
    expect(deriveCatalogUpgradeState({
      inCatalog: true,
      nativeUpdateAvailable: true,
      currentAcquisitionState: notInstalled,
      desiredAcquisitionState: downloading,
      hasPriorCatalogTarget: true,
    })).toEqual({
      _tag: "Upgrading",
      downloadId: "upgrade",
      stage: "downloading",
      completedBytes: 1,
      totalBytes: 10,
      bytesPerSecond: Option.none(),
    })
  })

  it("retains one stable identity as packages in a speculative bundle complete", () => {
    const target = standaloneBundle({ _tag: "Local", path: "/models/target.gguf" }).package
    const draft = {
      ...standaloneBundle({ _tag: "Local", path: "/models/draft.gguf" }).package,
      id: ModelPackageIdSchema.make("package-draft"),
    }
    const bundle: ServableModelBundle = {
      _tag: "SpeculativeDecoding",
      target,
      draftSource: { _tag: "Separate", draft },
      method: { _tag: "DSpark" },
    }
    const entries = new Map<string, ModelPackageEntry>([
      [target.id, {
        package: target,
        localState: { _tag: "Installed", path: "/models/target.gguf", origin: "Magnitude" },
        inspection: { _tag: "Pending" },
        catalogAttribution: { _tag: "NotCatalogTarget" },
      }],
      [draft.id, {
        package: draft,
        localState: { _tag: "NotInstalled" },
        inspection: { _tag: "Pending" },
        catalogAttribution: { _tag: "NotCatalogTarget" },
      }],
    ])
    const id = ModelDownloadIdSchema.make("bundle-download")
    const downloads: readonly ModelBundleDownload[] = [{
      id,
      bundle,
      state: {
        _tag: "Downloading",
        stage: "downloading",
        completedBytes: 1,
        totalBytes: 2,
        bytesPerSecond: Option.some(10),
      },
    }]

    expect(aggregateAcquisitionState(bundle, entries, downloads)).toMatchObject({
      _tag: "Downloading",
      downloadId: id,
      completedBytes: 1,
      totalBytes: 2,
    })
    expect(aggregateAcquisitionState(bundle, entries, [{
      ...downloads[0]!,
      state: { _tag: "Completed" },
    }])).toMatchObject({
      _tag: "NotInstalled",
      completedBytes: 1,
      totalBytes: 2,
    })
    expect(aggregateAcquisitionState(bundle, entries, [{
      ...downloads[0]!,
      state: {
        _tag: "Failed",
        completedBytes: 1,
        totalBytes: 2,
        failure: {
          _tag: "InsufficientDiskSpace",
          requiredBytes: 4,
          availableBytes: 3,
        },
        acknowledged: false,
      },
    }])).toMatchObject({
      _tag: "Failed",
      downloadId: id,
      failure: {
        _tag: "InsufficientDiskSpace",
        requiredBytes: 4,
        availableBytes: 3,
      },
    })
  })
})

describe("local model availability", () => {
  const providerModelIds = [ProviderModelIdSchema.make("test-configuration")]

  it("reports provider publication as preparing until it matches the package snapshot", () => {
    expect(availabilityFromProviderProjection(
      providerModelIds[0],
      new Map([[providerModelIds[0]!, {
        availability: { _tag: "Disabled", reason: "incompatible_runtime" },
      }]]),
      false,
      Option.none(),
    )).toEqual({ _tag: "Preparing", providerModelId: providerModelIds[0] })
  })

  it("keeps an assessed configuration installable before it has an offering", () => {
    expect(availabilityFromProviderProjection(
      undefined,
      new Map(),
      false,
      Option.none(),
    )).toEqual({ _tag: "Installable" })
  })

  it("exposes an authoritative current provider incompatibility", () => {
    expect(availabilityFromProviderProjection(
      providerModelIds[0],
      new Map([[providerModelIds[0]!, {
        availability: { _tag: "Disabled", reason: "incompatible_runtime" },
      }]]),
      true,
      Option.none(),
    )).toEqual({
      _tag: "Unavailable",
      providerModelId: Option.some(providerModelIds[0]),
      failure: {
        code: "incompatible_runtime",
        message: "This model configuration is not available to the local runtime",
        retryable: true,
      },
    })
  })

  it("uses only the provider offering for the exact configuration", () => {
    const otherProviderModelId = ProviderModelIdSchema.make("other-configuration")
    expect(availabilityFromProviderProjection(
      providerModelIds[0],
      new Map([
        [providerModelIds[0]!, {
          availability: { _tag: "Disabled", reason: "insufficient_resources" },
        }],
        [otherProviderModelId, { availability: { _tag: "Available" } }],
      ]),
      true,
      Option.none(),
    )).toMatchObject({
      _tag: "Unavailable",
      failure: { code: "insufficient_resources" },
    })
  })
})

describe("local model current memory headroom", () => {
  const memoryDomainId = LocalInferenceMemoryDomainIdSchema.make("system")
  const hardware: LocalInferenceHardware = {
    platform: "MacOS",
    architecture: "Arm64",
    productName: Option.none(),
    processor: Option.none(),
    logicalCores: 8,
    totalSystemMemoryBytes: 64 * GIB,
    availableSystemMemoryBytes: 8 * GIB,
    systemAllocationCapacityBytes: 64 * GIB,
    systemAllocationHeadroomBytes: 8 * GIB,
    abortReserveBytes: 4 * GIB,
    accelerators: [],
    memoryDomains: [{
      memoryDomainId,
      kind: "UnifiedMemory",
      totalBytes: 64 * GIB,
      stableCapacityBytes: 58 * GIB,
      availableBytes: Option.some(8 * GIB),
      sharesSystemMemory: true,
    }],
  }
  const assessment: readonly MemoryAssessment[] = [{
    memoryDomainId,
    capacityBytes: 64 * GIB,
    requiredBytes: 32 * GIB,
    compatibilityReserveBytes: 6 * GIB,
    remainingBytes: 26 * GIB,
  }]
  const instances = (
    lifecycle: object,
  ): Generated.ModelInstancesSnapshot => ({
    revision: 1,
    instances: [{
      id: "instance-test",
      configurationId: "configuration-test",
      lifecycle,
    }],
  } as unknown as Generated.ModelInstancesSnapshot)

  it("does not infer insufficient headroom while loading residency is indeterminate", () => {
    const memory = projectLocalModelMemory(assessment, hardware, instances({
      _tag: "Loading",
      stage: "loading",
      progress: Option.some(0.5),
      plannedAllocation: Option.some({
        contextWindowTokens: 100_000,
        parallelSequences: 4,
        physicalContextTokens: 400_000,
        requiredSystemMemoryBytes: 40 * GIB,
      }),
    }))

    expect(memory.currentHeadroomState).toEqual({ _tag: "NotObserved" })
  })

  it("does not infer insufficient headroom while a planned allocation is stopping", () => {
    const memory = projectLocalModelMemory(assessment, hardware, instances({
      _tag: "Stopping",
      reason: "user_stop",
      allocation: {
        _tag: "Planned",
        allocation: Option.some({
          contextWindowTokens: 100_000,
          parallelSequences: 4,
          physicalContextTokens: 400_000,
          requiredSystemMemoryBytes: 40 * GIB,
        }),
      },
    }))

    expect(memory.currentHeadroomState).toEqual({ _tag: "NotObserved" })
  })

  it("uses exact resident allocation when deciding current headroom", () => {
    const memory = projectLocalModelMemory(assessment, hardware, instances({
      _tag: "Ready",
      allocation: {
        contextWindowTokens: 100_000,
        parallelSequences: 4,
        physicalContextTokens: 400_000,
        memoryDomains: [{
          memoryDomainId,
          modelBytes: 32 * GIB,
          contextBytes: 8 * GIB,
          computeBytes: 0,
          auxiliaryBytes: 0,
        }],
      },
    }))

    expect(memory.currentHeadroomState._tag).toBe("Sufficient")
  })
})

describe("local model presentation", () => {
  const huggingFaceBundle = standaloneBundle({
    _tag: "HuggingFace",
    repository: "LiquidAI/LFM2.5-2.6B-GGUF",
    revision: "a".repeat(40),
  })
  const curated = {
    displayName: "Liquid LFM2.5 2.6B",
    variantLabel: ModelVariantLabelSchema.make("Q4"),
    description: "Curated model description.",
  }

  it("preserves curated metadata for an exact installed bundle", () => {
    expect(resolveBundlePresentation(
      huggingFaceBundle,
      curated,
    )).toEqual({
      displayName: "Liquid LFM2.5 2.6B",
      variantLabel: "Q4",
      description: "Curated model description.",
      license: Option.none(),
      quantization: "Q4_K - Medium",
      precisionLabel: "4-bit",
    })
  })

  it("does not transfer curated metadata to another bundle", () => {
    expect(resolveBundlePresentation(
      huggingFaceBundle,
      undefined,
    )).toEqual({
      displayName: "LFM2.5-2.6B-GGUF",
      variantLabel: "Q4_K - Medium",
      description: "",
      license: Option.none(),
      quantization: "Q4_K - Medium",
      precisionLabel: "4-bit",
    })
  })

  it("derives an uncurated local bundle name from its file", () => {
    expect(resolveBundlePresentation(
      standaloneBundle({ _tag: "Local", path: "/models" }, "local-model.gguf"),
      undefined,
    )).toEqual({
      displayName: "local-model.gguf",
      variantLabel: "Q4_K - Medium",
      description: "",
      license: Option.none(),
      quantization: "Q4_K - Medium",
      precisionLabel: "4-bit",
    })
  })

  it("presents only the target quantization for speculative bundles", () => {
    const targetBundle = standaloneBundle(
      { _tag: "Local", path: "/models" },
      "target.gguf",
      "Q4_K - Medium",
      "4-bit",
    )
    const draftBundle = standaloneBundle(
      { _tag: "Local", path: "/models" },
      "draft.gguf",
      "Q8_0",
      "8-bit",
    )
    if (targetBundle._tag !== "Standalone" || draftBundle._tag !== "Standalone") {
      throw new Error("test bundles must be standalone packages")
    }

    expect(resolveBundlePresentation({
      _tag: "SpeculativeDecoding",
      target: targetBundle.package,
      draftSource: { _tag: "Separate", draft: draftBundle.package },
      method: { _tag: "DSpark" },
    }, curated)).toMatchObject({
      quantization: "Q4_K - Medium",
      precisionLabel: "4-bit",
    })
  })
})
