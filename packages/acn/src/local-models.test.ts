import { Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelDownloadIdSchema,
  ModelFileIdSchema,
  ModelPackageIdSchema,
  ModelVariantLabelSchema,
  type ModelBundleDownload,
  type ModelPackageEntry,
  type ModelPackageSource,
  type ServableModelBundle,
} from "@magnitudedev/acn-protocol"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import {
  aggregateAcquisitionState,
  availabilityFromProviderProjection,
  deriveCatalogUpgradeState,
  resolveBundlePresentation,
} from "./local-models"

const standaloneBundle = (
  source: ModelPackageSource,
  path = "model.gguf",
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
      quantization: "Q4_K_M",
      quantizationName: "4-bit",
      architecture: "test",
      maximumContextLength: Option.some(50_000),
      intrinsicModelId: Option.none(),
      intrinsicQualityId: Option.none(),
    },
  },
})

describe("bundle acquisition projection", () => {
  it("keeps one download identity while a speculative bundle completes", () => {
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
    const downloadId = ModelDownloadIdSchema.make("bundle-download")
    const downloads: readonly ModelBundleDownload[] = [{
      id: downloadId,
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
      downloadId,
      completedBytes: 1,
      totalBytes: 2,
    })
  })

  it("derives an upgrade only for a superseded installed catalog target", () => {
    const notInstalled = { _tag: "NotInstalled" as const, completedBytes: 0, totalBytes: 10 }
    expect(deriveCatalogUpgradeState({
      inCatalog: true,
      nativeUpdateAvailable: true,
      currentAcquisitionState: notInstalled,
      desiredAcquisitionState: notInstalled,
      hasPriorCatalogTarget: true,
    })).toEqual({ _tag: "Available" })
  })
})

describe("local model availability", () => {
  const modelId = ProviderModelIdSchema.make("gemma:test")

  it("uses only the provider offering for the canonical model", () => {
    expect(availabilityFromProviderProjection(
      Option.some(modelId),
      new Map([[modelId, { availability: { _tag: "Available" } }]]),
      true,
      Option.none(),
    )).toEqual({ _tag: "Selectable", providerModelId: modelId })
  })

  it("keeps an assessed model installable before publication", () => {
    expect(availabilityFromProviderProjection(
      Option.none(),
      new Map(),
      false,
      Option.none(),
    )).toEqual({ _tag: "Installable" })
  })
})

describe("local model presentation", () => {
  it("preserves catalog presentation over artifact metadata", () => {
    const bundle = standaloneBundle({
      _tag: "HuggingFace",
      repository: "LiquidAI/LFM2.5-2.6B-GGUF",
      revision: "a".repeat(40),
    })
    expect(resolveBundlePresentation(bundle, {
      displayName: "Liquid LFM2.5 2.6B",
      variantLabel: ModelVariantLabelSchema.make("Q6"),
      description: "Curated model description.",
    })).toMatchObject({
      displayName: "Liquid LFM2.5 2.6B",
      variantLabel: "Q6",
    })
  })
})
