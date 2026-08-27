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
  availabilityFromProviderProjection,
  correlatedModelSync,
  deriveModelAcquisitionState,
  installedBundleFields,
  resolveBundlePresentation,
  selectableProviderModelId,
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

    expect(deriveModelAcquisitionState({
      currentInstalled: installedBundleFields(bundle, entries),
      download: correlatedModelSync(downloads, downloadId),
      updateAvailable: false,
      priorInstalled: undefined,
      residencyState: { _tag: "Unloaded" },
    })).toMatchObject({
      _tag: "Installing",
      progress: { stage: "downloading", completedBytes: 1, totalBytes: 2 },
    })
  })

  it("reports an update in progress while the installed version stays usable", () => {
    const installedBundle = standaloneBundle({ _tag: "Local", path: "/models/current.gguf" })
    const desired = {
      ...standaloneBundle({ _tag: "Local", path: "/models/next.gguf" }),
    }
    const desiredBundle: ServableModelBundle = {
      ...desired,
      package: { ...desired.package, id: ModelPackageIdSchema.make("package-next") },
    }
    const entries = new Map<string, ModelPackageEntry>([
      [installedBundle.package.id, {
        package: installedBundle.package,
        localState: { _tag: "Installed", path: "/models/current.gguf", origin: "Magnitude" },
        inspection: { _tag: "Pending" },
        catalogAttribution: { _tag: "NotCatalogTarget" },
      }],
    ])
    const downloads: readonly ModelBundleDownload[] = [{
      id: ModelDownloadIdSchema.make("update-download"),
      bundle: desiredBundle,
      state: {
        _tag: "Downloading",
        stage: "downloading",
        completedBytes: 5,
        totalBytes: 10,
        bytesPerSecond: Option.none(),
      },
    }]
    expect(deriveModelAcquisitionState({
      currentInstalled: installedBundleFields(installedBundle, entries),
      download: downloads[0],
      updateAvailable: true,
      priorInstalled: undefined,
      residencyState: { _tag: "Unloaded" },
    })).toMatchObject({
      _tag: "Updating",
      packages: [{ path: "/models/current.gguf" }],
      progress: { completedBytes: 5, totalBytes: 10 },
    })
  })

  it("reports an available update when a newly desired dependency is missing", () => {
    expect(deriveModelAcquisitionState({
      currentInstalled: undefined,
      download: undefined,
      updateAvailable: true,
      priorInstalled: {
        installedBytes: 10,
        packages: [{
          packageId: ModelPackageIdSchema.make("installed-target"),
          path: "/models/target.gguf",
          origin: "Magnitude",
        }],
      },
      residencyState: { _tag: "Unloaded" },
    })).toMatchObject({
      _tag: "UpdateAvailable",
      packages: [{ path: "/models/target.gguf" }],
    })
  })

  it("returns to NotInstalled after a cancelled download", () => {
    const bundle = standaloneBundle({ _tag: "Local", path: "/models/target.gguf" })
    const downloads: readonly ModelBundleDownload[] = [{
      id: ModelDownloadIdSchema.make("cancelled-download"),
      bundle,
      state: { _tag: "Cancelled", completedBytes: 1, totalBytes: 2 },
    }]
    expect(deriveModelAcquisitionState({
      currentInstalled: undefined,
      download: downloads[0],
      updateAvailable: false,
      priorInstalled: undefined,
      residencyState: { _tag: "Unloaded" },
    })).toEqual({ _tag: "NotInstalled" })
  })

  it("publishes queued installation immediately after ACN admission", () => {
    expect(deriveModelAcquisitionState({
      currentInstalled: undefined,
      download: undefined,
      syncState: { _tag: "Admitting", generation: 1, cancelRequested: false },
      downloadBytes: 10,
      updateAvailable: false,
      priorInstalled: undefined,
      residencyState: { _tag: "Unloaded" },
    })).toEqual({
      _tag: "Installing",
      progress: {
        stage: "queued",
        completedBytes: 0,
        totalBytes: 10,
        bytesPerSecond: Option.none(),
      },
    })
  })

  it("keeps a completed download active until ICN publishes disk truth", () => {
    const bundle = standaloneBundle({ _tag: "Local", path: "/models/target.gguf" })
    const downloadId = ModelDownloadIdSchema.make("completed-download")
    expect(deriveModelAcquisitionState({
      currentInstalled: undefined,
      download: {
        id: downloadId,
        bundle,
        state: { _tag: "Completed" },
      },
      syncState: { _tag: "Correlated", generation: 1, downloadId },
      downloadBytes: 10,
      updateAvailable: false,
      priorInstalled: undefined,
      residencyState: { _tag: "Unloaded" },
    })).toMatchObject({
      _tag: "Installing",
      progress: { stage: "publishing", completedBytes: 10, totalBytes: 10 },
    })
  })

  it("does not borrow a shared-bundle transfer from another model", () => {
    const bundle = standaloneBundle({ _tag: "Local", path: "/models/shared.gguf" })
    const sharedDownload: ModelBundleDownload = {
      id: ModelDownloadIdSchema.make("other-model-download"),
      bundle,
      state: {
        _tag: "Downloading",
        stage: "downloading",
        completedBytes: 1,
        totalBytes: 2,
        bytesPerSecond: Option.none(),
      },
    }

    expect(correlatedModelSync(
      [sharedDownload],
      ModelDownloadIdSchema.make("this-model-download"),
    )).toBeUndefined()
    expect(deriveModelAcquisitionState({
      currentInstalled: undefined,
      download: undefined,
      updateAvailable: false,
      priorInstalled: undefined,
      residencyState: { _tag: "Unloaded" },
    })).toEqual({ _tag: "NotInstalled" })
  })

  it("surfaces an unacknowledged failure until it is acknowledged", () => {
    const bundle = standaloneBundle({ _tag: "Local", path: "/models/target.gguf" })
    const failure = { _tag: "Interrupted" as const }
    const failedDownload = (acknowledged: boolean): ModelBundleDownload => ({
      id: ModelDownloadIdSchema.make("failed-download"),
      bundle,
      state: { _tag: "Failed", completedBytes: 1, totalBytes: 2, failure, acknowledged },
    })
    const inputs = {
      currentInstalled: undefined,
      updateAvailable: false,
      priorInstalled: undefined,
      residencyState: { _tag: "Unloaded" },
    } as const
    expect(deriveModelAcquisitionState({ ...inputs, download: failedDownload(false) }))
      .toEqual({ _tag: "InstallFailed", failure })
    expect(deriveModelAcquisitionState({ ...inputs, download: failedDownload(true) }))
      .toEqual({ _tag: "NotInstalled" })
  })
})

describe("local model availability", () => {
  const modelId = ProviderModelIdSchema.make("gemma:test")

  it("uses only the provider offering for the canonical model", () => {
    expect(availabilityFromProviderProjection(
      Option.some(modelId),
      new Map([[modelId, { availability: { _tag: "Available" } }]]),
    )).toEqual({ _tag: "Selectable", providerModelId: modelId })
  })

  it("keeps an assessed model installable before publication", () => {
    expect(availabilityFromProviderProjection(
      Option.none(),
      new Map(),
    )).toEqual({ _tag: "Installable" })
  })

  it("keeps an installed model preparing while package publication catches up", () => {
    expect(availabilityFromProviderProjection(
      Option.some(modelId),
      new Map([[modelId, { availability: { _tag: "Disabled", reason: "installation_unavailable" } }]]),
    )).toEqual({ _tag: "Preparing", providerModelId: modelId })
  })

  it("keeps an installed fallback selectable while its update is available", () => {
    expect(selectableProviderModelId(modelId, true)).toEqual(Option.some(modelId))
    expect(selectableProviderModelId(modelId, false)).toEqual(Option.none())
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
