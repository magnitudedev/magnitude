import { Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { UnsupportedReason, type ModelAssessmentsSnapshot } from "@magnitudedev/icn-protocol/schemas"
import { customModel, projectLocalModels } from "./local-models"
import { currentDownload, customModelDownloads } from "./custom-model-downloads"
import type { LocalModelSourcesState } from "./local-model-sources"
import {
  addedId,
  customModelsConfig,
  customSource,
  drafterDownload,
  drafterReference,
  foundId,
  importedId,
  nativeCustomModel,
  nativeDownload,
  targetDownload,
} from "./custom-models.test-support"

const capabilities = {
  vision: false,
  tools: true,
  structuredOutput: true,
  reasoning: { supported: false, efforts: [], defaultEffort: Option.none() },
}

const assessed = (assessment: Record<string, unknown>) => ({
  _tag: "Assessed" as const,
  assessment: assessment as never,
  capabilities: capabilities as never,
  drafter: undefined,
})

const fits = assessed({
  _tag: "Fits",
  assessmentId: "assessment",
  environmentId: "environment",
  profile: { contextLength: 32_768 },
  memory: { domains: [], totalRequiredBytes: 0, requiredSystemMemoryBytes: 0 },
  performance: [{ contextTokens: 32_768, estimatedTokensPerSecond: 30 }],
})

const unsupported = assessed({
  _tag: "Unsupported",
  environmentId: "environment",
  profile: Option.some({ contextLength: 32_768 }),
  reasons: [{ _tag: "Architecture", architecture: "deepseek4" }],
  detail: "",
})

const inputs = (entry = customSource(nativeCustomModel()), overrides: Partial<Parameters<typeof customModel>[0]> = {}) => ({
  entry,
  assessment: fits,
  instances: [],
  related: [],
  catalog: [],
  drafterChoice: { _tag: "Auto" } as const,
  ...overrides,
})

const attachedDrafter = {
  method: { _tag: "DFlash" },
  reference: drafterReference,
  state: { _tag: "Ready" },
  updateAvailable: false,
}

describe("custom model rows", () => {
  it("projects a found model as external, assessed, and unranked with its source tag", () => {
    const row = Option.getOrThrow(customModel(inputs()))
    expect(row.origin).toEqual({ _tag: "Found", source: { _tag: "LmStudio" } })
    expect(row.presentation.sourceLabel).toBe("LM Studio")
    expect(row.presentation.variantLabel).toBe("Q4_K_M")
    expect(row.presentation.sourceUrls).toEqual(["https://huggingface.co/other/found-GGUF"])
    expect(row.acquisitionState._tag).toBe("External")
    expect(row.servingState._tag).toBe("Assessed")
    expect(row.servingState).not.toHaveProperty("rankingScores")
  })

  it("shows the license ICN read for a custom model, and none when it read nothing", () => {
    const licensed = Option.getOrThrow(customModel(inputs(customSource(nativeCustomModel({ license: "apache-2.0" })))))
    expect(licensed.presentation.license).toEqual(Option.some("apache-2.0"))
    expect(Option.getOrThrow(customModel(inputs())).presentation.license).toEqual(Option.none())
    expect(Option.getOrThrow(customModel(inputs(customSource(nativeCustomModel({ license: "  " }))))).presentation.license)
      .toEqual(Option.none())
  })

  it("names a model by ICN's display name and the quantization its file names", () => {
    const row = Option.getOrThrow(customModel(inputs(customSource(nativeCustomModel({
      displayName: "MiniCPM5 1B",
      variantLabel: "Q3_K_M",
    })))))
    expect(row.presentation.displayName).toBe("MiniCPM5 1B")
    expect(row.presentation.variantLabel).toBe("Q3_K_M")
  })

  it("lists unsupported external models apart and keeps an unsupported added model removable", () => {
    expect(customModel(inputs(undefined, { assessment: unsupported }))).toEqual(Option.none())
    const added = customSource(nativeCustomModel({
      modelId: addedId,
      origin: { _tag: "Added", repository: "owner/model-GGUF", commit: "c0ffee" },
    }))
    const row = Option.getOrThrow(customModel(inputs(added, { assessment: unsupported })))
    expect(row.servingState._tag === "Assessed" && row.servingState.assessment._tag).toBe("Unsupported")
  })

  it("projects an added model's download as installation progress, then its failure", () => {
    const added = customSource(nativeCustomModel({
      modelId: addedId,
      origin: { _tag: "Added", repository: "owner/model-GGUF", commit: "c0ffee" },
      state: { _tag: "NotInstalled" },
    }))
    const config = customModelsConfig()
    const downloading = customModelDownloads(addedId, config, [targetDownload("d1", {
      _tag: "Downloading", completedBytes: 5, totalBytes: 10, stage: "downloading", bytesPerSecond: 2,
    })])
    expect(Option.getOrThrow(customModel(inputs(added, { related: downloading }))).acquisitionState).toEqual({
      _tag: "Installing",
      progress: { stage: "downloading", completedBytes: 5, totalBytes: 10, bytesPerSecond: Option.some(2) },
    })
    expect(Option.getOrThrow(customModel(inputs(added, { related: downloading }))).servingState._tag).toBe("Assessing")
    const failed = customModelDownloads(addedId, config, [targetDownload("d1", {
      _tag: "Failed", acknowledged: false, completedBytes: 5, totalBytes: 10, failure: { _tag: "NetworkUnavailable" },
    })])
    expect(Option.getOrThrow(customModel(inputs(added, { related: failed }))).acquisitionState).toEqual({
      _tag: "InstallFailed", failure: { _tag: "NetworkUnavailable" },
    })
    expect(Option.getOrThrow(customModel(inputs(added, { related: [] }))).acquisitionState).toEqual({ _tag: "NotInstalled" })
  })

  it("omits an external model whose files are gone and fails an unreadable one", () => {
    expect(customModel(inputs(customSource(nativeCustomModel({ state: { _tag: "NotInstalled" } }))))).toEqual(Option.none())
    const unreadable = Option.getOrThrow(customModel(inputs(customSource(nativeCustomModel({
      state: {
        _tag: "Unavailable",
        failure: { code: "unreadable", message: "Unreadable", retryable: false },
        installation: { _tag: "Resolved", installedBytes: 1, ownership: "External", primaryPath: "/m.gguf" },
      },
    })))))
    expect(unreadable.acquisitionState._tag).toBe("Unavailable")
    expect(unreadable.servingState._tag).toBe("Failed")
  })

  it("reports a drafter-only download, its failure, and a catalog drafter update", () => {
    const config = customModelsConfig()
    const entry = customSource(nativeCustomModel({ drafter: { ...attachedDrafter, state: { _tag: "NotInstalled" } } }))
    const downloading = customModelDownloads(foundId, config, [drafterDownload("dd", {
      _tag: "Pending", completedBytes: 0, totalBytes: 900,
    })])
    expect(Option.getOrThrow(customModel(inputs(entry, { related: downloading }))).drafter).toMatchObject(Option.some({
      availability: { _tag: "Downloading", progress: { stage: "queued", totalBytes: 900 } },
      updateAvailable: false,
    }))
    const failed = customModelDownloads(foundId, config, [drafterDownload("dd", {
      _tag: "Failed", acknowledged: false, completedBytes: 0, totalBytes: 900, failure: { _tag: "CorruptDownload" },
    })])
    expect(Option.getOrThrow(customModel(inputs(entry, { related: failed }))).drafter).toMatchObject(Option.some({
      availability: { _tag: "DownloadFailed", failure: { _tag: "CorruptDownload" } },
    }))
    const present = Option.getOrThrow(customModel(inputs(customSource(nativeCustomModel({ drafter: attachedDrafter })))))
    expect(Option.map(present.drafter, ({ availability }) => availability._tag)).toEqual(Option.some("Present"))

    // ICN reports the update on the attached drafter: a catalog drafter serving its earlier revision
    // while the catalog has a newer one. ACN shows it for every model using the drafter, whether or
    // not the catalog model itself is installed.
    const catalogDrafter = (updateAvailable: boolean) => customSource(nativeCustomModel({
      drafter: {
        method: { _tag: "DFlash" },
        reference: { _tag: "Catalog", modelId: "qwen3.6-35b-a3b:gguf:q4" },
        state: { _tag: "Ready" },
        updateAvailable,
      },
    }))
    const catalog = [{
      id: "qwen3.6-35b-a3b:gguf:q4",
      source: {
        displayName: "Qwen3.6 35B-A3B",
        desired: { speculativeMethod: Option.some({ _tag: "DFlash" }) },
        localState: { _tag: "NotInstalled" },
      },
    }] as never
    expect(Option.map(Option.getOrThrow(customModel(inputs(catalogDrafter(true), { catalog }))).drafter,
      ({ updateAvailable, name, label, availability }) => ({ updateAvailable, name, label, availability: availability._tag })))
      .toEqual(Option.some({
        updateAvailable: true,
        name: "Magnitude DFlash for Qwen3.6 35B-A3B",
        label: "Trained for Qwen3.6 35B-A3B",
        availability: "Present",
      }))
    expect(Option.map(Option.getOrThrow(customModel(inputs(catalogDrafter(false), { catalog }))).drafter,
      ({ updateAvailable }) => updateAvailable)).toEqual(Option.some(false))
  })
})

describe("attached drafters", () => {
  const recommendedDrafter = {
    category: "CatalogDrafter",
    reference: drafterReference,
    name: "Magnitude DFlash for Qwen3.5 4B",
    label: "Trained for Qwen3.5 4B",
    sizeBytes: 1_530_000_000,
    lineage: "Fingerprint",
    verdict: { _tag: "Bound", method: { _tag: "DFlash" }, memoryBytes: 1_530_000_000 },
    automatic: true,
    onThisComputer: false,
    method: { _tag: "DFlash" },
  }
  const builtInHeadsVerdict = { _tag: "Bound" as const, method: { _tag: "Mtp" as const }, memoryBytes: 619_000_000 }
  const withHeads = { ...fits, drafter: builtInHeadsVerdict }
  const recommendedButMissing = customSource(nativeCustomModel({
    drafter: { ...attachedDrafter, state: { _tag: "NotInstalled" } },
    recommendedDrafter,
  }))

  it("offers a found model's automatic drafter that was never downloaded instead of reporting a failure", () => {
    const row = Option.getOrThrow(customModel(inputs(recommendedButMissing, { assessment: withHeads })))
    expect(row.drafter).toEqual(Option.none())
    expect(Option.map(row.recommendedDrafter, ({ verdict, sizeBytes }) => ({ verdict, sizeBytes }))).toEqual(Option.some({
      verdict: { _tag: "Bound", method: { _tag: "DFlash" }, memoryBytes: 1_530_000_000 },
      sizeBytes: 1_530_000_000,
    }))
  })

  it("keeps a chosen drafter that is missing as an interrupted download with its own verdict", () => {
    const row = Option.getOrThrow(customModel(inputs(recommendedButMissing, {
      assessment: withHeads,
      drafterChoice: { _tag: "Reference", reference: drafterReference },
    })))
    expect(row.drafter).toMatchObject(Option.some({
      availability: { _tag: "DownloadFailed", failure: { _tag: "Interrupted" } },
      verdict: { _tag: "Bound", method: { _tag: "DFlash" }, memoryBytes: 1_530_000_000 },
      sizeBytes: 1_530_000_000,
    }))
  })

  it("attaches an imported model's automatic drafter in its folder once ICN finds it there", () => {
    const beside = { _tag: "Local", path: "/home/me/fx/g4imp/mtp-gemma-4-E2B-it-Q4_0.gguf" }
    const mtp = { _tag: "Bound", method: { _tag: "Mtp" }, memoryBytes: 200_000_000 }
    const imported = customSource(nativeCustomModel({
      modelId: importedId,
      origin: { _tag: "Imported", path: "/home/me/fx/g4imp/gemma-4-E2B-it-Q4_0.gguf" },
      drafter: { method: { _tag: "Mtp" }, reference: beside, state: { _tag: "Ready" }, updateAvailable: false },
      recommendedDrafter: {
        category: "NextToModel",
        reference: beside,
        name: "mtp-gemma-4-E2B-it-Q4_0.gguf",
        label: "In this folder",
        sizeBytes: 200_000_000,
        lineage: "Lineage",
        verdict: mtp,
        automatic: true,
        onThisComputer: true,
        method: { _tag: "Mtp" },
      },
    }))
    const row = Option.getOrThrow(customModel(inputs(imported)))
    expect(row.drafter).toMatchObject(Option.some({
      name: "mtp-gemma-4-E2B-it-Q4_0.gguf",
      availability: { _tag: "Present" },
      verdict: mtp,
    }))
  })

  it("never gives a present drafter the verdict of a different method", () => {
    const present = customSource(nativeCustomModel({ drafter: attachedDrafter, recommendedDrafter }))
    expect(Option.map(Option.getOrThrow(customModel(inputs(present, { assessment: withHeads }))).drafter, ({ verdict }) => verdict))
      .toEqual(Option.some({ _tag: "Bound", method: { _tag: "DFlash" }, memoryBytes: 1_530_000_000 }))
    const assessedDFlash = { _tag: "Bound" as const, method: { _tag: "DFlash" as const }, memoryBytes: 1_600_000_000 }
    expect(Option.map(Option.getOrThrow(customModel(inputs(present, { assessment: { ...fits, drafter: assessedDFlash } }))).drafter,
      ({ verdict }) => verdict)).toEqual(Option.some(assessedDFlash))
  })
})

describe("custom model downloads", () => {
  it("relates each download to the model it was started for: an added target, or a drafter alone", () => {
    const config = customModelsConfig()
    const target = targetDownload("t", { _tag: "Completed" })
    const drafter = drafterDownload("d", { _tag: "Completed" })
    const addedDrafter = drafterDownload("ad", { _tag: "Completed" }, addedId)
    const downloads = [target, drafter, addedDrafter]
    expect(customModelDownloads(addedId, config, downloads).map(({ kind, download }) => [kind, download.id]))
      .toEqual([["Target", "t"], ["Drafter", "ad"]])
    expect(customModelDownloads(foundId, config, downloads).map(({ kind }) => kind)).toEqual(["Drafter"])
    expect(customModelDownloads(importedId, config, downloads)).toEqual([])
  })

  it("lets a newer attempt supersede an older failure", () => {
    const config = customModelsConfig()
    const failed = targetDownload("old", {
      _tag: "Failed", acknowledged: false, completedBytes: 5, totalBytes: 10, failure: { _tag: "NetworkUnavailable" },
    })
    const added = customSource(nativeCustomModel({
      modelId: addedId,
      origin: { _tag: "Added", repository: "owner/model-GGUF", commit: "c0ffee" },
    }))
    const retried = customModelDownloads(addedId, config, [failed, targetDownload("new", { _tag: "Completed" })])
    expect(currentDownload(retried, "Target")).toEqual(Option.none())
    expect(Option.getOrThrow(customModel(inputs(added, { related: retried }))).acquisitionState._tag).toBe("Installed")
    expect(currentDownload(customModelDownloads(addedId, config, [failed]), "Target").pipe(Option.map(({ id }) => id)))
      .toEqual(Option.some("old"))
  })

  it("follows a catalog drafter's download, whose package names no attached repository file", () => {
    const catalogDrafter = nativeDownload("catalog", foundId, {
      repository: "magnitudedev/catalog-drafters", revision: "r1", path: "drafter.gguf",
    }, { _tag: "Downloading", completedBytes: 3, totalBytes: 9, stage: "downloading" })
    const config = customModelsConfig({
      drafters: { [foundId]: { _tag: "Reference", reference: { _tag: "Catalog", modelId: "qwen3.6-35b-a3b:gguf:q4" }, method: { _tag: "DFlash" } } },
    })
    expect(customModelDownloads(foundId, config, [catalogDrafter]).map(({ kind }) => kind)).toEqual(["Drafter"])
  })
})

describe("local models state", () => {
  const sources = (
    models: LocalModelSourcesState["customModels"],
    hiddenModels: LocalModelSourcesState["hiddenModels"] = [],
  ): LocalModelSourcesState => ({
    catalogRevision: 1,
    localModelsRevision: 2,
    reconciliationComplete: true,
    catalogModels: [],
    customModels: models,
    hiddenModels,
    unsupported: [{
      displayName: "",
      origin: { _tag: "Found", source: { _tag: "Ollama" } },
      path: "/ollama/blobs/sha256-1",
      reasons: [],
      detail: "unknown architecture",
    } as never],
  })
  const assessments = (modelIds: readonly string[]) => ({
    revision: 1,
    environmentId: "environment",
    catalog: { _tag: "Available", sourceRevision: 1, entries: [] },
    custom: {
      _tag: "Available",
      sourceRevision: 2,
      entries: modelIds.map((modelId) => ({
        subject: { _tag: "Custom", modelId },
        state: {
          _tag: "Assessed",
          capabilities,
          fingerprint: Option.none(),
          templateFingerprint: "t",
          profiles: [{
            _tag: "Fits",
            assessmentId: `a-${modelId}`,
            drafter: { _tag: "None" },
            memory: [],
            performance: [{ contextTokens: 32_768, estimatedTokensPerSecond: 30 }],
            profile: { contextLength: 32_768 },
          }],
        },
      })),
    },
  }) as unknown as ModelAssessmentsSnapshot

  it("lists hidden models apart, never as callable rows, and keeps unsupported reasons renderable", () => {
    const state = projectLocalModels({
      source: sources([customSource(nativeCustomModel({ drafter: attachedDrafter }))], [customSource(nativeCustomModel({
        modelId: "hf:other/hidden-GGUF/hidden.gguf",
        files: [{ path: "/models/hidden.gguf", sizeBytes: 1 }],
      }))]),
      assessment: assessments([foundId]),
      operations: [],
      runtime: [],
      removals: new Map(),
      config: customModelsConfig({ hidden: ["hf:other/hidden-GGUF/hidden.gguf"] }),
      downloads: [],
    })
    expect(state.models.map(({ modelId }) => modelId)).toEqual([foundId])
    expect(state.hidden.map(({ modelId }) => modelId)).toEqual(["hf:other/hidden-GGUF/hidden.gguf"])
    expect(state.unsupported).toEqual([{
      modelId: Option.none(),
      displayName: "sha256-1",
      origin: { _tag: "Found", source: { _tag: "Ollama" } },
      path: "/ollama/blobs/sha256-1",
      reasons: [{ _tag: "Unknown", kind: "Unknown", detail: "unknown architecture" }],
      detail: "unknown architecture",
    }])
    expect(state.preparation).toEqual({
      discovery: { complete: true, modelsFound: 1 },
      assessment: { complete: true, settledModels: 1, totalModels: 1 },
    })
  })
  it("follows a found exact target through its projector and then its drafter without ever reporting an update", () => {
    const modelId = "gemma-4-e2b-it-qat:gguf:q4"
    const desired = { metadata: { storageBytes: 1 }, profile: { contextLength: 32_768 }, speculativeMethod: Option.none() }
    const step = (ownership: string, missingComponents: readonly unknown[], speculative: boolean) => projectLocalModels({
      source: { ...sources([]), catalogModels: [{
        id: modelId,
        source: {
          aliases: Option.none(), description: "", desired, displayName: "Gemma 4 E2B", fidelityRank: 1, intelligence: 1,
          license: "apache-2.0", parameterization: { architecture: "dense", totalParameters: 1 }, quantizationAware: true,
          releaseDate: "2026-04-02", sourceUrls: [], support: { level: "supported" }, variantLabel: "Q4 QAT",
          localState: {
            _tag: "Installed",
            effective: { _tag: "Ready", model: { ...desired, speculativeMethod: speculative ? Option.some({ _tag: "Mtp" }) : Option.none() } },
            installation: { _tag: "Resolved", installedBytes: 1, primaryPath: "/files/model.gguf", ownership },
            updateState: { _tag: "Current" },
            missingComponents,
            storedComponents: [],
            drafterWithheld: false,
          },
        },
      } as unknown as LocalModelSourcesState["catalogModels"][number]] },
      assessment: assessments([]),
      operations: [],
      runtime: [],
      removals: new Map(),
      config: customModelsConfig(),
      downloads: [],
    }).models[0]
    const projector = { component: "Projector", downloadBytes: 987_000_000 }
    const drafter = { component: "Drafter", downloadBytes: 59_200_000 }
    const observed = [
      step("External", [projector, drafter], false),
      step("Mixed", [drafter], false),
      step("Mixed", [], true),
    ].map((model) => model?._tag === "Catalog" ? [model.acquisitionState._tag, model.missingComponents] : undefined)
    expect(observed).toEqual([
      ["Installed", [projector, drafter]],
      ["Installed", [drafter]],
      ["Installed", []],
    ])
  })

  it("lists catalog models with their aliases and leaves hiding an external copy to ICN", () => {
    const desired = {
      metadata: { storageBytes: 1 },
      profile: { contextLength: 32_768 },
      speculativeMethod: Option.none(),
    }
    const catalogEntry = (id: string, ownership: string, missingComponents: readonly unknown[] = []) => ({
      id,
      source: {
        aliases: Option.some([`hf:owner/${id}/model.gguf`]),
        description: "",
        desired,
        displayName: "Catalog Model",
        fidelityRank: 1,
        intelligence: 1,
        license: "apache-2.0",
        localState: {
          _tag: "Installed",
          effective: { _tag: "Ready", model: desired },
          installation: { _tag: "Resolved", installedBytes: 1, primaryPath: `/files/${id}.gguf`, ownership },
          updateState: { _tag: "Current" },
          missingComponents,
          storedComponents: [],
          drafterWithheld: false,
        },
        parameterization: { architecture: "dense", totalParameters: 1 },
        quantizationAware: false,
        releaseDate: "2026-08-29",
        sourceUrls: [],
        support: { level: "supported" },
        variantLabel: "Q4",
      },
    }) as unknown as LocalModelSourcesState["catalogModels"][number]
    const external = "qwen3.6-35b-a3b:gguf:q4"
    const owned = "qwen3.6-27b:gguf:q4"
    const state = projectLocalModels({
      source: { ...sources([]), catalogModels: [catalogEntry(external, "External"), catalogEntry(owned, "Magnitude")] },
      assessment: assessments([]),
      operations: [],
      runtime: [],
      removals: new Map(),
      config: customModelsConfig({ hidden: [external, owned] as never }),
      downloads: [],
    })
    expect(state.models.map(({ modelId }) => modelId)).toEqual([external, owned])
    expect(state.hidden).toEqual([])
    expect(state.models[1]?._tag === "Catalog" && state.models[1].catalogData.aliases).toEqual([`hf:owner/${owned}/model.gguf`])
    expect(state.models[0]?._tag === "Catalog" && state.models[0].catalogData.aliases).toEqual([`hf:owner/${external}/model.gguf`])
    const hiddenCopy = projectLocalModels({
      source: { ...sources([]), catalogModels: [catalogEntry(external, "External")] },
      assessment: assessments([]),
      operations: [],
      runtime: [],
      removals: new Map(),
      config: customModelsConfig({ hidden: [`hf:owner/${external}/model.gguf`] as never }),
      downloads: [],
    })
    expect(hiddenCopy.models[0]?._tag === "Catalog" && hiddenCopy.models[0].catalogData.aliases).toEqual([])

    // ICN no longer attributes a hidden external copy: the catalog model isn't installed and the copy
    // is a hidden found model under its alias, until it is unhidden.
    const alias = `hf:owner/${external}/model.gguf`
    const notInstalled = { ...catalogEntry(external, "External"), source: {
      ...catalogEntry(external, "External").source, localState: { _tag: "NotInstalled" },
    } } as unknown as LocalModelSourcesState["catalogModels"][number]
    const hidden = projectLocalModels({
      source: {
        ...sources([], [customSource(nativeCustomModel({ modelId: alias, files: [{ path: "/files/model.gguf", sizeBytes: 1 }] }))]),
        catalogModels: [notInstalled],
      },
      assessment: assessments([]),
      operations: [],
      runtime: [],
      removals: new Map(),
      config: customModelsConfig({ hidden: [alias] as never }),
      downloads: [],
    })
    expect(hidden.models.map((model) => [model.modelId, model.acquisitionState._tag])).toEqual([[external, "NotInstalled"]])
    expect(hidden.hidden.map(({ modelId }) => modelId)).toEqual([alias])
    const unhidden = projectLocalModels({
      source: { ...sources([]), catalogModels: [catalogEntry(external, "External")] },
      assessment: assessments([]),
      operations: [],
      runtime: [],
      removals: new Map(),
      config: customModelsConfig(),
      downloads: [],
    })
    expect(unhidden.models.map((model) => [model.modelId, model.acquisitionState._tag])).toEqual([[external, "Installed"]])
    expect(unhidden.hidden).toEqual([])

    // An exact target found without its projector is installed and current; the projector is
    // reported on its own so clients can offer it, never as an update.
    const lacking = projectLocalModels({
      source: {
        ...sources([]),
        catalogModels: [catalogEntry(external, "External", [{ component: "Projector", downloadBytes: 1_034_945_421 }])],
      },
      assessment: assessments([]),
      operations: [],
      runtime: [],
      removals: new Map(),
      config: customModelsConfig(),
      downloads: [],
    })
    const [lackingModel] = lacking.models
    expect(lackingModel?.acquisitionState._tag).toBe("Installed")
    expect(lackingModel?._tag === "Catalog" && lackingModel.missingComponents)
      .toEqual([{ component: "Projector", downloadBytes: 1_034_945_421 }])
    expect(state.models.every((model) => model._tag === "Catalog" && model.missingComponents.length === 0)).toBe(true)
  })

  it("moves a found model the engine cannot run into the unsupported list with its reasons", () => {
    const unsupportedId = "hf:other/unsupported-GGUF/unsupported.gguf"
    const snapshot = assessments([foundId, unsupportedId]) as unknown as {
      custom: { entries: Array<{ subject: { modelId: string }; state: { profiles: unknown[] } }> }
    }
    snapshot.custom.entries[1]!.state.profiles = [{
      _tag: "Unsupported",
      profile: Option.some({ contextLength: 32_768 }),
      reasons: [{ _tag: "Architecture", architecture: "deepseek4" }],
      detail: "unknown architecture deepseek4",
    }]
    const state = projectLocalModels({
      source: { ...sources([customSource(nativeCustomModel()), customSource(nativeCustomModel({
        modelId: unsupportedId,
        displayName: "Unsupported Model",
        files: [{ path: "/models/unsupported.gguf", sizeBytes: 1 }],
      }))]), unsupported: [] },
      assessment: snapshot as unknown as ModelAssessmentsSnapshot,
      operations: [],
      runtime: [],
      removals: new Map(),
      config: customModelsConfig(),
      downloads: [],
    })
    expect(state.models.map(({ modelId }) => modelId)).toEqual([foundId])
    expect(state.unsupported).toEqual([{
      modelId: Option.some(unsupportedId),
      displayName: "Unsupported Model",
      origin: { _tag: "Found", source: { _tag: "LmStudio" } },
      path: "/models/unsupported.gguf",
      reasons: [{ _tag: "Architecture", architecture: "deepseek4" }],
      detail: "unknown architecture deepseek4",
    }])
  })
  it("renders decoded engine reasons with optional fields and kernel statics", () => {
    const unsupportedId = "hf:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf"
    const snapshot = assessments([unsupportedId]) as unknown as {
      custom: { entries: Array<{ state: { profiles: unknown[] } }> }
    }
    snapshot.custom.entries[0]!.state.profiles = [{
      _tag: "Unsupported",
      profile: Option.some({ contextLength: 40_960 }),
      reasons: [
        { _tag: "Architecture", architecture: "qwen3" },
        { _tag: "Tokenizer", model: "gpt2", pre: Option.some("qwen2") },
        { _tag: "Tokenizer", model: "llama", pre: Option.none() },
        { _tag: "KernelDomain", backend: "metal", entry: "attention", statics: [{ name: "head_dim", value: 96 }] },
      ],
      detail: "no registered model family recognizes architecture qwen3",
    }]
    const state = projectLocalModels({
      source: { ...sources([customSource(nativeCustomModel({
        modelId: unsupportedId,
        displayName: "Qwen3-0.6B-Q4_K_M",
        files: [{ path: "/models/Qwen3-0.6B-Q4_K_M.gguf", sizeBytes: 1 }],
      }))]), unsupported: [] },
      assessment: snapshot as unknown as ModelAssessmentsSnapshot,
      operations: [],
      runtime: [],
      removals: new Map(),
      config: customModelsConfig(),
      downloads: [],
    })
    expect(state.models).toEqual([])
    expect(state.unsupported.map(({ reasons }) => reasons)).toEqual([[
      { _tag: "Architecture", architecture: "qwen3" },
      { _tag: "Tokenizer", model: "gpt2", pre: Option.some("qwen2") },
      { _tag: "Tokenizer", model: "llama", pre: Option.none() },
      { _tag: "KernelDomain", backend: "metal", entry: "attention", statics: [["head_dim", 96]] },
    ]])
  })

  it("renders a reason a newer engine raised as Unknown with its detail", () => {
    const unsupportedId = "hf:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf"
    const decodeReasons = Schema.decodeUnknownSync(Schema.parseJson(Schema.Array(UnsupportedReason)))
    const snapshot = assessments([unsupportedId]) as unknown as {
      custom: { entries: Array<{ state: { profiles: unknown[] } }> }
    }
    snapshot.custom.entries[0]!.state.profiles = [{
      _tag: "Unsupported",
      profile: Option.some({ contextLength: 40_960 }),
      reasons: decodeReasons(JSON.stringify([
        { _tag: "SlidingWindowPattern", detail: "interleaved sliding window", pattern: [1, 0, 1] },
        { _tag: "RopeScaling", kind: "longrope" },
      ])),
      detail: "interleaved sliding window",
    }]
    const state = projectLocalModels({
      source: { ...sources([customSource(nativeCustomModel({
        modelId: unsupportedId,
        displayName: "Qwen3-0.6B-Q4_K_M",
        files: [{ path: "/models/Qwen3-0.6B-Q4_K_M.gguf", sizeBytes: 1 }],
      }))]), unsupported: [] },
      assessment: snapshot as unknown as ModelAssessmentsSnapshot,
      operations: [],
      runtime: [],
      removals: new Map(),
      config: customModelsConfig(),
      downloads: [],
    })
    expect(state.unsupported.map(({ reasons }) => reasons)).toEqual([[
      { _tag: "Unknown", kind: "SlidingWindowPattern", detail: "interleaved sliding window" },
      { _tag: "RopeScaling", kind: "longrope" },
    ]])
  })
})
