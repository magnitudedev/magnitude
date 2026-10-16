import { Option } from "effect"
import { describe, expect, it } from "vitest"
import type { ModelAssessmentsSnapshot } from "@magnitudedev/icn-protocol/schemas"
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
  profile: { contextLength: 32_768 },
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

    const catalogDrafter = customSource(nativeCustomModel({
      drafter: { method: { _tag: "DFlash" }, reference: { _tag: "Catalog", modelId: "qwen3.6-35b-a3b:gguf:q4" }, state: { _tag: "Ready" } },
    }))
    const catalog = [{
      id: "qwen3.6-35b-a3b:gguf:q4",
      source: {
        displayName: "Qwen3.6 35B-A3B",
        desired: { speculativeMethod: Option.some({ _tag: "DFlash" }) },
        localState: { _tag: "Installed", updateState: { _tag: "Available" } },
      },
    }] as never
    expect(Option.map(Option.getOrThrow(customModel(inputs(catalogDrafter, { catalog }))).drafter,
      ({ updateAvailable, name, label }) => ({ updateAvailable, name, label })))
      .toEqual(Option.some({
        updateAvailable: true,
        name: "Magnitude DFlash for Qwen3.6 35B-A3B",
        label: "Trained for Qwen3.6 35B-A3B",
      }))
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
      drafters: { [foundId]: { _tag: "Reference", reference: { _tag: "Catalog", modelId: "qwen3.6-35b-a3b:gguf:q4" } } },
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
  it("lists catalog models with their aliases and leaves hiding an external copy to ICN", () => {
    const desired = {
      metadata: { storageBytes: 1 },
      profile: { contextLength: 32_768 },
      speculativeMethod: Option.none(),
    }
    const catalogEntry = (id: string, ownership: string) => ({
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
  })

  it("moves a found model the engine cannot run into the unsupported list with its reasons", () => {
    const unsupportedId = "hf:other/unsupported-GGUF/unsupported.gguf"
    const snapshot = assessments([foundId, unsupportedId]) as unknown as {
      custom: { entries: Array<{ subject: { modelId: string }; state: { profiles: unknown[] } }> }
    }
    snapshot.custom.entries[1]!.state.profiles = [{
      _tag: "Unsupported",
      profile: { contextLength: 32_768 },
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
      profile: { contextLength: 40_960 },
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
})
