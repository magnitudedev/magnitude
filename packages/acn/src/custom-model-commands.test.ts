import { BunFileSystem } from "@effect/platform-bun"
import { Effect, Layer, Option, Schema, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { mkdtempSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import {
  IcnCandidates,
  IcnDownloads,
  IcnLocalModels,
  IcnModelAssessments,
  type IcnCandidatesService,
} from "@magnitudedev/icn"
import {
  CandidateInspection,
  DrafterCandidate as NativeDrafterCandidate,
  type ModelCandidate,
  type ModelDownload,
} from "@magnitudedev/icn-protocol/schemas"
import {
  CustomFormModelIdSchema,
  HuggingFaceRepositoryIdSchema,
  type LocalModelsState,
} from "@magnitudedev/acn-protocol"
import { CustomModelCommands, CustomModelCommandsLive } from "./custom-model-commands"
import { CustomModelReconciler } from "./custom-model-reconciler"
import { CustomModels } from "./custom-models"
import { DrafterMethodsLive } from "./drafter-references"
import { LocalModelSources } from "./local-model-sources"
import { LocalModels, customModel } from "./local-models"
import { makeModelLocations } from "./model-locations"
import { ServingUsage } from "./serving-usage"
import {
  addedId,
  customModelsConfig,
  customSource,
  drafterReference,
  foundId,
  importedId,
  makeTestCustomModels,
  nativeCustomModel,
  targetDownload,
  drafterDownload,
} from "./custom-models.test-support"

const repository = HuggingFaceRepositoryIdSchema.make("owner/model-GGUF")

const bound = { _tag: "Bound", method: { _tag: "DFlash" }, memoryBytes: 900 }

const nativeCandidate = (overrides: Record<string, unknown> = {}) => Schema.decodeUnknownSync(NativeDrafterCandidate)({
  category: "CatalogDrafter",
  reference: drafterReference,
  label: "Magnitude DFlash",
  lineage: "Lineage",
  method: { _tag: "DFlash" },
  onThisComputer: false,
  sizeBytes: 900,
  verdict: bound,
  ...overrides,
})

const inspection = (onThisComputer?: string) => Schema.decodeUnknownSync(CandidateInspection)({
  source: { _tag: "HuggingFace", repository: "owner/model-GGUF", commit: "c0ffee" },
  displayName: "",
  versions: [
    {
      label: "Q4_K_M",
      modelId: "hf:owner/model-GGUF/model-Q4_K_M.gguf",
      files: [{ path: "model-Q4_K_M.gguf", sizeBytes: 100 }],
      sizeBytes: 100,
      recommended: true,
      ...(onThisComputer === undefined ? {} : { onThisComputer }),
    },
    {
      label: "Q2_K",
      modelId: "hf:owner/model-GGUF/model-Q2_K.gguf",
      files: [{ path: "model-Q2_K.gguf", sizeBytes: 50 }],
      sizeBytes: 50,
      recommended: false,
      assessment: { _tag: "Unsupported", profile: { contextLength: 4096 }, reasons: [{ _tag: "WeightTypes", types: [{ ggmlType: "Q2_K", tensorCount: 2 }] }], detail: "" },
    },
  ],
})

interface Recorded {
  readonly bundles: ModelCandidate[]
  readonly started: unknown[]
  readonly cancelled: string[]
  readonly acknowledged: string[]
  readonly removedFiles: string[]
  readonly drafterRequests: unknown[]
  pushes: number
}

const harness = (options: {
  readonly config?: ReturnType<typeof customModelsConfig>
  readonly inspect?: ReturnType<typeof inspection>
  readonly drafters?: readonly NativeDrafterCandidate[]
  readonly downloads?: readonly ModelDownload[]
  readonly customModels?: ReturnType<typeof nativeCustomModel>[]
} = {}) => {
  const recorded: Recorded = {
    bundles: [], started: [], cancelled: [], acknowledged: [], removedFiles: [], drafterRequests: [], pushes: 0,
  }
  const sourceModels = (options.customModels ?? [nativeCustomModel()]).map(customSource)
  const candidates: IcnCandidatesService = {
    search: () => Effect.succeed({
      hiddenCount: 4,
      results: [{
        repository: "HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive",
        architecture: Option.some("qwen35moe"),
        baseModels: [],
        commit: "c",
        downloads: Option.some(12),
        gated: false,
        private: false,
        lastModified: Option.none(),
        likes: Option.none(),
        tags: [],
        unsupported: [],
        variantOf: Option.some("qwen3.6-35b-a3b:gguf:q4"),
      }],
    }) as never,
    resolveRepository: () => Effect.die("unused"),
    inspect: () => Effect.succeed(options.inspect ?? inspection()),
    drafters: (request) => Effect.sync(() => {
      recorded.drafterRequests.push(request)
      return options.drafters ?? [nativeCandidate()]
    }),
    assess: () => Effect.die("unused"),
    bundle: (candidate) => Effect.sync(() => {
      recorded.bundles.push(candidate)
      return { _tag: "Standalone", package: { id: "bundle" } } as never
    }),
  }
  const layer = Effect.gen(function* () {
    const custom = yield* makeTestCustomModels(options.config ?? customModelsConfig())
    const rows = sourceModels.flatMap((entry) => Option.toArray(customModel({
      entry, assessment: undefined, instances: [], related: [], acceptance: [], catalog: [],
    })))
    const localState: LocalModelsState = {
      preparation: { discovery: { complete: true, modelsFound: rows.length }, assessment: { complete: true, settledModels: 0, totalModels: 0 } },
      models: rows,
      unsupported: [],
      hidden: [],
    }
    return Layer.mergeAll(
      Layer.succeed(IcnCandidates, candidates),
      Layer.succeed(IcnDownloads, {
        get: Effect.succeed({ revision: 1, state: { downloads: [...(options.downloads ?? [])] } }),
        changes: Stream.empty,
        initialized: Effect.succeed(true),
        refresh: Effect.void,
        start: (bundle: unknown, forModel: string) => Effect.sync(() => {
          recorded.started.push({ bundle, forModel })
          return Option.none()
        }),
        cancel: (id: string) => Effect.sync(() => { recorded.cancelled.push(id); return {} as ModelDownload }),
        acknowledgeFailure: (id: string) => Effect.sync(() => { recorded.acknowledged.push(id); return {} as ModelDownload }),
      } as never),
      Layer.succeed(IcnModelAssessments, { get: Effect.succeed({ revision: 1, state: { environmentId: "environment" } }) } as never),
      Layer.succeed(IcnLocalModels, {
        removeFiles: (modelId: string) => Effect.sync(() => {
          recorded.removedFiles.push(modelId)
          return { modelId, removed: true, freedBytes: 1 }
        }),
      } as never),
      Layer.succeed(LocalModelSources, { state: Effect.succeed({ customModels: sourceModels }) } as never),
      Layer.succeed(LocalModels, { state: Effect.succeed(localState) } as never),
      Layer.succeed(CustomModels, custom),
      Layer.succeed(CustomModelReconciler, { push: Effect.sync(() => { recorded.pushes += 1 }) }),
      Layer.succeed(ServingUsage, {
        acceptance: Effect.succeed([{
          model: foundId,
          drafter: "embedded",
          acceptance: { acceptanceRate: 0.9, tokensPerSecond: 10, withoutDrafterTokensPerSecond: Option.none(), slower: false },
        }]),
      } as never),
      DrafterMethodsLive,
      makeModelLocations({ platform: "linux", env: { OLLAMA_MODELS: "/nonexistent/ollama" }, home: "/nonexistent-home" })
        .pipe(Layer.provide(BunFileSystem.layer)),
      BunFileSystem.layer,
    )
  })
  const run = <A, E>(use: (commands: Effect.Effect.Success<typeof CustomModelCommands>, models: Effect.Effect.Success<typeof CustomModels>) => Effect.Effect<A, E>) =>
    Effect.runPromise(Effect.gen(function* () {
      const context = yield* layer
      return yield* Effect.gen(function* () {
        const commands = yield* CustomModelCommands
        const models = yield* CustomModels
        return yield* use(commands, models)
      }).pipe(Effect.provide(CustomModelCommandsLive.pipe(Layer.provideMerge(context))))
    }))
  return { recorded, run }
}

describe("finding models", () => {
  it("maps search results with uploader, variant, and hidden count", async () => {
    const { run } = harness()
    const result = await run((commands) => commands.searchHuggingFace({ query: "qwen", limit: Option.none(), all: false }))
    expect(result.hiddenCount).toBe(4)
    expect(result.results[0]).toMatchObject({
      displayName: "Qwen3.6 35B A3B Uncensored HauhauCS Aggressive",
      uploader: "HauhauCS",
      variantOf: Option.some("qwen3.6-35b-a3b:gguf:q4"),
      downloads: 12,
      architecture: Option.some("qwen35moe"),
      unsupported: Option.none(),
    })
  })

  it("inspects owner/repo:QUANT and selects that version", async () => {
    const { run } = harness()
    const result = await run((commands) => commands.inspectCandidate("owner/model-GGUF:q2_k"))
    expect(result.displayName).toBe("model")
    expect(result.selectedVersion).toEqual(Option.some("Q2_K"))
    expect(result.versions.map(({ modelId }) => modelId)).toEqual([
      "hf:owner/model-GGUF/model-Q4_K_M.gguf",
      "hf:owner/model-GGUF/model-Q2_K.gguf",
    ])
    expect(result.versions[1]?.assessment.pipe(Option.map(({ _tag }) => _tag))).toEqual(Option.some("Unsupported"))
  })

  it("refuses a version the repository does not have", async () => {
    const { run } = harness()
    const result = await run((commands) => commands.inspectCandidate("owner/model-GGUF:IQ1_M").pipe(Effect.either))
    expect(result._tag === "Left" && result.left._tag).toBe("InvalidModelReference")
  })

  it("orders drafter candidates and passes a different drafter to the engine", async () => {
    const { run, recorded } = harness({
      drafters: [
        nativeCandidate({ category: "BuiltIn", reference: { _tag: "Embedded" }, label: "Built in", lineage: "Exact", sizeBytes: 0 }),
        nativeCandidate({ category: "OnThisComputer", reference: { _tag: "Local", path: "/d/large.gguf" }, label: "Large", sizeBytes: 9 }),
        nativeCandidate({ category: "OnThisComputer", reference: { _tag: "Local", path: "/d/small.gguf" }, label: "Small", sizeBytes: 1 }),
        nativeCandidate(),
      ],
    })
    const result = await run((commands) => commands.listDrafterCandidates({ _tag: "Model", modelId: foundId }, Option.some("z-lab/d:dflash.gguf")))
    expect(result.map(({ label }) => label)).toEqual(["Magnitude DFlash", "Built in", "Small", "Large"])
    expect(result[1]?.measuredAcceptance.pipe(Option.map(({ acceptanceRate }) => acceptanceRate))).toEqual(Option.some(0.9))
    expect(recorded.drafterRequests[0]).toMatchObject({ _tag: "Model", modelId: foundId, other: Option.some("z-lab/d:dflash.gguf") })
  })
})

describe("adding models", () => {
  it("declares an added model with its drafter and downloads the bundle ICN builds", async () => {
    const { run, recorded } = harness()
    const result = await run((commands, models) => Effect.gen(function* () {
      const added = yield* commands.addHuggingFaceModel({ repository, revision: Option.none(), version: Option.none(), drafter: { _tag: "Auto" } })
      return { added, config: yield* models.get }
    }))
    expect(result.added).toMatchObject({ modelId: addedId, version: "Q4_K_M", outcome: "Started" })
    expect(result.config.declarations[addedId]?.origin).toEqual({ _tag: "Added", repository: "owner/model-GGUF", commit: "c0ffee" })
    expect(result.config.drafters[addedId]).toEqual({ _tag: "Reference", reference: drafterReference })
    expect(recorded.bundles[0]).toMatchObject({
      source: { _tag: "HuggingFace", repository: "owner/model-GGUF", commit: "c0ffee" },
      target: ["model-Q4_K_M.gguf"],
      drafter: Option.some({ method: { _tag: "DFlash" }, reference: drafterReference }),
    })
    expect(recorded.started).toEqual([{ bundle: { _tag: "Standalone", package: { id: "bundle" } }, forModel: addedId }])
    expect(recorded.pushes).toBeGreaterThan(0)
  })

  it("refuses an unsupported version before anything downloads", async () => {
    const { run, recorded } = harness()
    const result = await run((commands) => commands.addHuggingFaceModel({
      repository, revision: Option.none(), version: Option.some("Q2_K"), drafter: { _tag: "None" },
    }).pipe(Effect.either))
    expect(result._tag).toBe("Left")
    expect(recorded.started).toEqual([])
  })

  it("adds a version already on this computer by fetching only its drafter", async () => {
    const { run, recorded } = harness({ inspect: inspection(foundId) })
    const result = await run((commands, models) => Effect.gen(function* () {
      const added = yield* commands.addHuggingFaceModel({ repository, revision: Option.none(), version: Option.none(), drafter: { _tag: "Auto" } })
      return { added, config: yield* models.get }
    }))
    expect(result.added).toMatchObject({ modelId: foundId, outcome: "AlreadyOnThisComputer" })
    expect(result.config.declarations[foundId]).toBeUndefined()
    expect(recorded.bundles[0]).toMatchObject({ source: { _tag: "Local", path: "/models" }, target: ["found-Q8_0.gguf"] })
  })

  it("imports a GGUF in place under its header-digest identity", async () => {
    const { run } = harness({
      inspect: Schema.decodeUnknownSync(CandidateInspection)({
        source: { _tag: "Local", path: "/models/imported.gguf" },
        displayName: "Imported",
        versions: [{ label: "Q4_0", modelId: importedId, files: [{ path: "/models/imported.gguf", sizeBytes: 1 }], sizeBytes: 1, recommended: true }],
      }),
      config: customModelsConfig({ declarations: {} }),
    })
    const result = await run((commands, models) => Effect.gen(function* () {
      const imported = yield* commands.importModelFile("/models/imported.gguf", { _tag: "None" })
      return { imported, config: yield* models.get }
    }))
    expect(result.imported.modelId).toBe(importedId)
    expect(result.config.declarations[importedId]?.origin).toMatchObject({ _tag: "Imported", path: "/models/imported.gguf" })
    expect(result.config.drafters[importedId]).toEqual({ _tag: "None" })
  })
})

describe("drafter changes", () => {
  it("refuses a drafter the engine rejects", async () => {
    const { run } = harness({
      drafters: [nativeCandidate({ verdict: { _tag: "Rejected", reasons: [{ _tag: "TargetMismatch", detail: "x" }], detail: "hidden size" } })],
    })
    const result = await run((commands) => commands.setModelDrafter(foundId, { _tag: "Reference", reference: drafterReference }).pipe(Effect.either))
    expect(result._tag === "Left" && result.left._tag).toBe("InvalidModelReference")
  })

  it("turns a drafter off without downloading anything", async () => {
    const { run, recorded } = harness()
    const result = await run((commands, models) => commands.setModelDrafter(foundId, { _tag: "None" }).pipe(
      Effect.zipRight(models.get),
    ))
    expect(result.drafters[foundId]).toEqual({ _tag: "None" })
    expect(recorded.bundles).toEqual([])
  })

  it("refuses a model that is not on this computer", async () => {
    const { run } = harness()
    const result = await run((commands) => commands.setModelDrafter(CustomFormModelIdSchema.make("hf:a/b/c.gguf"), { _tag: "None" }).pipe(Effect.either))
    expect(result._tag === "Left" && result.left._tag).toBe("LocalModelMutationFailed")
  })
})

describe("removal by origin", () => {
  it("deletes an added model's copy, forgets an imported model, and hides a found model", async () => {
    const { run, recorded } = harness({
      downloads: [targetDownload("active", { _tag: "Downloading", completedBytes: 1, totalBytes: 2, stage: "downloading" })],
    })
    const config = await run((commands, models) => Effect.gen(function* () {
      yield* commands.remove(addedId)
      yield* commands.remove(importedId)
      yield* commands.remove(foundId)
      return yield* models.get
    }))
    expect(recorded.cancelled).toEqual(["active"])
    expect(recorded.removedFiles).toEqual([addedId])
    expect(Object.keys(config.declarations)).toEqual([])
    expect(config.hidden).toEqual([foundId])
  })

  it("unhides a hidden model", async () => {
    const { run } = harness({ config: customModelsConfig({ hidden: [foundId] }) })
    const config = await run((commands, models) => commands.unhide(foundId).pipe(Effect.zipRight(models.get)))
    expect(config.hidden).toEqual([])
  })
})

describe("drafter and download recovery", () => {
  it("cancels and acknowledges a drafter-only download", async () => {
    const { run, recorded } = harness({
      downloads: [
        drafterDownload("running", { _tag: "Pending", completedBytes: 0, totalBytes: 9 }),
        drafterDownload("failed", { _tag: "Failed", acknowledged: false, completedBytes: 0, totalBytes: 9, failure: { _tag: "Interrupted" } }),
      ],
    })
    await run((commands) => commands.cancelSync(foundId).pipe(Effect.zipRight(commands.acknowledgeSyncFailure(foundId))))
    expect(recorded.cancelled).toEqual(["running"])
    expect(recorded.acknowledged).toEqual(["failed"])
  })

  it("restarts a missing added model's download", async () => {
    const { run, recorded } = harness({
      customModels: [nativeCustomModel({
        modelId: addedId,
        origin: { _tag: "Added", repository: "owner/model-GGUF", commit: "c0ffee" },
        state: { _tag: "NotInstalled" },
      })],
    })
    const result = await run((commands) => commands.sync(addedId))
    expect(result.outcome).toBe("Started")
    expect(recorded.bundles[0]).toMatchObject({ target: ["model-Q4_K_M.gguf"] })
  })

  it("downloads a missing attached drafter on Sync", async () => {
    const { run, recorded } = harness({
      customModels: [nativeCustomModel({ drafter: { method: { _tag: "DFlash" }, reference: drafterReference, state: { _tag: "NotInstalled" } } })],
    })
    const result = await run((commands) => commands.sync(foundId))
    expect(result.outcome).toBe("Started")
    expect(recorded.bundles[0]).toMatchObject({ drafter: Option.some({ reference: drafterReference }) })
    expect(recorded.started).toEqual([{ bundle: { _tag: "Standalone", package: { id: "bundle" } }, forModel: foundId }])
  })
})

describe("model locations", () => {
  it("validates, adds, and removes user folders and turns automatic locations off", async () => {
    const folder = mkdtempSync(join(tmpdir(), "magnitude-folder-"))
    const file = join(folder, "a.gguf")
    writeFileSync(file, "")
    const { run } = harness()
    const result = await run((commands, models) => Effect.gen(function* () {
      const relative = yield* commands.addFolder("models").pipe(Effect.either)
      const missing = yield* commands.addFolder(join(folder, "missing")).pipe(Effect.either)
      const notDirectory = yield* commands.addFolder(file).pipe(Effect.either)
      yield* commands.addFolder(folder)
      yield* commands.addFolder(folder)
      yield* commands.setLocationEnabled("Ollama", false)
      const locations = yield* commands.listLocations
      const afterAdd = yield* models.get
      yield* commands.removeFolder(folder)
      return { relative, missing, notDirectory, locations, afterAdd, afterRemove: yield* models.get }
    }))
    expect(result.relative._tag === "Left" && result.relative.left._tag).toBe("InvalidDirectoryPath")
    expect(result.missing._tag === "Left" && result.missing.left._tag).toBe("DirectoryNotFound")
    expect(result.notDirectory._tag === "Left" && result.notDirectory.left._tag).toBe("PathNotDirectory")
    expect(result.afterAdd.folders).toEqual([folder])
    expect(result.afterAdd.disabledLocations).toEqual(["Ollama"])
    expect(result.locations.find(({ kind }) => kind === "Ollama")).toMatchObject({ enabled: false, automatic: true, found: false })
    expect(result.locations.find(({ kind }) => kind === "Folder")).toEqual({ kind: "Folder", path: folder, enabled: true, automatic: false, found: true })
    expect(result.afterRemove.folders).toEqual([])
  })
})

