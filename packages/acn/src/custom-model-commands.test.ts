import { BunFileSystem } from "@effect/platform-bun"
import { Effect, Layer, Option, Schema, Stream, SubscriptionRef } from "effect"
import { describe, expect, it } from "vitest"
import type { CustomModelsConfig } from "@magnitudedev/storage"
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs"
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
  type ModelDownloadEncoded,
} from "@magnitudedev/icn-protocol/schemas"
import {
  CatalogFormModelIdSchema,
  CustomFormModelIdSchema,
  DrafterReferenceSchema,
  HuggingFaceRepositoryIdSchema,
  type LocalModelsState,
} from "@magnitudedev/acn-protocol"
import { CustomModelCommands, CustomModelCommandsLive } from "./custom-model-commands"
import { CatalogModelSync } from "./catalog-model-sync"
import { CustomModelReconciler } from "./custom-model-reconciler"
import { CustomModels, localModelsConfiguration } from "./custom-models"
import { DrafterMethodsLive } from "./drafter-references"
import { LocalModelSources, type CatalogModelSource } from "./local-model-sources"
import { LocalModels, catalogAcquisition, customModel } from "./local-models"
import { customModelDownloads, type RelatedDownload } from "./custom-model-downloads"
import { makeModelLocations } from "./model-locations"
import {
  addedId,
  customModelsConfig,
  customSource,
  drafterReference,
  foundId,
  importedId,
  makeTestCustomModels,
  nativeCustomModel,
  nativeDownload,
  readyModel,
  targetDownload,
  drafterDownload,
} from "./custom-models.test-support"

const repository = HuggingFaceRepositoryIdSchema.make("owner/model-GGUF")

const bound = { _tag: "Bound", method: { _tag: "DFlash" }, memoryBytes: 900 }

const nativeCandidate = (overrides: Record<string, unknown> = {}) => Schema.decodeUnknownSync(NativeDrafterCandidate)({
  category: "CatalogDrafter",
  reference: drafterReference,
  name: "Magnitude DFlash for Model",
  label: "Magnitude DFlash",
  lineage: "Lineage",
  method: { _tag: "DFlash" },
  onThisComputer: false,
  sizeBytes: 900,
  verdict: bound,
  automatic: (overrides.category ?? "CatalogDrafter") !== "Other"
    && ((overrides.verdict as { _tag?: string } | undefined)?._tag ?? "Bound") === "Bound",
  ...overrides,
})

const inspection = (onThisComputer?: string, catalogModel?: string) => Schema.decodeUnknownSync(CandidateInspection)({
  source: { _tag: "HuggingFace", repository: "owner/model-GGUF", commit: "c0ffee" },
  displayName: "",
  versions: [
    {
      label: "Q4_K_M",
      modelId: "hf:owner/model-GGUF/model-Q4_K_M.gguf",
      files: [{ path: "model-Q4_K_M.gguf", sizeBytes: 100 }],
      sizeBytes: 100,
      downloadBytes: 100,
      recommended: true,
      ...(catalogModel === undefined ? {} : { catalogModel }),
      ...(onThisComputer === undefined ? {} : { onThisComputer }),
    },
    {
      label: "Q2_K",
      modelId: "hf:owner/model-GGUF/model-Q2_K.gguf",
      files: [{ path: "model-Q2_K.gguf", sizeBytes: 50 }],
      sizeBytes: 50,
      downloadBytes: 50,
      recommended: false,
      assessment: { _tag: "Unsupported", profile: { contextLength: 4096 }, reasons: [{ _tag: "WeightTypes", types: [{ ggmlType: "Q2_K", tensorCount: 2 }] }], detail: "" },
    },
  ],
})

interface DrafterList {
  readonly candidates: readonly NativeDrafterCandidate[]
  readonly entered: Option.Option<NativeDrafterCandidate["reference"]>
}

/** ICN's drafter list; `entered` is the listed candidate a different drafter resolved to. */
const listed = (candidates: readonly NativeDrafterCandidate[], entered?: NativeDrafterCandidate): DrafterList =>
  ({ candidates, entered: Option.map(Option.fromNullable(entered), ({ reference }) => reference) })

interface Recorded {
  readonly bundles: ModelCandidate[]
  readonly started: unknown[]
  readonly cancelled: string[]
  readonly acknowledged: string[]
  readonly removedFiles: string[]
  readonly drafterRequests: unknown[]
  readonly catalogSyncs: string[]
  pushes: number
  prunes: number
}

const harness = (options: {
  readonly config?: ReturnType<typeof customModelsConfig>
  readonly inspect?: ReturnType<typeof inspection>
  readonly inspectFailure?: unknown
  readonly drafters?: readonly NativeDrafterCandidate[]
  readonly draftersFor?: (request: { readonly other: Option.Option<string> }) => Effect.Effect<DrafterList, unknown>
  readonly downloads?: readonly ModelDownload[]
  readonly customModels?: ReturnType<typeof nativeCustomModel>[]
  readonly catalog?: readonly CatalogModelSource[]
  readonly catalogRows?: readonly LocalModelsState["models"][number][]
  readonly catalogInstalled?: readonly string[]
  readonly startDownload?: (bundle: unknown, forModel: string) => Option.Option<ModelDownload>
} = {}) => {
  const recorded: Recorded = {
    bundles: [], started: [], cancelled: [], acknowledged: [], removedFiles: [], drafterRequests: [], catalogSyncs: [], pushes: 0, prunes: 0,
  }
  const downloadList = Effect.runSync(SubscriptionRef.make<readonly ModelDownload[]>(options.downloads ?? []))
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
    inspect: () => options.inspectFailure === undefined
      ? Effect.succeed(options.inspect ?? inspection())
      : Effect.fail(options.inspectFailure as never),
    drafters: (request) => Effect.suspend(() => {
      recorded.drafterRequests.push(request)
      return options.draftersFor?.(request as { readonly other: Option.Option<string> })
        ?? Effect.succeed(listed(options.drafters ?? [nativeCandidate()]))
    }) as never,
    assess: () => Effect.die("unused"),
    bundle: (candidate) => Effect.sync(() => {
      recorded.bundles.push(candidate)
      return { _tag: "Standalone", package: { id: "bundle" } } as never
    }),
  }
  const layer = Effect.gen(function* () {
    const config = options.config ?? customModelsConfig()
    const custom = yield* makeTestCustomModels(config)
    const rows = sourceModels.flatMap((entry) => Option.toArray(customModel({
      entry, assessment: undefined, instances: [], related: [], catalog: options.catalog ?? [],
      drafterChoice: config.drafters[entry.id] ?? { _tag: "Auto" },
    })))
    const localState: LocalModelsState = {
      preparation: { discovery: { complete: true, modelsFound: rows.length }, assessment: { complete: true, settledModels: 0, totalModels: 0 } },
      models: [...(options.catalogRows ?? []), ...rows],
      unsupported: [],
      hidden: [],
    }
    return Layer.mergeAll(
      Layer.succeed(IcnCandidates, candidates),
      Layer.succeed(IcnDownloads, {
        get: SubscriptionRef.get(downloadList).pipe(Effect.map((list) => ({ revision: 1, state: { downloads: [...list] } }))),
        changes: downloadList.changes,
        initialized: Effect.succeed(true),
        refresh: Effect.void,
        start: (bundle: unknown, forModel: string) => Effect.gen(function* () {
          recorded.started.push({ bundle, forModel })
          const started = options.startDownload?.(bundle, forModel) ?? Option.none()
          if (Option.isSome(started)) yield* SubscriptionRef.update(downloadList, (list) => [...list, started.value])
          return started
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
        pruneDrafters: Effect.sync(() => {
          recorded.prunes += 1
          return { removed: [], freedBytes: 0 }
        }),
      } as never),
      Layer.succeed(LocalModelSources, {
        state: Effect.succeed({ customModels: sourceModels, hiddenModels: [], catalogModels: options.catalog ?? [] }),
      } as never),
      Layer.succeed(LocalModels, { state: Effect.succeed(localState) } as never),
      Layer.succeed(CustomModels, custom),
      Layer.succeed(CustomModelReconciler, { push: Effect.sync(() => { recorded.pushes += 1 }) }),
      Layer.succeed(CatalogModelSync, {
        installed: (modelId) => Effect.succeed((options.catalogInstalled ?? []).includes(modelId)),
        sync: (modelId) => Effect.sync(() => {
          recorded.catalogSyncs.push(modelId)
          return { outcome: "Installing" as const }
        }),
        downloadComponent: () => Effect.succeed({ started: false }),
      }),
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
  return { recorded, run, downloadList }
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

  it("projects an unreadable version without a serving profile", async () => {
    const wire = Schema.encodeSync(CandidateInspection)(inspection())
    const unreadable = Schema.decodeUnknownSync(CandidateInspection)({
      ...wire,
      versions: [wire.versions[0], {
        ...wire.versions[1],
        assessment: { _tag: "Unsupported", reasons: [{ _tag: "Unreadable" }], detail: "file is shorter than its header declares" },
      }],
    })
    const { run } = harness({ inspect: unreadable })
    const result = await run((commands) => commands.inspectCandidate("owner/model-GGUF"))
    expect(result.versions[1]?.assessment).toEqual(Option.some({
      _tag: "Unsupported",
      environmentId: expect.any(String),
      profile: Option.none(),
      reasons: [{ _tag: "Unreadable" }],
      detail: "file is shorter than its header declares",
    }))
  })

  it("reports the inference service's own message, and never a contract failure as unreachable", async () => {
    const integrity = harness({
      inspectFailure: {
        _tag: "GeneratedClientRemoteError",
        status: 422,
        body: { error: { code: "integrity_failed", message: "unknown GGUF tensor type 31" } },
      },
    })
    const rejected = await integrity.run((commands) => commands.inspectCandidate("owner/model-GGUF").pipe(Effect.either))
    expect(rejected._tag === "Left" && rejected.left).toMatchObject({
      _tag: "ModelSourceUnavailable",
      reason: "Unreadable",
      message: "Magnitude couldn't read this file. It may be incomplete, damaged, or use a format Magnitude doesn't support yet.",
      retryable: false,
    })

    const upstream = harness({
      inspectFailure: {
        _tag: "GeneratedClientRemoteError",
        status: 500,
        body: { error: { code: "inventory_error", message: "upstream model service failed: timed out" } },
      },
    })
    const failed = await upstream.run((commands) => commands.inspectCandidate("owner/model-GGUF").pipe(Effect.either))
    expect(failed._tag === "Left" && failed.left).toMatchObject({ reason: "SourceUnreachable", retryable: true })

    const invalid = harness({
      inspectFailure: { _tag: "GeneratedClientInvalidResponseError", status: 418, message: "undeclared status" },
    })
    const unreadable = await invalid.run((commands) => commands.inspectCandidate("owner/model-GGUF").pipe(Effect.either))
    expect(unreadable._tag === "Left" && unreadable.left._tag).toBe("ModelSourceUnavailable")
    expect(unreadable._tag === "Left" && unreadable.left.message).not.toMatch(/reach|connection/i)

    const transport = harness({ inspectFailure: { _tag: "GeneratedClientTransportError", cause: new Error("refused") } })
    const unreachable = await transport.run((commands) => commands.inspectCandidate("owner/model-GGUF").pipe(Effect.either))
    expect(unreachable._tag === "Left" && unreachable.left).toMatchObject({ _tag: "ModelSourceUnavailable", reason: "ServiceUnreachable", retryable: true })
  })

  it("asks for Hugging Face access and a token for a gated repository", async () => {
    const gated = harness({
      inspectFailure: {
        _tag: "GeneratedClientRemoteError",
        status: 409,
        body: { error: { code: "hugging_face_access_required", message: "GGUF header request returned HTTP 401 Unauthorized" } },
      },
    })
    const result = await gated.run((commands) => commands.addHuggingFaceModel({
      repository, revision: Option.none(), version: Option.none(), drafter: { _tag: "Auto" }, force: false
    }).pipe(Effect.either))
    expect(result._tag === "Left" && result.left).toMatchObject({
      _tag: "ModelSourceUnavailable",
      reason: "AccessRequired",
      message: "This repository requires accepting its terms on Hugging Face and an access token. Set HF_TOKEN to a token from an account that has accepted them, then try again.",
      retryable: false,
    })
  })

  it("says a repository Hugging Face won't show may be missing, private or gated, for inspect and add", async () => {
    const missing = harness({
      inspectFailure: {
        _tag: "GeneratedClientRemoteError",
        status: 409,
        body: { error: { code: "hugging_face_repository_not_found", message: "Repository not found" } },
      },
    })
    const message = "Repository not found, or it's private or gated. Check the name, or set HF_TOKEN if you have access."
    const inspected = await missing.run((commands) => commands.inspectCandidate("nobody-xyz/does-not-exist-GGUF").pipe(Effect.either))
    expect(inspected._tag === "Left" && inspected.left).toMatchObject({
      _tag: "ModelRepositoryNotFound",
      reference: "nobody-xyz/does-not-exist-GGUF",
      message,
    })
    const added = await missing.run((commands) => commands.addHuggingFaceModel({
      repository, revision: Option.none(), version: Option.none(), drafter: { _tag: "Auto" }, force: false
    }).pipe(Effect.either))
    expect(added._tag === "Left" && added.left).toMatchObject({ _tag: "ModelRepositoryNotFound", message })
  })

  it("refuses a version the repository does not have", async () => {
    const { run } = harness()
    const result = await run((commands) => commands.inspectCandidate("owner/model-GGUF:IQ1_M").pipe(Effect.either))
    expect(result._tag === "Left" && result.left._tag).toBe("InvalidModelReference")
  })

  it("keeps ICN's drafter order and passes a different drafter to the engine", async () => {
    const { run, recorded } = harness({
      drafters: [
        nativeCandidate({ category: "NextToModel", reference: { _tag: "Local", path: "/d/large.gguf" }, label: "Large", sizeBytes: 9 }),
        nativeCandidate({ category: "NextToModel", reference: { _tag: "Local", path: "/d/small.gguf" }, label: "Small", sizeBytes: 1 }),
        nativeCandidate(),
        nativeCandidate({ category: "BuiltIn", reference: { _tag: "Embedded" }, label: "Built in", lineage: "Exact", sizeBytes: 0 }),
      ],
    })
    const result = await run((commands) => commands.listDrafterCandidates({ _tag: "Model", modelId: foundId }, Option.some("z-lab/d:dflash.gguf")).pipe(Effect.map(({ candidates }) => candidates)))
    expect(result.map(({ label }) => label)).toEqual(["Large", "Small", "Magnitude DFlash", "Built in"])
    expect(recorded.drafterRequests[0]).toMatchObject({ _tag: "Model", modelId: foundId, other: Option.some("z-lab/d:dflash.gguf") })
  })
})

describe("a drafter entered by hand", () => {
  const hand = { _tag: "Local", path: "/drafters/hand-dspark.gguf" } as const
  const handBound = { _tag: "Bound", method: { _tag: "DSpark" }, memoryBytes: 1 }
  const pinnedByHand = customModelsConfig({ drafters: { [foundId]: { _tag: "Reference", reference: hand, method: { _tag: "DSpark" } } } })
  const entered = nativeCandidate({
    category: "Other", reference: hand, name: "hand-dspark.gguf", label: "Untested", lineage: "None",
    method: { _tag: "DSpark" }, onThisComputer: true, sizeBytes: 5, verdict: handBound,
  })

  it("is listed as Other with its verdict after ICN's list when ICN doesn't list it", async () => {
    const { run, recorded } = harness({
      config: pinnedByHand,
      draftersFor: ({ other }) => Effect.succeed(Option.isSome(other) ? listed([nativeCandidate(), entered], entered) : listed([nativeCandidate()])),
    })
    const result = await run((commands) => commands.listDrafterCandidates({ _tag: "Model", modelId: foundId }, Option.none()).pipe(Effect.map(({ candidates }) => candidates)))
    expect(result.map(({ category, reference }) => [category, reference])).toEqual([
      ["CatalogDrafter", drafterReference],
      ["Other", hand],
    ])
    expect(result[1]?.verdict).toEqual(handBound)
    expect(recorded.drafterRequests).toHaveLength(2)
    expect(recorded.drafterRequests[1]).toMatchObject({ _tag: "Model", modelId: foundId, other: Option.some(hand.path) })
  })

  it("is not checked again when ICN lists it, and is left out when it can't be checked", async () => {
    const nextToModel = harness({ config: pinnedByHand, drafters: [nativeCandidate(), nativeCandidate({ category: "NextToModel", reference: hand })] })
    await nextToModel.run((commands) => commands.listDrafterCandidates({ _tag: "Model", modelId: foundId }, Option.none()).pipe(Effect.map(({ candidates }) => candidates)))
    expect(nextToModel.recorded.drafterRequests).toHaveLength(1)
    const failing = harness({
      config: pinnedByHand,
      draftersFor: ({ other }) => Option.isSome(other) ? Effect.fail({ _tag: "NotFound", message: "gone" }) : Effect.succeed(listed([nativeCandidate()])),
    })
    const result = await failing.run((commands) => commands.listDrafterCandidates({ _tag: "Model", modelId: foundId }, Option.none()).pipe(Effect.map(({ candidates }) => candidates)))
    expect(result.map(({ category }) => category)).toEqual(["CatalogDrafter"])
  })
})

describe("a drafter entered by hand that is a listed candidate's file", () => {
  const catalogFile = Schema.decodeUnknownSync(DrafterReferenceSchema)({
    _tag: "HuggingFace", repository: "magnitudedev/Model-DFlash-GGUF", commit: "d1", path: "Model-DFlash-Q8_0.gguf",
  })
  const catalog = nativeCandidate({ onThisComputer: true })
  const resolvesToCatalog = ({ other }: { readonly other: Option.Option<string> }) =>
    Effect.succeed(Option.isSome(other) ? listed([catalog], catalog) : listed([catalog]))

  it("adds no Other row when the pinned file resolves to the listed catalog drafter", async () => {
    const { run } = harness({
      config: customModelsConfig({ drafters: { [foundId]: { _tag: "Reference", reference: catalogFile, method: { _tag: "DFlash" } } } }),
      draftersFor: resolvesToCatalog,
    })
    const result = await run((commands) => commands.listDrafterCandidates({ _tag: "Model", modelId: foundId }, Option.none()).pipe(Effect.map(({ candidates }) => candidates)))
    expect(result.map(({ category, reference }) => [category, reference])).toEqual([["CatalogDrafter", drafterReference]])
  })

  it("attaches the listed catalog drafter when its file is chosen by its Hugging Face path", async () => {
    const recommendedDrafter = {
      category: "CatalogDrafter", reference: drafterReference, name: "Magnitude DFlash for Model", label: "Magnitude DFlash",
      lineage: "Lineage", method: { _tag: "DFlash" }, onThisComputer: true, sizeBytes: 900, verdict: bound, automatic: true,
    }
    const recommended = harness({
      config: customModelsConfig({ drafters: {} }),
      customModels: [nativeCustomModel({ recommendedDrafter })],
      draftersFor: resolvesToCatalog,
    })
    const chosen = await recommended.run((commands, models) => commands.setModelDrafter(foundId, { _tag: "Reference", reference: catalogFile }).pipe(
      Effect.zipWith(models.get, (result, config) => ({ result, saved: config.drafters[foundId] })),
    ))
    expect(Option.map(chosen.result.drafter, ({ reference }) => reference)).toEqual(Option.some(drafterReference))
    expect(chosen.saved).toEqual({ _tag: "Auto" })

    const notRecommended = harness({ config: customModelsConfig({ drafters: {} }), draftersFor: resolvesToCatalog })
    const saved = await notRecommended.run((commands, models) => commands.setModelDrafter(foundId, { _tag: "Reference", reference: catalogFile }).pipe(
      Effect.zipRight(models.get),
    ))
    expect(saved.drafters[foundId]).toEqual({ _tag: "Reference", reference: drafterReference, method: { _tag: "DFlash" } })
  })
})

describe("adding models", () => {
  it("declares an added model with its drafter and downloads the bundle ICN builds", async () => {
    const { run, recorded } = harness()
    const result = await run((commands, models) => Effect.gen(function* () {
      const added = yield* commands.addHuggingFaceModel({ repository, revision: Option.none(), version: Option.none(), drafter: { _tag: "Auto" }, force: false })
      return { added, config: yield* models.get }
    }))
    expect(result.added).toMatchObject({ modelId: addedId, version: "Q4_K_M", outcome: "Started" })
    expect(result.config.declarations[addedId]?.origin).toEqual({ _tag: "Added", repository: "owner/model-GGUF", commit: "c0ffee", pendingCommit: Option.none() })
    expect(result.config.drafters[addedId]).toEqual({ _tag: "Auto" })
    expect(recorded.bundles[0]).toMatchObject({
      source: { _tag: "HuggingFace", repository: "owner/model-GGUF", commit: "c0ffee" },
      target: ["model-Q4_K_M.gguf"],
      drafter: Option.some({ method: { _tag: "DFlash" }, reference: drafterReference }),
    })
    expect(recorded.started).toEqual([{ bundle: { _tag: "Standalone", package: { id: "bundle" } }, forModel: addedId }])
    expect(recorded.pushes).toBeGreaterThan(0)
  })

  it("adds a version of a catalog model's repository that isn't its exact target as a custom model whose catalog drafter is Auto", async () => {
    const catalogDrafter = { _tag: "Catalog", modelId: CatalogFormModelIdSchema.make("qwen3.5-4b:gguf:q4") } as const
    const { run, recorded } = harness({ drafters: [nativeCandidate({ reference: catalogDrafter, lineage: "Exact" })] })
    const result = await run((commands, models) => Effect.gen(function* () {
      const added = yield* commands.addHuggingFaceModel({ repository, revision: Option.none(), version: Option.some("Q4_K_M"), drafter: { _tag: "Auto" }, force: false })
      return { added, config: yield* models.get }
    }))
    expect(result.added).toMatchObject({ _tag: "Custom", modelId: addedId, outcome: "Started" })
    expect(result.config.declarations[addedId]?.origin._tag).toBe("Added")
    expect(result.config.drafters[addedId]).toEqual({ _tag: "Auto" })
    expect(recorded.catalogSyncs).toEqual([])
    expect(recorded.bundles[0]?.drafter).toEqual(Option.some({ method: { _tag: "DFlash" }, reference: catalogDrafter }))
  })

  it("pins a drafter the user names when adding", async () => {
    const other = { _tag: "Local", path: "/drafters/dspark.gguf" } as const
    const { run } = harness({ drafters: [nativeCandidate({
      category: "Other", reference: other, method: { _tag: "DSpark" }, verdict: { _tag: "Bound", method: { _tag: "DSpark" }, memoryBytes: 1 },
    })] })
    const config = await run((commands, models) => commands.addHuggingFaceModel({
      repository, revision: Option.none(), version: Option.none(), drafter: { _tag: "Reference", reference: other }, force: false,
    }).pipe(Effect.zipRight(models.get)))
    expect(config.drafters[addedId]).toEqual({ _tag: "Reference", reference: other, method: { _tag: "DSpark" } })
  })

  it("installs the catalog model a version exactly is instead of declaring a Hugging Face model", async () => {
    const catalogModel = "qwen3.6-35b-a3b:gguf:q4"
    const { run, recorded } = harness({ inspect: inspection(undefined, catalogModel) })
    const result = await run((commands, models) => Effect.gen(function* () {
      const before = yield* models.get
      const added = yield* commands.addHuggingFaceModel({ repository, revision: Option.none(), version: Option.none(), drafter: { _tag: "Auto" }, force: false })
      return { added, before, config: yield* models.get }
    }))
    expect(result.added).toEqual({ _tag: "Catalog", modelId: catalogModel, version: "Q4_K_M", outcome: "Started" })
    expect(recorded.catalogSyncs).toEqual([catalogModel])
    expect(result.config).toEqual(result.before)
    expect(recorded.started).toEqual([])
  })

  it("reports an installed catalog model as already on this computer and installs nothing", async () => {
    const catalogModel = "qwen3.6-35b-a3b:gguf:q4"
    const { run, recorded } = harness({ inspect: inspection(catalogModel, catalogModel), catalogInstalled: [catalogModel] })
    const added = await run((commands) => commands.addHuggingFaceModel({
      repository, revision: Option.none(), version: Option.some("Q4_K_M"), drafter: { _tag: "Auto" }, force: false,
    }))
    expect(added).toEqual({ _tag: "Catalog", modelId: catalogModel, version: "Q4_K_M", outcome: "AlreadyOnThisComputer" })
    expect(recorded.catalogSyncs).toEqual([])
    expect(recorded.started).toEqual([])
  })

  it("refuses a drafter choice for a catalog model, which keeps its own", async () => {
    const { run, recorded } = harness({ inspect: inspection(undefined, "qwen3.6-35b-a3b:gguf:q4") })
    const result = await run((commands) => commands.addHuggingFaceModel({
      repository, revision: Option.none(), version: Option.none(), drafter: { _tag: "None" }, force: false,
    }).pipe(Effect.either))
    expect(result._tag === "Left" && result.left).toMatchObject({
      _tag: "InvalidModelReference",
      message: "owner/model-GGUF is the catalog model qwen3.6-35b-a3b:gguf:q4, which comes with its drafter.",
    })
    expect(recorded.catalogSyncs).toEqual([])
  })

  it("downloads only the target when the chosen drafter is already on this computer", async () => {
    const { run, recorded } = harness({ drafters: [nativeCandidate({ onThisComputer: true })] })
    const result = await run((commands, models) => Effect.gen(function* () {
      const added = yield* commands.addHuggingFaceModel({ repository, revision: Option.none(), version: Option.none(), drafter: { _tag: "Auto" }, force: false })
      return { added, config: yield* models.get }
    }))
    expect(result.added._tag === "Custom" && result.added.drafter.pipe(Option.map(({ onThisComputer }) => onThisComputer))).toEqual(Option.some(true))
    expect(result.config.drafters[addedId]).toEqual({ _tag: "Auto" })
    expect(recorded.bundles[0]?.drafter).toEqual(Option.none())
  })

  it("refuses an unsupported version before anything downloads", async () => {
    const { run, recorded } = harness()
    const result = await run((commands) => commands.addHuggingFaceModel({
      repository, revision: Option.none(), version: Option.some("Q2_K"), drafter: { _tag: "None" }, force: false
    }).pipe(Effect.either))
    expect(result._tag === "Left" && result.left).toMatchObject({
      _tag: "ModelVersionNotRunnable",
      version: "Q2_K",
      verdict: { _tag: "Unsupported", reasons: [{ _tag: "WeightTypes" }] },
    })
    expect(recorded.started).toEqual([])
  })

  it("refuses a version that needs more memory than this computer can provide unless forced", async () => {
    const tooLarge = Schema.decodeUnknownSync(CandidateInspection)({
      source: { _tag: "HuggingFace", repository: "owner/model-GGUF", commit: "c0ffee" },
      displayName: "",
      versions: [{
        label: "Q8_0",
        modelId: "hf:owner/model-GGUF/model-Q8_0.gguf",
        files: [{ path: "model-Q8_0.gguf", sizeBytes: 100 }],
        sizeBytes: 100,
        downloadBytes: 100,
        recommended: false,
        assessment: {
          _tag: "DoesNotFit",
          assessmentId: "assessment",
          profile: { contextLength: 4096 },
          memory: [{ memoryDomainId: "system", capacityBytes: 8, compatibilityReserveBytes: 0, remainingBytes: -4, requiredBytes: 12 }],
          limitingResource: "system",
          deficitBytes: 4,
          drafter: { _tag: "None" },
        },
      }],
    })
    const refused = harness({ inspect: tooLarge })
    const result = await refused.run((commands) => commands.addHuggingFaceModel({
      repository, revision: Option.none(), version: Option.some("Q8_0"), drafter: { _tag: "None" }, force: false,
    }).pipe(Effect.either))
    expect(result._tag === "Left" && result.left).toMatchObject({
      _tag: "ModelVersionNotRunnable",
      version: "Q8_0",
      verdict: { _tag: "DoesNotFit", deficitBytes: 4 },
    })
    expect(refused.recorded.started).toEqual([])

    const forced = harness({ inspect: tooLarge })
    const added = await forced.run((commands) => commands.addHuggingFaceModel({
      repository, revision: Option.none(), version: Option.some("Q8_0"), drafter: { _tag: "None" }, force: true,
    }))
    expect(added).toMatchObject({ version: "Q8_0", outcome: "Started" })
    expect(forced.recorded.started).toHaveLength(1)
  })

  it("adds a version already on this computer by fetching only its drafter", async () => {
    const { run, recorded } = harness({ inspect: inspection(foundId) })
    const result = await run((commands, models) => Effect.gen(function* () {
      const added = yield* commands.addHuggingFaceModel({ repository, revision: Option.none(), version: Option.none(), drafter: { _tag: "Auto" }, force: false })
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
        versions: [{ label: "Q4_0", modelId: importedId, files: [{ path: "/models/imported.gguf", sizeBytes: 1 }], sizeBytes: 1, downloadBytes: 0, recommended: true }],
      }),
      config: customModelsConfig({ declarations: {} }),
    })
    const result = await run((commands, models) => Effect.gen(function* () {
      const imported = yield* commands.importModelFile("/models/imported.gguf", { _tag: "None" })
      return { imported, config: yield* models.get }
    }))
    expect(result.imported).toMatchObject({ modelId: importedId, outcome: "Started" })
    expect(result.config.declarations[importedId]?.origin).toMatchObject({ _tag: "Imported", path: "/models/imported.gguf" })
    expect(result.config.drafters[importedId]).toEqual({ _tag: "None" })
  })

  it("imports a copy of a found model as that model: one more location, never a replacement", async () => {
    const config = customModelsConfig({ declarations: {}, drafters: { [foundId]: { _tag: "None" } } })
    const { run, recorded } = harness({
      inspect: Schema.decodeUnknownSync(CandidateInspection)({
        source: { _tag: "Local", path: "/copies/copy.gguf" },
        displayName: "Copy",
        versions: [{
          label: "Q8_0", modelId: importedId, onThisComputer: foundId,
          files: [{ path: "/copies/copy.gguf", sizeBytes: 1 }], sizeBytes: 1, downloadBytes: 0, recommended: true,
        }],
      }),
      config,
    })
    const result = await run((commands, models) => Effect.gen(function* () {
      const imported = yield* commands.importModelFile("/copies/copy.gguf", { _tag: "Auto" })
      return { imported, config: yield* models.get }
    }))
    expect(result.imported).toMatchObject({ modelId: foundId, outcome: "AlreadyOnThisComputer" })
    expect(result.config.declarations[importedId]?.origin).toMatchObject({ _tag: "Imported", path: "/copies/copy.gguf" })
    expect(result.config.declarations[foundId]).toBeUndefined()
    expect(result.config.drafters[foundId]).toEqual({ _tag: "None" })
    expect(result.config.drafters[importedId]).toBeUndefined()
    expect(recorded.started).toEqual([])
  })
})

describe("importing a catalog model's bytes", () => {
  const catalogModel = "minicpm5-1b:gguf:q4"
  const catalogBytes = (path: string) => Schema.decodeUnknownSync(CandidateInspection)({
    source: { _tag: "Local", path },
    displayName: "Renamed",
    versions: [{
      label: "Q4_K_M", modelId: importedId, catalogModel, onThisComputer: catalogModel,
      files: [{ path, sizeBytes: 1 }], sizeBytes: 1, downloadBytes: 0, recommended: true,
    }],
  })

  // QA (linux-debian, round 3c): a link to a file holding the catalog target's bytes returned
  // "Started" with a `local:` ID that no listing knew.
  it("is that catalog model, already on this computer, and records nothing", async () => {
    const config = customModelsConfig({ declarations: {}, drafters: {} })
    const { run, recorded } = harness({ inspect: catalogBytes("/stash/link-to-renamed.gguf"), config })
    const result = await run((commands, models) => Effect.gen(function* () {
      const imported = yield* commands.importModelFile("/stash/link-to-renamed.gguf", { _tag: "Auto" })
      return { imported, config: yield* models.get }
    }))
    expect(result.imported).toEqual({
      modelId: catalogModel, drafter: Option.none(), outcome: "AlreadyOnThisComputer", drafterDownloadStarted: false,
    })
    expect(result.config.declarations).toEqual({})
    expect(result.config.drafters).toEqual({})
    expect(recorded.started).toEqual([])
  })

  it("refuses a drafter choice, since the catalog model keeps its own", async () => {
    const { run } = harness({ inspect: catalogBytes("/models/copy.gguf"), config: customModelsConfig({ declarations: {} }) })
    const failure = await run((commands) => commands.importModelFile("/models/copy.gguf", { _tag: "None" }).pipe(Effect.flip))
    expect(failure.message).toContain(`is the catalog model ${catalogModel}`)
  })
})

describe("importing through a link", () => {
  const otherImportedId = CustomFormModelIdSchema.make(`local:${"b".repeat(64)}`)
  const linkedFiles = () => {
    const root = mkdtempSync(join(tmpdir(), "magnitude-links-"))
    const file = join(root, "store", "model-Q4_K_M.gguf")
    const link = join(root, "link.gguf")
    const second = join(root, "second-link.gguf")
    mkdirSync(join(root, "store"))
    writeFileSync(file, "gguf")
    symlinkSync(file, link)
    symlinkSync(link, second)
    return { root, file, link, second }
  }
  const linkInspection = (path: string, onThisComputer?: string) => Schema.decodeUnknownSync(CandidateInspection)({
    source: { _tag: "Local", path },
    displayName: "Linked",
    versions: [{
      label: "Q4_K_M", modelId: importedId, ...(onThisComputer === undefined ? {} : { onThisComputer }),
      files: [{ path, sizeBytes: 4 }], sizeBytes: 4, downloadBytes: 0, recommended: true,
    }],
  })
  const importedDeclaration = (path: string) => ({ origin: { _tag: "Imported", path, headerDigest: "b".repeat(64) }, target: [path] })

  it("declares nothing for a link to a file a found model already lists", async () => {
    const { root, file, link } = linkedFiles()
    try {
      const { run } = harness({
        config: customModelsConfig({ declarations: {}, drafters: {} }),
        customModels: [nativeCustomModel({ files: [{ path: file, sizeBytes: 4 }] })],
        inspect: linkInspection(link, foundId),
      })
      const result = await run((commands, models) => Effect.gen(function* () {
        const imported = yield* commands.importModelFile(link, { _tag: "None" })
        return { imported, config: yield* models.get }
      }))
      expect(result.imported).toMatchObject({ modelId: foundId, outcome: "AlreadyOnThisComputer" })
      expect(result.config.declarations).toEqual({})
    } finally {
      rmSync(root, { recursive: true, force: true })
    }
  })

  it("declares a file reached through a second link only once", async () => {
    const { root, link, second } = linkedFiles()
    try {
      const { run } = harness({
        config: customModelsConfig({ declarations: { [otherImportedId]: importedDeclaration(link) }, drafters: {} }),
        customModels: [],
        inspect: linkInspection(second),
      })
      const config = await run((commands, models) => commands.importModelFile(second, { _tag: "None" }).pipe(Effect.zipRight(models.get)))
      expect(Object.keys(config.declarations)).toEqual([otherImportedId])
    } finally {
      rmSync(root, { recursive: true, force: true })
    }
  })

  it("forgets a declaration left by a link when the model it names is removed", async () => {
    const { root, file, link } = linkedFiles()
    try {
      const { run } = harness({
        config: customModelsConfig({
          declarations: { [otherImportedId]: importedDeclaration(link) },
          drafters: { [otherImportedId]: { _tag: "None" } },
        }),
        customModels: [nativeCustomModel({ files: [{ path: file, sizeBytes: 4 }] })],
      })
      const config = await run((commands, models) => commands.remove(foundId).pipe(Effect.zipRight(models.get)))
      expect(config.hidden).toEqual([foundId])
      expect(config.declarations).toEqual({})
      expect(config.drafters).toEqual({})
    } finally {
      rmSync(root, { recursive: true, force: true })
    }
  })
})

describe("hiding a model that isn't on this computer", () => {
  it("fails for an unknown custom or catalog ID and changes nothing", async () => {
    const { run } = harness({ config: customModelsConfig({ declarations: {} }) })
    const unknownCustom = CustomFormModelIdSchema.make(`local:${"c".repeat(64)}`)
    const unknownCatalog = CatalogFormModelIdSchema.make("no-such-model:gguf:q4")
    const result = await run((commands, models) => Effect.gen(function* () {
      const custom = yield* commands.hide(unknownCustom).pipe(Effect.flip)
      const catalog = yield* commands.hide(unknownCatalog).pipe(Effect.flip)
      return { custom, catalog, hidden: (yield* models.get).hidden }
    }))
    expect(result.custom).toMatchObject({ code: "custom_model_not_found", message: `${unknownCustom} is not a model on this computer.` })
    expect(result.catalog).toMatchObject({ code: "custom_model_not_found", message: `${unknownCatalog} is not a model on this computer.` })
    expect(result.hidden).toEqual([])
  })

  it("hides a catalog model named by its alias", async () => {
    const external = CatalogFormModelIdSchema.make("qwen3.6-35b-a3b:gguf:q4")
    const alias = CustomFormModelIdSchema.make("hf:unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf")
    const { run } = harness({
      catalogRows: [{
        _tag: "Catalog",
        modelId: external,
        acquisitionState: {
          _tag: "Installed",
          installation: { _tag: "Resolved", installedBytes: 1, primaryPath: "/nonexistent-home/m.gguf", ownership: "External" },
          residencyState: { _tag: "Unloaded" },
        },
      } as unknown as LocalModelsState["models"][number]],
      catalog: [{ id: external, source: { aliases: Option.some([alias]) } }] as unknown as readonly CatalogModelSource[],
    })
    const hidden = await run((commands, models) => commands.hide(alias).pipe(Effect.zipRight(models.get), Effect.map(({ hidden }) => hidden)))
    expect(hidden).toEqual([alias])
  })
})

describe("importing a file Magnitude can't run", () => {
  const unrunnable = (fields: Record<string, unknown>) => Schema.decodeUnknownSync(CandidateInspection)({
    source: { _tag: "Local", path: "/models/deepseek.gguf" },
    displayName: "DeepSeek",
    versions: [],
    ...fields,
  })

  it("refuses an unsupported version with the engine's reasons and declares nothing", async () => {
    const { run, recorded } = harness({
      config: customModelsConfig({ declarations: {} }),
      inspect: unrunnable({ versions: [{
        label: "IQ1_M", modelId: importedId, files: [{ path: "/models/deepseek.gguf", sizeBytes: 1 }], sizeBytes: 1, downloadBytes: 0, recommended: false,
        assessment: { _tag: "Unsupported", profile: { contextLength: 4096 }, reasons: [{ _tag: "WeightTypes", types: [{ ggmlType: "IQ1_M", tensorCount: 3 }] }], detail: "" },
      }] }),
    })
    const result = await run((commands, models) => Effect.gen(function* () {
      const imported = yield* commands.importModelFile("/models/deepseek.gguf", { _tag: "Auto" }).pipe(Effect.either)
      return { imported, config: yield* models.get }
    }))
    expect(result.imported._tag === "Left" && result.imported.left).toMatchObject({
      _tag: "ModelVersionNotRunnable",
      version: "deepseek.gguf",
      verdict: { _tag: "Unsupported", reasons: [{ _tag: "WeightTypes", types: [{ ggmlType: "IQ1_M", tensorCount: 3 }] }] },
    })
    expect(result.config.declarations).toEqual({})
    expect(recorded.started).toEqual([])
  })

  it("refuses an architecture without kernels with its reason", async () => {
    const { run } = harness({
      config: customModelsConfig({ declarations: {} }),
      inspect: unrunnable({ unsupportedArchitecture: [{ _tag: "Architecture", architecture: "deepseek4" }] }),
    })
    const result = await run((commands) => commands.importModelFile("/models/deepseek.gguf", { _tag: "Auto" }).pipe(Effect.either))
    expect(result._tag === "Left" && result.left).toMatchObject({
      _tag: "ModelVersionNotRunnable",
      version: "deepseek.gguf",
      verdict: { _tag: "Unsupported", reasons: [{ _tag: "Architecture", architecture: "deepseek4" }] },
    })
  })
})

describe("adding a repository whose architecture has no kernels", () => {
  it("refuses with the engine's reasons", async () => {
    const { run, recorded } = harness({
      inspect: Schema.decodeUnknownSync(CandidateInspection)({
        source: { _tag: "HuggingFace", repository: "owner/model-GGUF", commit: "c0ffee" },
        displayName: "DeepSeek",
        versions: [],
        unsupportedArchitecture: [{ _tag: "Architecture", architecture: "deepseek4" }],
      }),
    })
    const result = await run((commands) => commands.addHuggingFaceModel({
      repository, revision: Option.none(), version: Option.none(), drafter: { _tag: "Auto" }, force: false,
    }).pipe(Effect.either))
    expect(result._tag === "Left" && result.left).toMatchObject({
      _tag: "ModelVersionNotRunnable",
      version: "owner/model-GGUF",
      verdict: { _tag: "Unsupported", reasons: [{ _tag: "Architecture", architecture: "deepseek4" }] },
    })
    expect(recorded.started).toEqual([])
  })
})

describe("importing a folder", () => {
  it("refuses a folder as an imported model and declares nothing", async () => {
    const folder = mkdtempSync(join(tmpdir(), "gguf-import-"))
    try {
      const { run, recorded } = harness({ config: customModelsConfig({ declarations: {} }) })
      const result = await run((commands, models) => Effect.gen(function* () {
        const imported = yield* commands.importModelFile(folder, { _tag: "Auto" }).pipe(Effect.either)
        return { imported, config: yield* models.get }
      }))
      expect(result.imported._tag === "Left" && result.imported.left).toMatchObject({
        _tag: "InvalidModelReference",
        message: `${folder} is a folder. Choose a GGUF file, or add the folder as a model location.`,
      })
      expect(result.config.declarations).toEqual({})
      expect(recorded.started).toEqual([])
    } finally {
      rmSync(folder, { recursive: true, force: true })
    }
  })
})

describe("drafter changes", () => {
  it("refuses a drafter the engine rejects", async () => {
    const { run } = harness({
      drafters: [nativeCandidate({ verdict: { _tag: "Rejected", reasons: [{ _tag: "TargetMismatch", detail: "x" }], detail: "hidden size" } })],
    })
    const result = await run((commands) => commands.setModelDrafter(foundId, { _tag: "Reference", reference: drafterReference }).pipe(Effect.either))
    expect(result._tag === "Left" && result.left).toMatchObject({
      _tag: "DrafterNotUsable",
      reasons: [{ _tag: "TargetMismatch", detail: "x" }],
    })
    expect(JSON.stringify(result)).not.toContain("hidden size")
  })

  it("turns a drafter off without downloading anything", async () => {
    const { run, recorded } = harness()
    const result = await run((commands, models) => commands.setModelDrafter(foundId, { _tag: "None" }).pipe(
      Effect.zipRight(models.get),
    ))
    expect(result.drafters[foundId]).toEqual({ _tag: "None" })
    expect(recorded.bundles).toEqual([])
    expect(recorded.prunes).toBe(0)
  })

  it("never prunes on any drafter change: built-in heads, another drafter, none or auto", async () => {
    const builtIn = { _tag: "Embedded" } as const
    const { run, recorded } = harness({
      drafters: [
        nativeCandidate({ onThisComputer: true }),
        nativeCandidate({ category: "BuiltIn", reference: builtIn, label: "Built in", lineage: "Exact", sizeBytes: 0, method: { _tag: "Mtp" }, verdict: { _tag: "Bound", method: { _tag: "Mtp" }, memoryBytes: 1 } }),
      ],
    })
    await run((commands) => Effect.gen(function* () {
      yield* commands.setModelDrafter(foundId, { _tag: "Reference", reference: builtIn })
      yield* commands.setModelDrafter(foundId, { _tag: "Reference", reference: drafterReference })
      yield* commands.setModelDrafter(foundId, { _tag: "None" })
      yield* commands.setModelDrafter(foundId, { _tag: "Auto" })
    }))
    expect(recorded.prunes).toBe(0)
    expect(recorded.removedFiles).toEqual([])
  })

  it("keeps the drafter's files when it is turned off, so turning it back on downloads nothing", async () => {
    const { run, recorded } = harness({ drafters: [nativeCandidate({ onThisComputer: true })] })
    const chosen = await run((commands) => commands.setModelDrafter(foundId, { _tag: "None" }).pipe(
      Effect.zipRight(commands.setModelDrafter(foundId, { _tag: "Auto" })),
    ))
    expect(recorded.prunes).toBe(0)
    expect(recorded.removedFiles).toEqual([])
    expect(chosen.drafterDownloadStarted).toBe(false)
    expect(recorded.started).toEqual([])
  })

  it("says a drafter download started only when one did", async () => {
    const onDisk = harness({ drafters: [nativeCandidate({ category: "NextToModel", onThisComputer: true })] })
    const chosen = await onDisk.run((commands) => commands.setModelDrafter(foundId, { _tag: "Auto" }))
    expect(chosen.drafterDownloadStarted).toBe(false)
    expect(onDisk.recorded.started).toEqual([])
    const missing = harness()
    expect((await missing.run((commands) => commands.setModelDrafter(foundId, { _tag: "Auto" }))).drafterDownloadStarted).toBe(true)
    expect(missing.recorded.started).toHaveLength(1)
  })

  it("keeps Auto, and picking the recommended drafter, saved as Auto; another drafter is pinned with its method", async () => {
    const other = { _tag: "Local", path: "/drafters/dspark.gguf" } as const
    const recommendedDrafter = {
      category: "CatalogDrafter", reference: drafterReference, name: "Magnitude DFlash for Model", label: "Magnitude DFlash",
      lineage: "Lineage", method: { _tag: "DFlash" }, onThisComputer: true, sizeBytes: 900, verdict: bound, automatic: true,
    }
    const { run } = harness({
      customModels: [nativeCustomModel({ recommendedDrafter })],
      drafters: [
        nativeCandidate({ onThisComputer: true }),
        nativeCandidate({
          category: "NextToModel", reference: other, method: { _tag: "DSpark" }, onThisComputer: true,
          verdict: { _tag: "Bound", method: { _tag: "DSpark" }, memoryBytes: 1 },
        }),
      ],
    })
    const saved = await run((commands, models) => Effect.gen(function* () {
      yield* commands.setModelDrafter(foundId, { _tag: "Reference", reference: drafterReference })
      const recommended = (yield* models.get).drafters[foundId]
      yield* commands.setModelDrafter(foundId, { _tag: "Reference", reference: other })
      const pinned = (yield* models.get).drafters[foundId]
      yield* commands.setModelDrafter(foundId, { _tag: "Auto" })
      const automatic = (yield* models.get).drafters[foundId]
      return { recommended, pinned, automatic }
    }))
    expect(saved.recommended).toEqual({ _tag: "Auto" })
    expect(saved.pinned).toEqual({ _tag: "Reference", reference: other, method: { _tag: "DSpark" } })
    expect(saved.automatic).toEqual({ _tag: "Auto" })
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
    expect(recorded.prunes).toBe(2)
    expect(Object.keys(config.declarations)).toEqual([])
    expect(config.hidden).toEqual([foundId])
  })

  it("removes only Magnitude's copy of an added model also found in llama.cpp, and never hides it", async () => {
    const { run, recorded } = harness({
      customModels: [nativeCustomModel({
        modelId: addedId,
        origin: { _tag: "Added", repository: "owner/model-GGUF", commit: "c0ffee" },
        files: [{ path: "/store/model-Q4_K_M.gguf", sizeBytes: 1_000 }],
        state: {
          _tag: "Ready",
          catalogAttribution: { _tag: "NotInCatalog" },
          installation: { _tag: "Resolved", installedBytes: 1_000, ownership: "Mixed", primaryPath: "/store/model-Q4_K_M.gguf" },
          model: readyModel,
        },
      })],
    })
    const config = await run((commands, models) => commands.remove(addedId).pipe(Effect.zipRight(models.get)))
    expect(recorded.removedFiles).toEqual([addedId])
    expect(config.declarations[addedId]).toBeUndefined()
    expect(config.drafters[addedId]).toBeUndefined()
    expect(config.hidden).toEqual([])
  })

  it("refuses to hide a model added from Hugging Face or imported, and hides a found one", async () => {
    const { run } = harness()
    const result = await run((commands, models) => Effect.gen(function* () {
      const added = yield* commands.hide(addedId).pipe(Effect.flip)
      const imported = yield* commands.hide(importedId).pipe(Effect.flip)
      yield* commands.hide(foundId)
      return { added, imported, hidden: (yield* models.get).hidden }
    }))
    expect(result.added).toMatchObject({
      code: "managed_model_not_hideable",
      message: `${addedId} was added to Magnitude, so it can't be hidden. Remove it with \`magnitude catalog remove ${addedId}\` instead.`,
    })
    expect(result.imported).toMatchObject({ code: "managed_model_not_hideable" })
    expect(result.imported.message).toContain("was imported into Magnitude")
    expect(result.hidden).toEqual([foundId])
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
    expect(result.outcome).toBe("Installing")
    expect(recorded.bundles[0]).toMatchObject({ target: ["model-Q4_K_M.gguf"] })
  })

  it("resumes an added model's interrupted download when the service starts", async () => {
    const { run, recorded } = harness({
      customModels: [nativeCustomModel(), nativeCustomModel({
        modelId: addedId,
        origin: { _tag: "Added", repository: "owner/model-GGUF", commit: "c0ffee" },
        state: { _tag: "NotInstalled" },
      })],
    })
    await run(() => Effect.sleep("50 millis"))
    expect(recorded.started).toEqual([{ bundle: { _tag: "Standalone", package: { id: "bundle" } }, forModel: addedId }])
  })

  it("resumes a drafter download a restart interrupted when the service starts", async () => {
    const { run, recorded } = harness({
      customModels: [nativeCustomModel({ drafter: { method: { _tag: "DFlash" }, reference: drafterReference, state: { _tag: "NotInstalled" }, updateAvailable: false } })],
    })
    await run(() => Effect.sleep("50 millis"))
    expect(recorded.started).toEqual([{ bundle: { _tag: "Standalone", package: { id: "bundle" } }, forModel: foundId }])
  })

  it("lists an added model's drafters from its declared repository file while it downloads", async () => {
    const { run, recorded } = harness({
      customModels: [nativeCustomModel({
        modelId: addedId,
        origin: { _tag: "Added", repository: "owner/model-GGUF", commit: "c0ffee" },
        state: { _tag: "NotInstalled" },
      })],
    })
    const result = await run((commands) => commands.listDrafterCandidates({ _tag: "Model", modelId: addedId }, Option.none()).pipe(Effect.map(({ candidates }) => candidates)))
    expect(result.map(({ label }) => label)).toEqual(["Magnitude DFlash"])
    expect(recorded.drafterRequests.at(-1)).toMatchObject({
      _tag: "Candidate",
      candidate: { source: { _tag: "HuggingFace", repository: "owner/model-GGUF", commit: "c0ffee" }, target: ["model-Q4_K_M.gguf"] },
    })
  })

  it("downloads a missing attached drafter on Sync", async () => {
    const { run, recorded } = harness({
      customModels: [nativeCustomModel({ drafter: { method: { _tag: "DFlash" }, reference: drafterReference, state: { _tag: "NotInstalled" }, updateAvailable: false } })],
    })
    const result = await run((commands) => commands.sync(foundId))
    expect(result.outcome).toBe("DownloadingDrafter")
    expect(recorded.bundles[0]).toMatchObject({ drafter: Option.some({ reference: drafterReference }) })
    expect(recorded.started).toEqual([{ bundle: { _tag: "Standalone", package: { id: "bundle" } }, forModel: foundId }])
  })
})

describe("drafter updates", () => {
  const catalogDrafter = { _tag: "Catalog", modelId: "qwen3.6-35b-a3b:gguf:q4" } as const
  const catalog = (updateState: string) => [{
    id: "qwen3.6-35b-a3b:gguf:q4",
    source: {
      displayName: "Qwen3.6 35B-A3B",
      desired: { speculativeMethod: Option.some({ _tag: "DFlash" }) },
      localState: { _tag: "Installed", updateState: { _tag: updateState } },
    },
  }] as unknown as readonly CatalogModelSource[]

  it("updates a catalog drafter with a newer revision, and is current otherwise", async () => {
    const withDrafter = (updateAvailable: boolean) =>
      [nativeCustomModel({ drafter: { method: { _tag: "DFlash" }, reference: catalogDrafter, state: { _tag: "Ready" }, updateAvailable } })]
    // ICN reports the update on the drafter itself, whether or not its catalog model is installed.
    const updating = harness({ customModels: withDrafter(true), catalog: catalog("NotInstalled") })
    expect((await updating.run((commands) => commands.sync(foundId))).outcome).toBe("Updating")
    const current = harness({ customModels: withDrafter(false), catalog: catalog("Current") })
    expect((await current.run((commands) => commands.sync(foundId))).outcome).toBe("AlreadyCurrent")
    expect(current.recorded.started).toEqual([])
    // Its catalog model's update is of other parts; the current drafter is already here.
    const catalogOnly = harness({ customModels: withDrafter(false), catalog: catalog("Available") })
    expect((await catalogOnly.run((commands) => commands.sync(foundId))).outcome).toBe("AlreadyCurrent")
  })
})

describe("added model updates", () => {
  const newer = "b".repeat(40)
  const sameRepositoryDrafter = { _tag: "HuggingFace", repository: "owner/model-GGUF", commit: "c0ffee", path: "mtp.gguf" } as const
  const movedDrafter = { ...sameRepositoryDrafter, commit: newer }
  const newerTarget = ["model-Q4_K_M.gguf", "model-Q4_K_M-part2.gguf"]
  const addedReady = (updateCommit?: string) => nativeCustomModel({
    modelId: addedId,
    origin: { _tag: "Added", repository: "owner/model-GGUF", commit: "c0ffee" },
    files: [{ path: "/store/model-Q4_K_M.gguf", sizeBytes: 1_000 }],
    drafter: { method: { _tag: "Mtp" }, reference: sameRepositoryDrafter, state: { _tag: "Ready" }, updateAvailable: false },
    state: {
      _tag: "Ready",
      catalogAttribution: { _tag: "NotInCatalog" },
      installation: { _tag: "Resolved", installedBytes: 1_000, ownership: "Magnitude", primaryPath: "/store/model-Q4_K_M.gguf" },
      model: readyModel,
    },
    ...(updateCommit === undefined ? {} : { updateCommit }),
  })
  // At the newer commit the same version, named by its primary file, gains a second file and a projector.
  const newerInspection = Schema.decodeUnknownSync(CandidateInspection)({
    source: { _tag: "HuggingFace", repository: "owner/model-GGUF", commit: newer },
    displayName: "",
    projector: { path: "mmproj.gguf", sizeBytes: 10 },
    versions: [{
      label: "Q4_K_M",
      modelId: addedId,
      files: newerTarget.map((path) => ({ path, sizeBytes: 60 })),
      sizeBytes: 120,
      downloadBytes: 120,
      recommended: true,
    }],
  })
  const config = (pendingCommit?: string) => customModelsConfig({
    declarations: {
      [addedId]: {
        origin: { _tag: "Added", repository: "owner/model-GGUF", commit: "c0ffee", ...(pendingCommit === undefined ? {} : { pendingCommit }) },
        target: ["model-Q4_K_M.gguf"],
      },
    },
    drafters: { [addedId]: { _tag: "Reference", reference: sameRepositoryDrafter, method: { _tag: "Mtp" } } },
  })
  const update = (state: ModelDownloadEncoded["state"]) =>
    nativeDownload("update", addedId, { repository: "owner/model-GGUF", revision: newer, path: "model-Q4_K_M.gguf" }, state)
  const running = update({ _tag: "Downloading", stage: "downloading", completedBytes: 30, totalBytes: 120, bytesPerSecond: 10 })
  const movedCandidate = nativeCandidate({
    category: "NextToModel", reference: movedDrafter, method: { _tag: "Mtp" }, verdict: { _tag: "Bound", method: { _tag: "Mtp" }, memoryBytes: 1 },
  })
  const updating = (options: { readonly pendingCommit?: string; readonly downloads?: readonly ModelDownload[]; readonly start?: boolean } = {}) => harness({
    config: config(options.pendingCommit),
    customModels: [addedReady(newer)],
    inspect: newerInspection,
    drafters: [movedCandidate],
    downloads: options.downloads,
    startDownload: options.start === false ? undefined : () => Option.some(running),
  })
  const pinnedCommit = (config: CustomModelsConfig) => {
    const origin = config.declarations[addedId]?.origin
    return origin?._tag === "Added" ? { commit: origin.commit, pendingCommit: origin.pendingCommit } : undefined
  }
  const settled = <A, E>(effect: Effect.Effect<A, E>) => effect.pipe(Effect.zipLeft(Effect.sleep("30 millis")))
  /** The configuration once the update has settled as `expected` says, as the background settlement gets to it. */
  const settledTo = (
    models: Effect.Effect.Success<typeof CustomModels>,
    expected: (pinned: ReturnType<typeof pinnedCommit>) => boolean,
  ) => Effect.gen(function* () {
    for (let attempt = 0; attempt < 200; attempt += 1) {
      const current = yield* models.get
      if (expected(pinnedCommit(current))) return current
      yield* Effect.sleep("10 millis")
    }
    return yield* models.get
  })

  it("shows Update for an added model whose repository has a newer target, and Updating while it downloads", () => {
    const row = (model: ReturnType<typeof nativeCustomModel>, related: readonly RelatedDownload[] = []) => Option.getOrThrow(customModel({
      entry: customSource(model), assessment: undefined, instances: [], related, catalog: [],
      drafterChoice: { _tag: "Auto" },
    }))
    expect(row(addedReady(newer)).acquisitionState._tag).toBe("UpdateAvailable")
    expect(row(addedReady()).acquisitionState._tag).toBe("Installed")
    // The download of the newer commit counts as the target's, so the model reads as updating, still installed.
    const related = customModelDownloads(addedId, config(newer), [running])
    expect(related.map(({ kind }) => kind)).toEqual(["Target"])
    expect(row(addedReady(newer), related).acquisitionState).toMatchObject({
      _tag: "Updating",
      installation: { primaryPath: "/store/model-Q4_K_M.gguf" },
      progress: { completedBytes: 30, totalBytes: 120 },
    })
  })

  it("keeps serving the pinned commit while the update downloads, and pins the newer one only once it finishes", async () => {
    const { run, recorded, downloadList } = updating()
    const result = await run((commands, models) => Effect.gen(function* () {
      const synced = yield* commands.sync(addedId)
      const during = yield* settled(models.get)
      yield* SubscriptionRef.set(downloadList, [update({ _tag: "Completed" })])
      const after = yield* settledTo(models, (pinned) => pinned?.commit === newer)
      return { synced, during, after }
    }))
    expect(result.synced.outcome).toBe("Updating")
    expect(recorded.bundles[0]).toMatchObject({
      source: { _tag: "HuggingFace", repository: "owner/model-GGUF", commit: newer },
      target: newerTarget,
      projector: Option.some({ path: "mmproj.gguf" }),
      drafter: Option.some({ method: { _tag: "Mtp" }, reference: movedDrafter }),
    })
    expect(recorded.started).toEqual([{ bundle: { _tag: "Standalone", package: { id: "bundle" } }, forModel: addedId }])
    // ICN keeps the old commit, so the old copy is still the model and serves.
    expect(pinnedCommit(result.during)).toEqual({ commit: "c0ffee", pendingCommit: Option.some(newer) })
    expect(localModelsConfiguration(result.during, []).declarations).toMatchObject([
      { modelId: addedId, origin: { _tag: "Added", commit: "c0ffee" }, target: ["model-Q4_K_M.gguf"] },
    ])
    expect(result.during.drafters[addedId]).toEqual({ _tag: "Reference", reference: sameRepositoryDrafter, method: { _tag: "Mtp" } })
    expect(result.after.declarations[addedId]).toEqual({
      origin: { _tag: "Added", repository: "owner/model-GGUF", commit: newer, pendingCommit: Option.none() },
      target: newerTarget,
      projector: Option.some("mmproj.gguf"),
    })
    expect(result.after.drafters[addedId]).toEqual({ _tag: "Reference", reference: movedDrafter, method: { _tag: "Mtp" } })
  })

  it("abandons a cancelled or failed update and stays on the pinned commit", async () => {
    for (const ended of [
      update({ _tag: "Cancelled", completedBytes: 30, totalBytes: 120 }),
      update({ _tag: "Failed", acknowledged: false, completedBytes: 30, totalBytes: 120, failure: { _tag: "Interrupted" } }),
    ]) {
      const { run, downloadList } = updating()
      const result = await run((commands, models) => Effect.gen(function* () {
        yield* commands.sync(addedId)
        yield* settled(commands.cancelSync(addedId))
        yield* SubscriptionRef.set(downloadList, [ended])
        return yield* settledTo(models, (pinned) => Option.isNone(pinned?.pendingCommit ?? Option.none()))
      }))
      expect(pinnedCommit(result)).toEqual({ commit: "c0ffee", pendingCommit: Option.none() })
      expect(result.drafters[addedId]).toEqual({ _tag: "Reference", reference: sameRepositoryDrafter, method: { _tag: "Mtp" } })
    }
  })

  it("resumes an update a restart interrupted, and settles one that finished meanwhile", async () => {
    const interrupted = updating({ pendingCommit: newer })
    const resumed = await interrupted.run((_, models) => Effect.gen(function* () {
      for (let attempt = 0; attempt < 200 && interrupted.recorded.started.length === 0; attempt += 1) yield* Effect.sleep("10 millis")
      return yield* settled(models.get)
    }))
    expect(interrupted.recorded.started).toEqual([{ bundle: { _tag: "Standalone", package: { id: "bundle" } }, forModel: addedId }])
    expect(pinnedCommit(resumed)).toEqual({ commit: "c0ffee", pendingCommit: Option.some(newer) })

    const finished = updating({ pendingCommit: newer, downloads: [update({ _tag: "Completed" })] })
    const pinned = await finished.run((_, models) => settledTo(models, (current) => current?.commit === newer))
    expect(finished.recorded.started).toEqual([])
    expect(pinnedCommit(pinned)).toEqual({ commit: newer, pendingCommit: Option.none() })
  })

  it("is current without a newer target and changes nothing", async () => {
    const { run, recorded } = harness({ config: config(), customModels: [addedReady()], inspect: newerInspection })
    const result = await run((commands, models) => Effect.gen(function* () {
      const synced = yield* commands.sync(addedId)
      return { synced, config: yield* models.get }
    }))
    expect(result.synced.outcome).toBe("AlreadyCurrent")
    expect(pinnedCommit(result.config)).toEqual({ commit: "c0ffee", pendingCommit: Option.none() })
    expect(recorded.started).toEqual([])
  })
})

describe("catalog models found elsewhere", () => {
  const catalogRow = (modelId: string, ownership: string, primaryPath: string) => ({
    _tag: "Catalog",
    modelId,
    acquisitionState: {
      _tag: "Installed",
      installation: { _tag: "Resolved", installedBytes: 1, primaryPath, ownership },
      residencyState: { _tag: "Unloaded" },
    },
  }) as unknown as LocalModelsState["models"][number]
  const external = CatalogFormModelIdSchema.make("qwen3.6-35b-a3b:gguf:q4")
  const owned = CatalogFormModelIdSchema.make("qwen3.6-27b:gguf:q4")
  const cachePath = "/nonexistent-home/.cache/huggingface/hub/models--o--r/snapshots/c/m.gguf"

  it("hides a catalog model whose files are only in an external location by its files' found-model IDs, and refuses one Magnitude owns", async () => {
    const alias = CustomFormModelIdSchema.make("hf:unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf")
    const { run } = harness({
      catalogRows: [catalogRow(external, "External", cachePath), catalogRow(owned, "Magnitude", "/store/m.gguf")],
      catalog: [{ id: external, source: { aliases: Option.some([alias]) } }] as unknown as readonly CatalogModelSource[],
    })
    const result = await run((commands, models) => Effect.gen(function* () {
      const files = yield* commands.externalCatalogFiles(external)
      yield* commands.hide(external)
      const refused = yield* commands.hide(owned).pipe(Effect.flip)
      const hidden = (yield* models.get).hidden
      yield* commands.unhide(external)
      const afterUnhide = (yield* models.get).hidden
      yield* commands.hide(external)
      yield* commands.unhide(alias)
      return { files, refused, hidden, afterUnhide, afterAliasUnhide: (yield* models.get).hidden }
    }))
    expect(result.files).toEqual(Option.some({ path: cachePath, location: Option.some("HuggingFaceCache") }))
    expect(result.hidden).toEqual([alias])
    expect(result.afterAliasUnhide).toEqual([])
    expect(result.refused.message).toBe(`${owned} is a catalog model. Remove it with \`magnitude catalog remove ${owned}\` instead.`)
    expect(result.afterUnhide).toEqual([])
  })

  it("hides a found catalog copy that serves with the catalog drafter from Magnitude's store, and Remove points to Hide", async () => {
    // ICN's catalog state for weights found only in a Hugging Face cache, beside the catalog drafter Magnitude
    // downloaded: ownership follows the weights, so the projected row is External.
    const native = {
      localState: {
        _tag: "Installed",
        effective: { _tag: "Ready", model: readyModel },
        installation: { _tag: "Resolved", installedBytes: 2, primaryPath: cachePath, ownership: "External" },
        updateState: { _tag: "Current" },
        missingComponents: [{ component: "Projector", downloadBytes: 1 }],
      },
      desired: { metadata: { storageBytes: 2 } },
    } as unknown as Parameters<typeof catalogAcquisition>[0]
    const row = {
      _tag: "Catalog",
      modelId: external,
      acquisitionState: catalogAcquisition(native, undefined, { _tag: "Unloaded" }),
    } as unknown as LocalModelsState["models"][number]
    const alias = CustomFormModelIdSchema.make("hf:unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf")
    const { run } = harness({
      catalogRows: [row],
      catalog: [{ id: external, source: { aliases: Option.some([alias]) } }] as unknown as readonly CatalogModelSource[],
    })
    const result = await run((commands, models) => Effect.gen(function* () {
      const files = yield* commands.externalCatalogFiles(external)
      yield* commands.hide(external)
      return { files, hidden: (yield* models.get).hidden }
    }))
    expect(result.hidden).toEqual([alias])
    expect(Option.isSome(result.files)).toBe(true)
  })

  it("hides a catalog model whose external copy is already hidden again without change, and unhides it by its catalog ID", async () => {
    const alias = CustomFormModelIdSchema.make("hf:unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf")
    const { run } = harness({
      config: customModelsConfig({ hidden: [alias] }),
      catalogRows: [{ _tag: "Catalog", modelId: external, acquisitionState: { _tag: "NotInstalled" } } as unknown as LocalModelsState["models"][number]],
      catalog: [{ id: external, source: { aliases: Option.some([alias]) } }] as unknown as readonly CatalogModelSource[],
    })
    const result = await run((commands, models) => Effect.gen(function* () {
      yield* commands.hide(external)
      const hidden = (yield* models.get).hidden
      yield* commands.unhide(external)
      return { hidden, afterUnhide: (yield* models.get).hidden }
    }))
    expect(result.hidden).toEqual([alias])
    expect(result.afterUnhide).toEqual([])
  })

  it("attributes files in any of a location's directories to that location", async () => {
    const systemOllama = "/var/lib/ollama/blobs/sha256-0000"
    const { run } = harness({ catalogRows: [catalogRow(external, "External", systemOllama)] })
    const files = await run((commands) => commands.externalCatalogFiles(external))
    expect(files).toEqual(Option.some({ path: systemOllama, location: Option.some("Ollama") }))
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
    expect(result.locations.find(({ kind }) => kind === "Ollama")).toMatchObject({ enabled: false, automatic: true, state: "NotFound" })
    expect(result.locations.find(({ kind }) => kind === "Folder"))
      .toEqual({ kind: "Folder", paths: [{ path: folder, state: "Found" }], enabled: true, automatic: false, state: "Found" })
    expect(result.locations.find(({ kind }) => kind === "Ollama")?.paths.map(({ path }) => path)).toEqual([
      "/nonexistent/ollama", "/nonexistent-home/.ollama/models", "/usr/share/ollama/.ollama/models", "/var/lib/ollama",
    ])
    expect(result.afterRemove.folders).toEqual([])
  })
})

