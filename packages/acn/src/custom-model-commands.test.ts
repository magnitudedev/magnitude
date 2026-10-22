import { BunFileSystem } from "@effect/platform-bun"
import { Effect, Either, Layer, Option, Schema, Stream, SubscriptionRef } from "effect"
import { describe, expect, it } from "vitest"
import type { CustomModelsConfig } from "@magnitudedev/storage"
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
  CandidateInspectionProgress,
  DrafterCandidate as NativeDrafterCandidate,
  ModelDownload as NativeModelDownload,
  type ModelDownload,
  type ModelDownloadEncoded,
} from "@magnitudedev/icn-protocol/schemas"
import {
  CatalogFormModelIdSchema,
  CustomFormModelIdSchema,
  DrafterReferenceSchema,
  HuggingFaceRepositoryIdSchema,
  LocalModelMutationFailed,
  type LocalModelsState,
} from "@magnitudedev/acn-protocol"
import { CustomModelCommands, CustomModelCommandsLive } from "./custom-model-commands"
import { CatalogModelSync } from "./catalog-model-sync"
import { CustomModelReconciler } from "./custom-model-reconciler"
import { CustomModels, configuredDrafterChoice, localModelsConfiguration } from "./custom-models"
import { HuggingFaceAccount } from "./hugging-face-account"
import { LocalModelSources, type CatalogModelSource } from "./local-model-sources"
import { LocalModels, catalogAcquisition, customModel } from "./local-models"
import { customModelDownloads, type RelatedDownload } from "./custom-model-downloads"
import { makeModelLocations } from "./model-locations"
import {
  addedId,
  customModelsConfig,
  customSource,
  drafterReference,
  drafterSetting,
  foundId,
  localId,
  makeTestCustomModels,
  modelPackage,
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

const progressOf = (phase: unknown, assessments: readonly unknown[]) => Schema.decodeUnknownSync(CandidateInspectionProgress)({
  phase,
  versions: [
    { label: "Q4_K_M", sizeBytes: 100, ...(assessments[0] === undefined ? {} : { assessment: assessments[0] }) },
    { label: "Q2_K", sizeBytes: 50, ...(assessments[1] === undefined ? {} : { assessment: assessments[1] }) },
  ],
})

const unsupportedQ2K = { _tag: "Unsupported", profile: { contextLength: 4096 }, reasons: [{ _tag: "WeightTypes", types: [{ ggmlType: "Q2_K", tensorCount: 2 }] }], detail: "" }

interface DrafterList {
  readonly candidates: readonly NativeDrafterCandidate[]
  readonly entered: Option.Option<NativeDrafterCandidate["reference"]>
}

/** ICN's drafter list; `entered` is the listed candidate a different drafter resolved to. */
const listed = (candidates: readonly NativeDrafterCandidate[], entered?: NativeDrafterCandidate): DrafterList =>
  ({ candidates, entered: Option.map(Option.fromNullable(entered), ({ reference }) => reference) })

/** A download ICN was asked to start: the model, and which part of it (`none` for the model itself). */
interface StartedDownload {
  readonly modelId: string
  readonly part: Option.Option<"Drafter" | "Update">
}

const startedTarget = (modelId: string): StartedDownload => ({ modelId, part: Option.none() })
const startedDrafter = (modelId: string): StartedDownload => ({ modelId, part: Option.some("Drafter") })
const startedUpdate = (modelId: string): StartedDownload => ({ modelId, part: Option.some("Update") })

interface Recorded {
  readonly started: StartedDownload[]
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
  readonly progress?: readonly ReturnType<typeof progressOf>[]
  readonly progressFailure?: unknown
  readonly drafters?: readonly NativeDrafterCandidate[]
  readonly draftersFor?: (request: { readonly other: Option.Option<string> }) => Effect.Effect<DrafterList, unknown>
  readonly downloads?: readonly ModelDownload[]
  readonly customModels?: ReturnType<typeof nativeCustomModel>[]
  readonly catalog?: readonly CatalogModelSource[]
  readonly catalogRows?: readonly LocalModelsState["models"][number][]
  readonly catalogInstalled?: readonly string[]
  readonly signedIn?: boolean
  readonly startDownload?: (started: StartedDownload) => Option.Option<ModelDownload>
  readonly pushFailure?: LocalModelMutationFailed
  /** The local models as ICN lists them now, when they differ from the custom models' own rows. */
  readonly listedModels?: readonly LocalModelsState["models"][number][]
  /** Whether the local models change once after the commands start. */
  readonly localModelsChange?: boolean
} = {}) => {
  const recorded: Recorded = {
    started: [], cancelled: [], acknowledged: [], removedFiles: [], drafterRequests: [], catalogSyncs: [], pushes: 0, prunes: 0,
  }
  const downloadList = Effect.runSync(SubscriptionRef.make<readonly ModelDownload[]>(options.downloads ?? []))
  const sourceModels = (options.customModels ?? [nativeCustomModel()]).map(customSource)
  const candidates: IcnCandidatesService = {
    search: () => Effect.succeed({
      results: [{
        repository: "HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive",
        architecture: Option.some("qwen35moe"),
        commit: "c",
        downloads: Option.some(12),
        gated: false,
        versionCount: Option.none(),
        private: false,
        lastModified: Option.none(),
        likes: Option.none(),
        tags: [],
        unsupported: [],
        variantOf: Option.some("qwen3.6-35b-a3b:gguf:q4"),
      }, {
        repository: "unsloth/DeepSeek-V4-Flash-GGUF",
        architecture: Option.some("deepseek4"),
        commit: "d",
        downloads: Option.some(3),
        gated: true,
        versionCount: Option.none(),
        private: false,
        lastModified: Option.none(),
        likes: Option.none(),
        tags: [],
        unsupported: [{ _tag: "Architecture", architecture: "deepseek4" }],
        variantOf: Option.none(),
      }],
    }) as never,
    resolveRepository: () => Effect.die("unused"),
    inspect: () => options.inspectFailure === undefined
      ? Effect.succeed(options.inspect ?? inspection())
      : Effect.fail(options.inspectFailure as never),
    inspectWithProgress: () => Stream.concat(
      Stream.fromIterable((options.progress ?? []).map((progress) => ({ _tag: "Progress" as const, progress }))),
      options.progressFailure === undefined
        ? Stream.make({ _tag: "Inspected" as const, inspection: options.inspect ?? inspection() })
        : Stream.fail(options.progressFailure as never),
    ) as never,
    drafters: (request) => Effect.suspend(() => {
      recorded.drafterRequests.push(request)
      return options.draftersFor?.(request as { readonly other: Option.Option<string> })
        ?? Effect.succeed(listed(options.drafters ?? [nativeCandidate()]))
    }) as never,
    assess: () => Effect.die("unused"),
  }
  const layer = Effect.gen(function* () {
    const config = options.config ?? customModelsConfig()
    const custom = yield* makeTestCustomModels(config)
    const rows = sourceModels.flatMap((entry) => Option.toArray(customModel({
      entry, assessment: undefined, instances: [], related: [], catalog: options.catalog ?? [],
      drafterChoice: configuredDrafterChoice(config, entry.id),
    })))
    const localState: LocalModelsState = {
      preparation: { discovery: { complete: true, modelsFound: rows.length }, assessment: { complete: true, settledModels: 0, totalModels: 0 } },
      models: [...(options.catalogRows ?? []), ...(options.listedModels ?? rows)],
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
        start: (modelId: string, part: StartedDownload["part"]) => Effect.gen(function* () {
          recorded.started.push({ modelId, part })
          const started = options.startDownload?.({ modelId, part }) ?? Option.none()
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
      Layer.succeed(LocalModels, {
        state: Effect.succeed(localState),
        changes: options.localModelsChange ? Stream.make(undefined) : Stream.never,
      } as never),
      Layer.succeed(CustomModels, custom),
      Layer.succeed(CustomModelReconciler, {
        push: Effect.sync(() => { recorded.pushes += 1 }).pipe(
          Effect.zipRight(options.pushFailure === undefined ? Effect.void : Effect.fail(options.pushFailure)),
        ),
      }),
      Layer.succeed(CatalogModelSync, {
        installed: (modelId) => Effect.succeed((options.catalogInstalled ?? []).includes(modelId)),
        sync: (modelId) => Effect.sync(() => {
          recorded.catalogSyncs.push(modelId)
          return { outcome: "Installing" as const }
        }),
        downloadComponent: () => Effect.succeed({ started: false }),
      }),
      Layer.succeed(HuggingFaceAccount, {
        account: Effect.succeed(options.signedIn
          ? { _tag: "SignedIn" as const, username: "ada", avatarUrl: Option.none() }
          : { _tag: "SignedOut" as const, reason: Option.none() }),
      } as never),
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
  it("maps search results with uploader, variant, and an unsupported architecture's reason", async () => {
    const { run } = harness()
    const result = await run((commands) => commands.searchHuggingFace({ query: "qwen", limit: Option.none() }))
    expect(result.results[1]).toMatchObject({
      displayName: "DeepSeek V4 Flash",
      variantOf: Option.none(),
      unsupported: Option.some([{ _tag: "Architecture", architecture: "deepseek4" }]),
      gated: true,
    })
    expect(result.results[0]).toMatchObject({
      displayName: "Qwen3.6 35B A3B Uncensored HauhauCS Aggressive",
      uploader: "HauhauCS",
      variantOf: Option.some("qwen3.6-35b-a3b:gguf:q4"),
      downloads: 12,
      architecture: Option.some("qwen35moe"),
      unsupported: Option.none(),
      gated: false,
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

  it("relays the inspection's progress in order, each verdict projected, then the inspection", async () => {
    const { run } = harness({
      progress: [
        progressOf({ _tag: "ReadingHeaders", completed: 1, total: 2 }, []),
        progressOf({ _tag: "CheckingVersions", completed: 1, total: 2 }, [undefined, unsupportedQ2K]),
        progressOf({ _tag: "CheckingDrafters", version: "Q4_K_M", completed: 0, total: 1 }, [undefined, unsupportedQ2K]),
      ],
    })
    const updates = await run((commands) => Stream.runCollect(commands.inspectCandidateWithProgress("owner/model-GGUF:q2_k")))
    const [reading, checking, drafters, done] = [...updates]
    expect([...updates].map(({ _tag }) => _tag)).toEqual(["Inspecting", "Inspecting", "Inspecting", "Inspected"])
    expect(reading?._tag === "Inspecting" && reading.progress.phase).toEqual({ _tag: "ReadingHeaders", completed: 1, total: 2 })
    expect(checking?._tag === "Inspecting" && checking.progress.versions.map(({ label, assessment }) =>
      [label, Option.map(assessment, ({ _tag }) => _tag)])).toEqual([["Q4_K_M", Option.none()], ["Q2_K", Option.some("Unsupported")]])
    expect(drafters?._tag === "Inspecting" && drafters.progress.phase).toEqual({ _tag: "CheckingDrafters", version: "Q4_K_M", completed: 0, total: 1 })
    expect(done?._tag === "Inspected" && done.inspection.selectedVersion).toEqual(Option.some("Q2_K"))
  })

  it("reports a failure after the inspection's progress began as the request/response inspection would", async () => {
    const { run } = harness({
      progress: [progressOf({ _tag: "ReadingHeaders", completed: 0, total: 2 }, [])],
      progressFailure: {
        _tag: "GeneratedClientRemoteError",
        status: 200,
        body: { error: { type: "server_error", code: "inventory_error", message: "upstream model service failed: timed out" } },
      },
    })
    const result = await run((commands) => Stream.runCollect(commands.inspectCandidateWithProgress("owner/model-GGUF")).pipe(Effect.either))
    expect(result._tag === "Left" && result.left).toMatchObject({
      _tag: "ModelSourceUnavailable",
      reason: "SourceUnreachable",
      message: "upstream model service failed: timed out",
      retryable: true,
    })
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

  it("asks to sign in to Hugging Face, or to accept the terms, for a gated repository", async () => {
    const refused = (code: string, signedIn: boolean) => harness({
      signedIn,
      inspectFailure: {
        _tag: "GeneratedClientRemoteError",
        status: 409,
        body: { error: { code, message: "GGUF header request returned HTTP 401 Unauthorized" } },
      },
    })
    const add = (commands: Effect.Effect.Success<typeof CustomModelCommands>) => commands.addHuggingFaceModel({
      repository, revision: Option.none(), version: Option.none(), drafter: { _tag: "Auto" }, force: false
    }).pipe(Effect.either)
    const signIn = await refused("hugging_face_sign_in_required", false).run(add)
    expect(signIn._tag === "Left" && signIn.left).toMatchObject({
      _tag: "HuggingFaceAccessRequired",
      repository,
      access: "SignIn",
      message: "Sign in to Hugging Face to use this model.",
    })
    const terms = await refused("hugging_face_terms_required", true).run((commands) =>
      commands.inspectCandidate(`https://huggingface.co/${repository}`).pipe(Effect.either))
    expect(terms._tag === "Left" && terms.left).toMatchObject({
      _tag: "HuggingFaceAccessRequired",
      repository,
      access: "AcceptTerms",
      message: "Accept this model's terms on Hugging Face, then try again. Some models also need the author's approval.",
    })
    expect(JSON.stringify(terms)).not.toMatch(/HF_TOKEN/)
  })

  it("says a repository Hugging Face won't show may be missing, private or gated, for inspect and add", async () => {
    const missing = (signedIn: boolean) => harness({
      signedIn,
      inspectFailure: {
        _tag: "GeneratedClientRemoteError",
        status: 409,
        body: { error: { code: "hugging_face_repository_not_found", message: "Repository not found" } },
      },
    })
    const message = "Repository not found, or it's private or gated. Check the name, or sign in to Hugging Face if you have access."
    const inspected = await missing(false).run((commands) => commands.inspectCandidate("nobody-xyz/does-not-exist-GGUF").pipe(Effect.either))
    expect(inspected._tag === "Left" && inspected.left).toMatchObject({
      _tag: "ModelRepositoryNotFound",
      reference: "nobody-xyz/does-not-exist-GGUF",
      message,
    })
    const added = await missing(false).run((commands) => commands.addHuggingFaceModel({
      repository, revision: Option.none(), version: Option.none(), drafter: { _tag: "Auto" }, force: false
    }).pipe(Effect.either))
    expect(added._tag === "Left" && added.left).toMatchObject({ _tag: "ModelRepositoryNotFound", message })
    const signedIn = await missing(true).run((commands) => commands.inspectCandidate("nobody-xyz/does-not-exist-GGUF").pipe(Effect.either))
    expect(signedIn._tag === "Left" && signedIn.left).toMatchObject({
      message: "Repository not found, or your Hugging Face account doesn't have access to it. Check the name.",
    })
  })

  it("says a repository without GGUF files has none Magnitude can use", async () => {
    const { run } = harness({
      inspectFailure: {
        _tag: "GeneratedClientRemoteError",
        status: 409,
        body: {
          error: {
            code: "hugging_face_repository_without_gguf",
            message: "This repository has no GGUF files Magnitude can use.",
          },
        },
      },
    })
    const result = await run((commands) => commands.inspectCandidate("meta-llama/Llama-3.2-1B-Instruct").pipe(Effect.either))
    expect(result._tag === "Left" && result.left).toMatchObject({
      _tag: "ModelSourceUnavailable",
      reason: "Unavailable",
      message: "This repository has no GGUF files Magnitude can use.",
      retryable: false,
    })
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
  const pinnedByHand = customModelsConfig({ drafters: { [foundId]: hand.path } })
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
      config: customModelsConfig({ drafters: { [foundId]: "hf:magnitudedev/Model-DFlash-GGUF/Model-DFlash-Q8_0.gguf@d1" } }),
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
    // Automatic choice is stored by leaving the model out.
    expect(chosen.saved).toBeUndefined()

    const notRecommended = harness({ config: customModelsConfig({ drafters: {} }), draftersFor: resolvesToCatalog })
    const saved = await notRecommended.run((commands, models) => commands.setModelDrafter(foundId, { _tag: "Reference", reference: catalogFile }).pipe(
      Effect.zipRight(models.get),
    ))
    expect(saved.drafters[foundId]).toEqual({
      _tag: "HuggingFace", repository: "magnitudedev/model-DFlash-GGUF", path: "dflash.gguf", revision: "d1",
    })
  })
})

describe("adding models", () => {
  const notYetAdded = () => customModelsConfig({ added: {} })

  it("stores an added model at its commit and asks the inference service to download it", async () => {
    const { run, recorded } = harness({ config: notYetAdded() })
    const result = await run((commands, models) => Effect.gen(function* () {
      const added = yield* commands.addHuggingFaceModel({ repository, revision: Option.none(), version: Option.none(), drafter: { _tag: "Auto" }, force: false })
      return { added, config: yield* models.get }
    }))
    expect(result.added).toMatchObject({ modelId: addedId, version: "Q4_K_M", outcome: "Started" })
    expect(result.config.added[addedId]).toEqual({ revision: "c0ffee", projector: Option.none() })
    // Automatic choice is stored by leaving the model out of `drafters`.
    expect(result.config.drafters[addedId]).toBeUndefined()
    expect(recorded.started).toEqual([startedTarget(addedId)])
    expect(recorded.pushes).toBeGreaterThan(0)
  })

  it("stores an added model's projector from the inspection", async () => {
    const withProjector = Schema.decodeUnknownSync(CandidateInspection)({
      ...Schema.encodeSync(CandidateInspection)(inspection()),
      projector: { path: "mmproj-F16.gguf", sizeBytes: 10 },
    })
    const { run } = harness({ config: notYetAdded(), inspect: withProjector })
    const config = await run((commands, models) => commands.addHuggingFaceModel({
      repository, revision: Option.none(), version: Option.none(), drafter: { _tag: "Auto" }, force: false,
    }).pipe(Effect.zipRight(models.get)))
    expect(config.added[addedId]).toEqual({ revision: "c0ffee", projector: Option.some("mmproj-F16.gguf") })
  })

  it("adds a version of a catalog model's repository that isn't its exact target as a custom model whose catalog drafter is Auto", async () => {
    const catalogDrafter = { _tag: "Catalog", modelId: CatalogFormModelIdSchema.make("qwen3.5-4b:gguf:q4") } as const
    const { run, recorded } = harness({ config: notYetAdded(), drafters: [nativeCandidate({ reference: catalogDrafter, lineage: "Exact" })] })
    const result = await run((commands, models) => Effect.gen(function* () {
      const added = yield* commands.addHuggingFaceModel({ repository, revision: Option.none(), version: Option.some("Q4_K_M"), drafter: { _tag: "Auto" }, force: false })
      return { added, config: yield* models.get }
    }))
    expect(result.added).toMatchObject({ _tag: "Custom", modelId: addedId, outcome: "Started" })
    expect(result.added._tag === "Custom" && Option.map(result.added.drafter, ({ reference }) => reference)).toEqual(Option.some(catalogDrafter))
    expect(result.config.added[addedId]?.revision).toBe("c0ffee")
    expect(result.config.drafters[addedId]).toBeUndefined()
    expect(recorded.catalogSyncs).toEqual([])
    expect(recorded.started).toEqual([startedTarget(addedId)])
  })

  it("pins a drafter the user names when adding, as config.json spells it", async () => {
    const other = { _tag: "Local", path: "/drafters/dspark.gguf" } as const
    const { run } = harness({ config: notYetAdded(), drafters: [nativeCandidate({
      category: "Other", reference: other, method: { _tag: "DSpark" }, verdict: { _tag: "Bound", method: { _tag: "DSpark" }, memoryBytes: 1 },
    })] })
    const config = await run((commands, models) => commands.addHuggingFaceModel({
      repository, revision: Option.none(), version: Option.none(), drafter: { _tag: "Reference", reference: other }, force: false,
    }).pipe(Effect.zipRight(models.get)))
    expect(config.drafters[addedId]).toEqual({ _tag: "Local", path: "/drafters/dspark.gguf" })
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

  it("asks only for the model when the chosen drafter is already on this computer, leaving its files to the inference service", async () => {
    const { run, recorded } = harness({ config: notYetAdded(), drafters: [nativeCandidate({ onThisComputer: true })] })
    const result = await run((commands, models) => Effect.gen(function* () {
      const added = yield* commands.addHuggingFaceModel({ repository, revision: Option.none(), version: Option.none(), drafter: { _tag: "Auto" }, force: false })
      return { added, config: yield* models.get }
    }))
    expect(result.added._tag === "Custom" && result.added.drafter.pipe(Option.map(({ onThisComputer }) => onThisComputer))).toEqual(Option.some(true))
    expect(result.added._tag === "Custom" && result.added.drafterDownloadStarted).toBe(false)
    expect(result.config.drafters[addedId]).toBeUndefined()
    expect(recorded.started).toEqual([startedTarget(addedId)])
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
          projector: { _tag: "None" },
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
    expect(result.config.added[foundId]).toBeUndefined()
    expect(recorded.started).toEqual([startedDrafter(foundId)])
  })

  // QA linux-cuda round 3c: pulling a version that is already a found model ("already on this computer") declares
  // nothing, so the model stays found and Remove hides it, as it hides any found model.
  it("hides a found model on Remove after adding it reported it as already on this computer", async () => {
    const { run, recorded } = harness({ inspect: inspection(foundId) })
    const config = await run((commands, models) => Effect.gen(function* () {
      yield* commands.addHuggingFaceModel({ repository, revision: Option.none(), version: Option.none(), drafter: { _tag: "Auto" }, force: false })
      yield* commands.remove(foundId)
      return yield* models.get
    }))
    expect(config.added[foundId]).toBeUndefined()
    expect(config.hidden).toEqual([foundId])
    expect(recorded.removedFiles).toEqual([])
  })

})

describe("hiding a model that isn't on this computer", () => {
  it("fails for an unknown custom or catalog ID and changes nothing", async () => {
    const { run } = harness({ config: customModelsConfig({ added: {} }) })
    const unknownCustom = CustomFormModelIdSchema.make("local:unknown-model-q4_k_m:cccccccccccc")
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

  it("hides a catalog model named by its alias by its catalog ID", async () => {
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
    expect(hidden).toEqual([external])
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
    expect(recorded.started).toEqual([])
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
    const missing = harness({ startDownload: () => Option.some(drafterDownload("drafter", { _tag: "Pending", completedBytes: 0, totalBytes: 9 })) })
    expect((await missing.run((commands) => commands.setModelDrafter(foundId, { _tag: "Auto" }))).drafterDownloadStarted).toBe(true)
    expect(missing.recorded.started).toEqual([startedDrafter(foundId)])
    // Everything the drafter needs is already here: the inference service admits no download.
    const nothingToFetch = harness()
    expect((await nothingToFetch.run((commands) => commands.setModelDrafter(foundId, { _tag: "Auto" }))).drafterDownloadStarted).toBe(false)
  })

  it("keeps Auto, and picking the recommended drafter, saved as Auto; another drafter is pinned without its method", async () => {
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
    expect(saved.recommended).toBeUndefined()
    expect(saved.pinned).toEqual({ _tag: "Local", path: other.path })
    expect(saved.automatic).toBeUndefined()
  })

  it("refuses a model that is not on this computer", async () => {
    const { run } = harness()
    const result = await run((commands) => commands.setModelDrafter(CustomFormModelIdSchema.make("hf:a/b/c.gguf"), { _tag: "None" }).pipe(Effect.either))
    expect(result._tag === "Left" && result.left._tag).toBe("LocalModelMutationFailed")
  })
})

describe("removal by origin", () => {
  it("deletes an added model's copy and hides a found model", async () => {
    const { run, recorded } = harness({
      config: customModelsConfig({ drafters: { [addedId]: "none", [foundId]: drafterSetting } }),
      downloads: [targetDownload("active", { _tag: "Downloading", completedBytes: 1, totalBytes: 2, stage: "downloading" })],
    })
    const config = await run((commands, models) => Effect.gen(function* () {
      yield* commands.remove(addedId)
      yield* commands.remove(foundId)
      return yield* models.get
    }))
    expect(recorded.cancelled).toEqual(["active"])
    expect(recorded.removedFiles).toEqual([addedId])
    expect(recorded.prunes).toBe(1)
    expect(config.added).toEqual({})
    expect(Object.keys(config.drafters)).toEqual([foundId])
    expect(config.hidden).toEqual([foundId])
  })

  it("hides a model found in a folder by its local ID", async () => {
    const { run, recorded } = harness({
      customModels: [nativeCustomModel({ modelId: localId, origin: { _tag: "Found", source: { _tag: "Folder", path: "/models" } } })],
    })
    const config = await run((commands, models) => commands.remove(localId).pipe(Effect.zipRight(models.get)))
    expect(config.hidden).toEqual([localId])
    expect(recorded.removedFiles).toEqual([])
  })

  it("ends a removed model's downloads, so none resumes after a restart", async () => {
    const { run, recorded } = harness({
      downloads: [
        drafterDownload("running", { _tag: "Pending", completedBytes: 0, totalBytes: 9 }),
        drafterDownload("failed", { _tag: "Failed", acknowledged: false, completedBytes: 0, totalBytes: 9, failure: { _tag: "Interrupted" } }),
      ],
    })
    const config = await run((commands, models) => commands.remove(foundId).pipe(Effect.zipRight(models.get)))
    expect(recorded.cancelled).toEqual(["running"])
    expect(recorded.acknowledged).toEqual(["failed"])
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
    expect(config.added[addedId]).toBeUndefined()
    expect(config.drafters[addedId]).toBeUndefined()
    expect(config.hidden).toEqual([])
  })

  it("refuses to hide a model added from Hugging Face, and hides a found one", async () => {
    const { run } = harness()
    const result = await run((commands, models) => Effect.gen(function* () {
      const added = yield* commands.hide(addedId).pipe(Effect.flip)
      yield* commands.hide(foundId)
      yield* commands.hide(foundId)
      return { added, hidden: (yield* models.get).hidden }
    }))
    expect(result.added).toMatchObject({
      code: "managed_model_not_hideable",
      message: `${addedId} was added to Magnitude, so it can't be hidden. Remove it with \`magnitude catalog remove ${addedId}\` instead.`,
    })
    expect(result.hidden).toEqual([foundId])
  })

  it("unhides a hidden model", async () => {
    const { run } = harness({ config: customModelsConfig({ hidden: [foundId] }) })
    const config = await run((commands, models) => commands.unhide(foundId).pipe(Effect.zipRight(models.get)))
    expect(config.hidden).toEqual([])
  })

  it("matches a local model hidden under an older label by its hash, for hide and unhide", async () => {
    const olderLabel = CustomFormModelIdSchema.make("local:old-name-q4_k_m:0123456789ab")
    const { run } = harness({
      config: customModelsConfig({ hidden: [olderLabel] }),
      customModels: [nativeCustomModel({ modelId: localId, origin: { _tag: "Found", source: { _tag: "Folder", path: "/models" } } })],
    })
    const result = await run((commands, models) => Effect.gen(function* () {
      yield* commands.hide(localId)
      const hidden = (yield* models.get).hidden
      yield* commands.unhide(localId)
      return { hidden, afterUnhide: (yield* models.get).hidden }
    }))
    expect(result.hidden).toEqual([olderLabel])
    expect(result.afterUnhide).toEqual([])
  })
})

describe("stored IDs following the models they name", () => {
  const olderLabel = CustomFormModelIdSchema.make("local:old-name-q4_k_m:0123456789ab")
  const folderModel = () => nativeCustomModel({ modelId: localId, origin: { _tag: "Found", source: { _tag: "Folder", path: "/models" } } })
  const renamedTo = (models: Effect.Effect.Success<typeof CustomModels>, done: (config: CustomModelsConfig) => boolean) => Effect.gen(function* () {
    for (let attempt = 0; attempt < 200; attempt += 1) {
      const current = yield* models.get
      if (done(current)) return current
      yield* Effect.sleep("10 millis")
    }
    return yield* models.get
  })

  it("renames a local model stored under an older label in hidden and drafters to its current ID", async () => {
    const { run } = harness({
      config: customModelsConfig({ hidden: [olderLabel], drafters: { [olderLabel]: "none", [foundId]: drafterSetting } }),
      customModels: [folderModel(), nativeCustomModel()],
      localModelsChange: true,
    })
    const config = await run((_, models) => renamedTo(models, (current) => current.hidden.includes(localId)))
    expect(config.hidden).toEqual([localId])
    expect(config.drafters).toEqual({
      [localId]: { _tag: "None" },
      [foundId]: { _tag: "HuggingFace", repository: "magnitudedev/model-DFlash-GGUF", path: "dflash.gguf", revision: "d1" },
    })
  })

  it("renames a catalog model's alias in hidden to its catalog ID, and drops a drafter choice for it", async () => {
    const external = CatalogFormModelIdSchema.make("qwen3.6-35b-a3b:gguf:q4")
    const alias = CustomFormModelIdSchema.make("local:qwen3.6-35b-a3b-q4_k_m:abcdefabcdef")
    const { run } = harness({
      config: customModelsConfig({ hidden: [alias], drafters: { [alias]: "none" } }),
      catalogRows: [{
        _tag: "Catalog", modelId: external, catalogData: { aliases: [alias] }, acquisitionState: { _tag: "NotInstalled" },
      } as unknown as LocalModelsState["models"][number]],
      localModelsChange: true,
    })
    const config = await run((_, models) => renamedTo(models, (current) => current.hidden.includes(external)))
    expect(config.hidden).toEqual([external])
    expect(config.drafters).toEqual({})
  })

  it("keeps IDs that name no listed model as they are", async () => {
    const unlisted = CustomFormModelIdSchema.make("local:gone-q8_0:ffffffffffff")
    const { run } = harness({
      config: customModelsConfig({ hidden: [unlisted], drafters: { [unlisted]: "none" } }),
      customModels: [folderModel()],
      localModelsChange: true,
    })
    const config = await run((_, models) => Effect.sleep("50 millis").pipe(Effect.zipRight(models.get)))
    expect(config.hidden).toEqual([unlisted])
    expect(Object.keys(config.drafters)).toEqual([unlisted])
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
      startDownload: () => Option.some(targetDownload("target", { _tag: "Pending", completedBytes: 0, totalBytes: 10 })),
    })
    const result = await run((commands) => commands.sync(addedId))
    expect(result.outcome).toBe("Installing")
    expect(recorded.started).toEqual([startedTarget(addedId)])
  })

  // QA mac-r4-14: a cancelled queued pull came back after a restart, because ACN took every added model still
  // missing its files for an interrupted download. ICN resumes exactly the downloads still requested.
  it("starts no download when the service starts, so a cancelled one stays cancelled", async () => {
    const { run, recorded } = harness({
      customModels: [
        nativeCustomModel({ drafter: { method: { _tag: "DFlash" }, reference: drafterReference, state: { _tag: "NotInstalled" }, updateAvailable: false } }),
        nativeCustomModel({
          modelId: addedId,
          origin: { _tag: "Added", repository: "owner/model-GGUF", commit: "c0ffee" },
          state: { _tag: "NotInstalled" },
        }),
      ],
    })
    await run(() => Effect.sleep("50 millis"))
    expect(recorded.started).toEqual([])
  })

  it("lists an added model's drafters from its repository files while it downloads", async () => {
    const { run, recorded } = harness({
      customModels: [nativeCustomModel({
        modelId: addedId,
        origin: { _tag: "Added", repository: "owner/model-GGUF", commit: "c0ffee" },
        files: [{ path: "model-Q4_K_M.gguf", sizeBytes: 1_000 }],
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
      startDownload: () => Option.some(drafterDownload("drafter", { _tag: "Pending", completedBytes: 0, totalBytes: 9 })),
    })
    const result = await run((commands) => commands.sync(foundId))
    expect(result.outcome).toBe("DownloadingDrafter")
    expect(recorded.started).toEqual([startedDrafter(foundId)])
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
    const updating = harness({
      customModels: withDrafter(true),
      catalog: catalog("NotInstalled"),
      startDownload: () => Option.some(drafterDownload("drafter", { _tag: "Pending", completedBytes: 0, totalBytes: 9 })),
    })
    expect((await updating.run((commands) => commands.sync(foundId))).outcome).toBe("Updating")
    expect(updating.recorded.started).toEqual([startedDrafter(foundId)])
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
  const sameRepositorySetting = "hf:owner/model-GGUF/mtp.gguf@c0ffee"
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
  const config = (drafter: string = sameRepositorySetting) => customModelsConfig({
    added: { [addedId]: { revision: "c0ffee" } },
    drafters: { [addedId]: drafter },
  })
  // At the newer commit the same version, named by its first shard, gains a second shard and a projector; ICN fetches
  // them with the same-repository drafter at that commit.
  const newerTarget = modelPackage("owner/model-GGUF", newer, [
    "model-Q4_K_M.gguf", "model-Q4_K_M-part2.gguf", { path: "mmproj.gguf", role: "projector" },
  ])
  const update = (state: ModelDownloadEncoded["state"], fetched: { readonly drafter: boolean } = { drafter: true }) =>
    Schema.decodeUnknownSync(NativeModelDownload)({
      id: "update",
      forModel: addedId,
      bundle: fetched.drafter
        ? {
            _tag: "SpeculativeDecoding",
            method: { _tag: "Mtp" },
            target: newerTarget,
            draftSource: { _tag: "Separate", draft: modelPackage("owner/model-GGUF", newer, ["mtp.gguf"]) },
          }
        : { _tag: "Standalone", package: newerTarget },
      state,
    })
  const running = update({ _tag: "Downloading", stage: "downloading", completedBytes: 30, totalBytes: 120, bytesPerSecond: 10 })
  const updating = (options: { readonly downloads?: readonly ModelDownload[]; readonly drafter?: string } = {}) => harness({
    config: config(options.drafter),
    customModels: [addedReady(newer)],
    downloads: options.downloads,
    startDownload: () => Option.some(running),
  })
  const settled = <A, E>(effect: Effect.Effect<A, E>) => effect.pipe(Effect.zipLeft(Effect.sleep("30 millis")))
  /** The configuration once the model is pinned to `revision`, as the background pinning gets to it. */
  const pinnedTo = (models: Effect.Effect.Success<typeof CustomModels>, revision: string) => Effect.gen(function* () {
    for (let attempt = 0; attempt < 200; attempt += 1) {
      const current = yield* models.get
      if (current.added[addedId]?.revision === revision) return current
      yield* Effect.sleep("10 millis")
    }
    return yield* models.get
  })
  const pinnedSetting = {
    revision: newer,
    projector: Option.some("mmproj.gguf"),
  }
  const sameRepositoryAt = (revision: string) => ({ _tag: "HuggingFace", repository: "owner/model-GGUF", path: "mtp.gguf", revision })

  it("shows Update for an added model whose repository has a newer target, and Updating while it downloads", () => {
    const row = (model: ReturnType<typeof nativeCustomModel>, related: readonly RelatedDownload[] = []) => Option.getOrThrow(customModel({
      entry: customSource(model), assessment: undefined, instances: [], related, catalog: [],
      drafterChoice: { _tag: "Auto" },
    }))
    expect(row(addedReady(newer)).acquisitionState._tag).toBe("UpdateAvailable")
    expect(row(addedReady()).acquisitionState._tag).toBe("Installed")
    // The download of the newer commit counts as the target's, so the model reads as updating, still installed.
    const related = customModelDownloads(addedId, config(), [running])
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
      const after = yield* pinnedTo(models, newer)
      return { synced, during, after }
    }))
    expect(result.synced.outcome).toBe("Updating")
    // ICN works out what the update fetches; ACN asks for it by intent.
    expect(recorded.started).toEqual([startedUpdate(addedId)])
    // Nothing changes while it downloads, so ICN keeps serving the pinned commit.
    expect(result.during.added[addedId]).toEqual({ revision: "c0ffee", projector: Option.none() })
    expect(localModelsConfiguration(result.during, []).added).toEqual([
      { repository: "owner/model-GGUF", path: "model-Q4_K_M.gguf", revision: "c0ffee", projector: Option.none() },
    ])
    expect(result.during.drafters[addedId]).toEqual(sameRepositoryAt("c0ffee"))
    // Once it completes: the newer revision, its projector, and the same-repository drafter it fetched.
    expect(result.after.added[addedId]).toEqual(pinnedSetting)
    expect(result.after.drafters[addedId]).toEqual(sameRepositoryAt(newer))
  })

  it("pins an update that finished while the service was not running", async () => {
    const { run, recorded } = updating({ downloads: [update({ _tag: "Completed" })] })
    const config = await run((_, models) => pinnedTo(models, newer))
    expect(recorded.started).toEqual([])
    expect(config.added[addedId]).toEqual(pinnedSetting)
    expect(config.drafters[addedId]).toEqual(sameRepositoryAt(newer))
  })

  it("moves the revision but keeps a drafter the update didn't fetch, or one from another repository", async () => {
    const notFetched = updating({ downloads: [update({ _tag: "Completed" }, { drafter: false })] })
    const kept = await notFetched.run((_, models) => pinnedTo(models, newer))
    expect(kept.added[addedId]).toEqual(pinnedSetting)
    expect(kept.drafters[addedId]).toEqual(sameRepositoryAt("c0ffee"))

    const otherRepository = updating({ drafter: drafterSetting, downloads: [update({ _tag: "Completed" })] })
    const other = await otherRepository.run((_, models) => pinnedTo(models, newer))
    expect(other.added[addedId]).toEqual(pinnedSetting)
    expect(other.drafters[addedId]).toEqual({
      _tag: "HuggingFace", repository: "magnitudedev/model-DFlash-GGUF", path: "dflash.gguf", revision: "d1",
    })
  })

  it("ignores a cancelled or failed update and stays on the pinned commit", async () => {
    for (const ended of [
      update({ _tag: "Cancelled", completedBytes: 30, totalBytes: 120 }),
      update({ _tag: "Failed", acknowledged: false, completedBytes: 30, totalBytes: 120, failure: { _tag: "Interrupted" } }),
    ]) {
      const { run, recorded, downloadList } = updating()
      const result = await run((commands, models) => Effect.gen(function* () {
        yield* commands.sync(addedId)
        yield* settled(commands.cancelSync(addedId))
        yield* settled(SubscriptionRef.set(downloadList, [ended]))
        return yield* models.get
      }))
      expect(recorded.cancelled).toEqual(["update"])
      expect(result.added[addedId]).toEqual({ revision: "c0ffee", projector: Option.none() })
      expect(result.drafters[addedId]).toEqual(sameRepositoryAt("c0ffee"))
    }
  })

  it("is current without a newer target and changes nothing", async () => {
    const { run, recorded } = harness({ config: config(), customModels: [addedReady()] })
    const result = await run((commands, models) => Effect.gen(function* () {
      const synced = yield* commands.sync(addedId)
      return { synced, config: yield* models.get }
    }))
    expect(result.synced.outcome).toBe("AlreadyCurrent")
    expect(result.config.added[addedId]).toEqual({ revision: "c0ffee", projector: Option.none() })
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

  it("hides a catalog model whose files are only in an external location by its catalog ID, and refuses one Magnitude owns", async () => {
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
    expect(result.hidden).toEqual([external])
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
    expect(result.hidden).toEqual([external])
    expect(Option.isSome(result.files)).toBe(true)
  })

  it("unhides a catalog model named by an alias, removing its catalog ID and every alias by identity", async () => {
    const alias = CustomFormModelIdSchema.make("local:qwen3.6-35b-a3b-q4_k_m:abcdefabcdef")
    const olderLabel = CustomFormModelIdSchema.make("local:qwen3.6-old-q4_k_m:abcdefabcdef")
    const { run } = harness({
      config: customModelsConfig({ hidden: [external, olderLabel, foundId] }),
      catalogRows: [{ _tag: "Catalog", modelId: external, acquisitionState: { _tag: "NotInstalled" } } as unknown as LocalModelsState["models"][number]],
      catalog: [{ id: external, source: { aliases: Option.some([alias]) } }] as unknown as readonly CatalogModelSource[],
    })
    const hidden = await run((commands, models) => commands.unhide(alias).pipe(Effect.zipRight(models.get), Effect.map(({ hidden }) => hidden)))
    expect(hidden).toEqual([foundId])
  })

  it("hides an already hidden catalog model again without change, and unhides it by its catalog ID", async () => {
    const alias = CustomFormModelIdSchema.make("hf:unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf")
    const { run } = harness({
      config: customModelsConfig({ hidden: [external] }),
      catalogRows: [{ _tag: "Catalog", modelId: external, acquisitionState: { _tag: "NotInstalled" } } as unknown as LocalModelsState["models"][number]],
      catalog: [{ id: external, source: { aliases: Option.some([alias]) } }] as unknown as readonly CatalogModelSource[],
    })
    const result = await run((commands, models) => Effect.gen(function* () {
      yield* commands.hide(external)
      const hidden = (yield* models.get).hidden
      yield* commands.unhide(external)
      return { hidden, afterUnhide: (yield* models.get).hidden }
    }))
    expect(result.hidden).toEqual([external])
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
    expect(result.afterAdd.paths).toEqual([folder])
    expect(result.afterAdd.locations).toEqual({ ollama: false })
    expect(result.locations.find(({ kind }) => kind === "Ollama")).toMatchObject({ enabled: false, automatic: true, state: "NotFound" })
    expect(result.locations.find(({ kind }) => kind === "Folder"))
      .toEqual({ kind: "Folder", paths: [{ path: folder, state: "Found" }], enabled: true, automatic: false, state: "Found" })
    expect(result.locations.find(({ kind }) => kind === "Ollama")?.paths.map(({ path }) => path)).toEqual([
      "/nonexistent/ollama", "/nonexistent-home/.ollama/models", "/usr/share/ollama/.ollama/models", "/var/lib/ollama",
    ])
    expect(result.afterRemove.paths).toEqual([])
  })

  it("fails a location change the inference service refuses", async () => {
    const folder = mkdtempSync(join(tmpdir(), "magnitude-folder-"))
    const refused = new LocalModelMutationFailed({ code: "local_models_configuration_failed", message: "refused", retryable: true })
    const { run } = harness({ pushFailure: refused })
    const result = await run((commands) => Effect.gen(function* () {
      return {
        add: yield* commands.addFolder(folder).pipe(Effect.either),
        remove: yield* commands.removeFolder(folder).pipe(Effect.either),
        disable: yield* commands.setLocationEnabled("Ollama", false).pipe(Effect.either),
        enable: yield* commands.setLocationEnabled("Ollama", true).pipe(Effect.either),
      }
    }))
    for (const outcome of Object.values(result)) expect(outcome).toEqual(Either.left(refused))
  })
})

