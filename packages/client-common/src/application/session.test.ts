import { describe, expect, it } from "vitest"
import { Atom, Registry, Result } from "@effect-atom/atom-react"
import { Context, Deferred, Effect, Either, Layer, Option, Schema, Stream } from "effect"
import { HuggingFaceArtifactSelectorSchema, HuggingFaceRepositoryIdSchema } from "@magnitudedev/sdk/desktop-host"
import { ApplicationSession, ApplicationSessionLive, HostAction, modelTrayPresentation } from "./session"
import { ApplicationHost } from "./host"
import { ApplicationRouterMemory } from "./router"
import { LocalModels } from "../local-models/service"
import { makeSetupModel } from "./fixtures/model"
import { LocalInferenceMemoryDomainIdSchema, type CatalogLocalModel, type LocalModelsState } from "@magnitudedev/sdk"

const observed = (residencyState: Extract<CatalogLocalModel["acquisitionState"], { _tag: "Installed" }>["residencyState"]): LocalModelsState => ({
  preparation: { discovery: { complete: true, modelsFound: 1 }, assessment: { complete: true, settledModels: 1, totalModels: 1 } },
  unsupported: [],
  hidden: [],
  models: [{ ...makeSetupModel(true), acquisitionState: {
    _tag: "Installed", installation: { _tag: "Resolved", primaryPath: "/models/test.gguf", installedBytes: 1, ownership: "Magnitude" }, residencyState,
  } }],
})

describe("desktop model retirement controls", () => {
  it("keeps Stop available while native cleanup is incomplete", () => {
    const presentation = modelTrayPresentation(observed({ _tag: "Stopping", reason: "user_stop", allocation: { _tag: "Planned", allocation: Option.none() } }))
    expect(presentation.label).toContain("Stopping")
    expect(presentation.canStop).toBe(true)
  })
  it("removes Stop only after observed unloading", () => {
    expect(modelTrayPresentation(observed({ _tag: "Unloaded" }))).toEqual({ label: "No model loaded", status: Option.none(), canStop: false })
  })
})

describe("tray model line", () => {
  it("shows a measured stage's progress and an unmeasured stage's work", () => {
    const optimizing = modelTrayPresentation(observed({ _tag: "Loading", stage: "optimizing", fraction: 0.34, plannedAllocation: Option.none() }))
    expect(Option.map(optimizing.status, ({ phase, detail }) => ({ phase, detail }))).toEqual(
      Option.some({ phase: "Optimizing", detail: { _tag: "Progress", fraction: 0.34 } }))
    expect(optimizing.label).toMatch(/ · Optimizing 34%$/)
    const preparing = modelTrayPresentation(observed({ _tag: "Loading", stage: "preparing", fraction: 0, plannedAllocation: Option.none() }))
    expect(Option.map(preparing.status, ({ phase, detail }) => ({ phase, detail }))).toEqual(
      Option.some({ phase: "Preparing", detail: { _tag: "Working" } }))
  })
  it("shows the memory a loaded model holds", () => {
    const loaded = modelTrayPresentation(observed({ _tag: "Ready", allocation: { contextWindowTokens: 4096, memoryDomains: [
      { memoryDomainId: LocalInferenceMemoryDomainIdSchema.make("unified"), modelBytes: 16 * 1024 ** 3, contextBytes: 2 * 1024 ** 3, computeBytes: 0, auxiliaryBytes: 0 },
    ] } }))
    expect(Option.map(loaded.status, ({ detail }) => detail)).toEqual(Option.some({ _tag: "Memory", text: "18 GB" }))
    expect(loaded.label).toMatch(/ · Loaded · 18 GB$/)
  })
})

describe("host actions", () => {
  const repository = HuggingFaceRepositoryIdSchema.make("unsloth/Qwen3.6-27B-GGUF")
  const file = HuggingFaceArtifactSelectorSchema.make("Qwen3.6-27B-Q4_K_M.gguf")
  const withSession = <A>(actions: ReadonlyArray<HostAction>, use: (session: ApplicationSession, registry: Registry.Registry) => Effect.Effect<A>) => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const delivered = yield* Deferred.make<void>()
    const context = yield* Layer.build(ApplicationSessionLive.pipe(
      Layer.provideMerge(ApplicationRouterMemory),
      Layer.provide(Layer.succeed(ApplicationHost, {
        window: Option.none(),
        appearance: { read: Effect.succeed("system" as const), save: () => Effect.void },
        shell: Option.some({
          actions: Stream.concat(Stream.fromIterable(actions), Stream.execute(Deferred.succeed(delivered, undefined))),
          presentModel: () => Effect.void,
          resolveQuitFailure: () => Effect.void,
          retryService: Effect.void,
          application: Stream.never,
        }),
      })),
      Layer.provide(Layer.succeed(LocalModels, { state: Atom.make(Result.initial()), stop: Effect.void } as unknown as LocalModels)),
      Layer.provideMerge(Registry.layer),
    ))
    yield* Deferred.await(delivered)
    return yield* use(Context.get(context, ApplicationSession), Context.get(context, Registry.AtomRegistry))
  })))

  it("opens a Hugging Face link on the catalog with the add request set", () => withSession([{ _tag: "OpenHuggingFaceModel", repository, file: Option.some(file) }], (session, registry) => Effect.sync(() => {
    expect(registry.get(session.page)).toBe("catalog")
    expect(registry.get(session.addModelRequest)).toEqual(Option.some({ _tag: "HuggingFace", repository, file: Option.some(file) }))
  })))

  it("sets and clears a search request from the app itself", () => withSession([], (session, registry) => Effect.gen(function* () {
    expect(registry.get(session.addModelRequest)).toEqual(Option.none())
    yield* session.requestAddModel({ _tag: "Search", query: "Qwen3.6 27B" })
    expect(registry.get(session.page)).toBe("catalog")
    expect(registry.get(session.addModelRequest)).toEqual(Option.some({ _tag: "Search", query: "Qwen3.6 27B" }))
    yield* session.clearAddModelRequest
    expect(registry.get(session.addModelRequest)).toEqual(Option.none())
  })))

  it("decodes the wire form with and without a file and rejects unsafe values", () => {
    const decode = Schema.decodeUnknownEither(HostAction)
    expect(decode({ _tag: "OpenHuggingFaceModel", repository: "owner/repo" })).toEqual(Either.right({ _tag: "OpenHuggingFaceModel", repository: "owner/repo", file: Option.none() }))
    expect(Either.isRight(decode({ _tag: "OpenHuggingFaceModel", repository: "owner/repo", file: "dir/model.gguf" }))).toBe(true)
    expect(Either.isLeft(decode({ _tag: "OpenHuggingFaceModel", repository: "owner/repo/extra" }))).toBe(true)
    expect(Either.isLeft(decode({ _tag: "OpenHuggingFaceModel", repository: "owner/repo", file: "../model.gguf" }))).toBe(true)
  })
})
