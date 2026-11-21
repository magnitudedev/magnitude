import { describe, expect, it } from "vitest"
import { Atom, Registry, Result } from "@effect-atom/atom-react"
import { Context, Deferred, Effect, Either, Layer, Option, Schema, Stream } from "effect"
import { HuggingFaceArtifactSelectorSchema, HuggingFaceRepositoryIdSchema } from "@magnitudedev/sdk/desktop-host"
import { ApplicationSession, ApplicationSessionLive, HostAction, modelTrayPresentation } from "./session"
import { ApplicationHost } from "./host"
import { ApplicationRouterMemory } from "./router"
import { LocalModels } from "../local-models/service"
import { makeSetupModel } from "./fixtures/model"
import { Client as EffectQueryClient } from "@magnitudedev/effect-query"
import { AcnQueries } from "../operations"
import { ClientEffectQuery } from "../state/client-effect-query"
import { fakeAcnImplementationsLayer } from "../state/fake-acn-implementations"
import { LocalInferenceDeviceIdSchema, LocalInferenceMemoryDomainIdSchema, type CatalogLocalModel, type LocalModelsState } from "@magnitudedev/sdk"

const observed = (residencyState: Extract<CatalogLocalModel["acquisitionState"], { _tag: "Installed" }>["residencyState"]): LocalModelsState => ({
  preparation: { discovery: { complete: true, modelsFound: 1 }, assessment: { complete: true, settledModels: 1, totalModels: 1 }, measurement: { _tag: "Measured" } },
  unsupported: [],
  hidden: [],
  models: [{ ...makeSetupModel(true), acquisitionState: {
    _tag: "Installed", installation: { _tag: "Resolved", primaryPath: "/models/test.gguf", installedBytes: 1, ownership: "Magnitude" }, residencyState,
  } }],
})

const ready = observed({ _tag: "Unloaded" })
const none: LocalModelsState = { ...ready, models: [makeSetupModel(false)] }
const downloading: LocalModelsState = { ...ready, models: [{ ...makeSetupModel(false), acquisitionState: { _tag: "Installing" } } as unknown as CatalogLocalModel] }

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
    ], promptGroupPlaces: { held: 1, capacity: 1 }, streaming: Option.none() }, device: { deviceId: LocalInferenceDeviceIdSchema.make("metal:0"), backend: "metal" } }))
    expect(Option.map(loaded.status, ({ detail }) => detail)).toEqual(Option.some({ _tag: "Memory", text: "19.3 GB" }))
    expect(loaded.label).toMatch(/ · Loaded · 19\.3 GB$/)
  })
})

describe("host actions", () => {
  const repository = HuggingFaceRepositoryIdSchema.make("unsloth/Qwen3.6-27B-GGUF")
  const file = HuggingFaceArtifactSelectorSchema.make("Qwen3.6-27B-Q4_K_M.gguf")
  const withSession = <A, E>(
    actions: ReadonlyArray<HostAction>,
    use: (session: ApplicationSession, registry: Registry.Registry) => Effect.Effect<A, E>,
    onboarding: { current: "Pending" | "Completed" } = { current: "Completed" },
    localModels: Atom.Writable<Result.Result<LocalModelsState>> = Atom.make<Result.Result<LocalModelsState>>(Result.initial()),
    loads: string[] = [],
  ) => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const delivered = yield* Deferred.make<void>()
    const effectQuery = EffectQueryClient.make(AcnQueries, fakeAcnImplementationsLayer((tag) => {
      if (tag === "GetOnboarding") return Effect.sync(() => ({ onboarding: onboarding.current }))
      if (tag === "CompleteOnboarding") return Effect.sync(() => { onboarding.current = "Completed" }).pipe(Effect.as({}))
      return Effect.dieMessage(`Unexpected RPC in session test: ${tag}`)
    }))
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
          chooseOpenPath: () => Effect.succeed(Option.none()),
          application: Stream.never,
        }),
      })),
      Layer.provide(Layer.succeed(LocalModels, { state: localModels, stop: Effect.void, load: (modelId: string) => Effect.sync(() => { loads.push(modelId) }) } as unknown as LocalModels)),
      Layer.provide(Layer.succeed(ClientEffectQuery, effectQuery)),
      Layer.provideMerge(Registry.layer),
    ))
    yield* Deferred.await(delivered)
    const registry = Context.get(context, Registry.AtomRegistry)
    const session = Context.get(context, ApplicationSession)
    registry.get(session.onboardingPending)
    yield* Registry.toStream(registry, effectQuery.Configuration.GetOnboarding({})).pipe(
      Stream.filter(state => Result.isSuccess(state.result)),
      Stream.take(1),
      Stream.runDrain,
    )
    return yield* use(session, registry)
  })))

  it("reports onboarding pending over the requested page", () => withSession([{ _tag: "Navigate", page: "chat" }], (session, registry) => Effect.sync(() => {
    expect(registry.get(session.onboardingPending)).toEqual(Option.some(true))
    expect(registry.get(session.page)).toBe("chat")
  }), { current: "Pending" }))

  it("opens Models once onboarding is completed while its download is still running", () => {
    const models = Atom.make<Result.Result<LocalModelsState>>(Result.success(downloading))
    return withSession([{ _tag: "Navigate", page: "chat" }], (session, registry) => Effect.gen(function* () {
      registry.set(session.completeOnboarding, Option.none())
      yield* Registry.getResult(registry, session.completeOnboarding, { suspendOnWaiting: true })
      expect(registry.get(session.onboardingPending)).toEqual(Option.some(false))
      expect(registry.get(session.page)).toBe("models")
    }), { current: "Pending" }, models)
  })

  it("loads the model onboarding downloaded and opens Chat once its download finishes", () => {
    const models = Atom.make<Result.Result<LocalModelsState>>(Result.success(downloading))
    const loads: string[] = []
    const modelId = downloading.models[0]!.modelId
    return withSession([], (session, registry) => Effect.gen(function* () {
      registry.set(session.completeOnboarding, Option.some(modelId))
      yield* Registry.getResult(registry, session.completeOnboarding, { suspendOnWaiting: true })
      expect(registry.get(session.page)).toBe("models")
      expect(loads).toEqual([])
      registry.set(models, Result.success(ready))
      yield* Effect.sleep("20 millis")
      expect(loads).toEqual([modelId])
      expect(registry.get(session.page)).toBe("chat")
    }), { current: "Pending" }, models, loads)
  })

  it("does nothing more once onboarding's download is cancelled", () => {
    const models = Atom.make<Result.Result<LocalModelsState>>(Result.success(downloading))
    const loads: string[] = []
    return withSession([], (session, registry) => Effect.gen(function* () {
      registry.set(session.completeOnboarding, Option.some(downloading.models[0]!.modelId))
      yield* Registry.getResult(registry, session.completeOnboarding, { suspendOnWaiting: true })
      registry.set(models, Result.success({ ...downloading, models: [] }))
      yield* Effect.sleep("20 millis")
      registry.set(models, Result.success(ready))
      yield* Effect.sleep("20 millis")
      expect(loads).toEqual([])
      expect(registry.get(session.page)).toBe("models")
    }), { current: "Pending" }, models, loads)
  })

  it("opens Chat once onboarding is completed with a model ready", () => {
    const models = Atom.make<Result.Result<LocalModelsState>>(Result.success(ready))
    return withSession([{ _tag: "Navigate", page: "models" }], (session, registry) => Effect.gen(function* () {
      registry.set(session.completeOnboarding, Option.none())
      yield* Registry.getResult(registry, session.completeOnboarding, { suspendOnWaiting: true })
      expect(registry.get(session.page)).toBe("chat")
    }), { current: "Pending" }, models)
  })

  it("starts on Chat until the service answers, and stays there if it never does", () => withSession([], (session, registry) => Effect.gen(function* () {
    expect(registry.get(session.onboardingPending)).toEqual(Option.some(false))
    yield* Effect.sleep("20 millis")
    expect(registry.get(session.page)).toBe("chat")
  })))

  it("lands on Chat when a model is ready", () => {
    const models = Atom.make<Result.Result<LocalModelsState>>(Result.initial())
    return withSession([], (session, registry) => Effect.gen(function* () {
      registry.set(models, Result.success(ready))
      yield* Effect.sleep("20 millis")
      expect(registry.get(session.page)).toBe("chat")
    }), { current: "Completed" }, models)
  })

  it("lands on Models when no model is downloaded, or only a download is running", () => Effect.runPromise(Effect.forEach([none, downloading], (state) => Effect.promise(() => {
    const models = Atom.make<Result.Result<LocalModelsState>>(Result.initial())
    return withSession([], (session, registry) => Effect.gen(function* () {
      registry.set(models, Result.success(state))
      yield* Effect.sleep("20 millis")
      expect(registry.get(session.page)).toBe("models")
    }), { current: "Completed" }, models)
  }), { discard: true })))

  it("waits for the service to finish finding models, then decides once", () => {
    const models = Atom.make<Result.Result<LocalModelsState>>(Result.initial())
    return withSession([], (session, registry) => Effect.gen(function* () {
      registry.set(models, Result.success({ ...none, preparation: { ...none.preparation, discovery: { complete: false, modelsFound: 0 } } }))
      yield* Effect.sleep("20 millis")
      expect(registry.get(session.page)).toBe("chat")
      registry.set(models, Result.success(none))
      yield* Effect.sleep("20 millis")
      expect(registry.get(session.page)).toBe("models")
      registry.set(models, Result.success(ready))
      yield* Effect.sleep("20 millis")
      expect(registry.get(session.page)).toBe("models")
    }), { current: "Completed" }, models)
  })

  it("keeps the page someone navigated to before the service answered", () => {
    const models = Atom.make<Result.Result<LocalModelsState>>(Result.initial())
    return withSession([{ _tag: "Navigate", page: "usage" }], (session, registry) => Effect.gen(function* () {
      registry.set(models, Result.success(none))
      yield* Effect.sleep("20 millis")
      expect(registry.get(session.page)).toBe("usage")
    }), { current: "Completed" }, models)
  })

  it("keeps the main page in the main area while Settings is open over it", () => withSession([], (session, registry) => Effect.gen(function* () {
    registry.get(session.mainPage)
    yield* session.navigate("models")
    yield* session.navigate("settings")
    expect(registry.get(session.mainPage)).toBe("models")
    yield* session.navigate("chat")
    yield* session.navigate("connections")
    expect(registry.get(session.mainPage)).toBe("connections")
    yield* session.navigate("usage")
    expect(registry.get(session.mainPage)).toBe("connections")
  })))

  it("starts Models over when Models is chosen while it is shown, but not when Settings closes over it", () => withSession([], (session, registry) => Effect.gen(function* () {
    yield* session.navigate("models")
    const opened = registry.get(session.visit)
    yield* session.navigate("settings")
    yield* session.navigate("models")
    expect(registry.get(session.visit)).toBe(opened)
    yield* session.navigate("models")
    expect(registry.get(session.visit)).toBe(opened + 1)
  })))

  it("drops a request to add a model when Models is chosen again, as Explore's back arrow does", () => withSession([], (session, registry) => Effect.gen(function* () {
    yield* session.requestAddModel({ _tag: "Search", query: "Qwen3.6 27B" })
    yield* session.navigate("settings")
    yield* session.navigate("models")
    expect(registry.get(session.addModelRequest)).toEqual(Option.some({ _tag: "Search", query: "Qwen3.6 27B" }))
    yield* session.navigate("models")
    expect(registry.get(session.addModelRequest)).toEqual(Option.none())
  })))

  it("opens a Hugging Face link on Models with the add request set", () => withSession([{ _tag: "OpenHuggingFaceModel", repository, file: Option.some(file) }], (session, registry) => Effect.sync(() => {
    expect(registry.get(session.page)).toBe("models")
    expect(registry.get(session.addModelRequest)).toEqual(Option.some({ _tag: "HuggingFace", repository, file: Option.some(file) }))
  })))

  it("sets and clears a search request from the app itself", () => withSession([], (session, registry) => Effect.gen(function* () {
    yield* session.navigate("chat")
    expect(registry.get(session.addModelRequest)).toEqual(Option.none())
    yield* session.requestAddModel({ _tag: "Search", query: "Qwen3.6 27B" })
    expect(registry.get(session.page)).toBe("models")
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
    expect(Either.isLeft(decode({ _tag: "Navigate", page: "discover" }))).toBe(true)
  })
})
