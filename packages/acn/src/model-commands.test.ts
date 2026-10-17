import { Effect, Fiber, Layer, Option, Schema, SubscriptionRef } from "effect"
import { CatalogModelSyncLive } from "./catalog-model-sync"
import { describe, expect, it } from "vitest"
import { modelCommandFailure, type ModelCommandsApi } from "./model-commands"
import { CustomModelCommands, type CustomModelCommandsApi } from "./custom-model-commands"
import { LocalModels } from "./local-models"

const UPGRADED_ALIAS = "hf:unsloth/Qwen3.5-4B-MTP-GGUF/Qwen3.5-4B-UD-Q4_K_XL.gguf"
const localModels = {
  state: Effect.succeed({
    models: [{ _tag: "Catalog", modelId: "model-a:gguf:test", catalogData: { aliases: [UPGRADED_ALIAS] } }],
    hidden: [],
  }),
} as never

const customCalls: string[] = []
const customCommands = {
  remove: (modelId: string) => Effect.sync(() => { customCalls.push(`custom remove ${modelId}`); return {} }),
  sync: (modelId: string) => Effect.sync(() => { customCalls.push(`custom sync ${modelId}`); return { outcome: "Installing" as const } }),
  cancelSync: (modelId: string) => Effect.sync(() => { customCalls.push(`custom cancel ${modelId}`); return {} }),
  acknowledgeSyncFailure: (modelId: string) => Effect.sync(() => { customCalls.push(`custom acknowledge ${modelId}`); return {} }),
  externalCatalogFiles: (modelId: string) => Effect.succeed(modelId === "model-c:gguf:test"
    ? Option.some({ path: "/home/ada/.cache/huggingface/hub/models--o--r/snapshots/c/m.gguf", location: Option.some("HuggingFaceCache") })
    : Option.none()),
} as unknown as CustomModelCommandsApi

describe("modelCommandFailure", () => {
  it("preserves the server's structured remote failure details", () => {
    const failure = modelCommandFailure("load_model", {
      _tag: "GeneratedClientRemoteError",
      operationId: "ensureModelInstance",
      status: 409,
      headers: {},
      body: {
        error: {
          code: "memory_shortage",
          message: "Not enough memory to load the selected model",
          type: "model_error",
          param: Option.none(),
        },
      },
    } as never)

    expect(failure).toMatchObject({
      _tag: "LocalModelMutationFailed",
      code: "memory_shortage",
      message: "Not enough memory to load the selected model",
      retryable: true,
    })
  })

  it("uses the underlying transport message instead of the wrapper tag", () => {
    const failure = modelCommandFailure("load_model", {
      _tag: "GeneratedClientTransportError",
      operationId: "ensureModelInstance",
      cause: new Error("ICN connection closed while loading"),
    } as never)

    expect(failure).toMatchObject({
      _tag: "LocalModelMutationFailed",
      code: "model_load_model_transport_failed",
      message: "ICN connection closed while loading",
      retryable: true,
    })
  })
})

it("retries active and slot Stop against the exact stopping instance", async () => {
  const { Effect, Schema } = await import("effect")
  const { IcnClient, IcnInstances, IcnCatalog, IcnCatalogInstallations } = await import("@magnitudedev/icn")
  const { ModelInstancesSnapshot } = await import("@magnitudedev/icn-protocol/schemas")
  const { PRIMARY_SLOT_ID } = await import("@magnitudedev/acn-protocol")
  const { ModelCommands, ModelCommandsLive } = await import("./model-commands")
  const { ModelSlotController } = await import("./model-slot-controller")
  const { LocalModelRemovals } = await import("./local-model-removals")
  const stopped: string[] = []
  const snapshot = Schema.decodeUnknownSync(ModelInstancesSnapshot)({ revision: 1, instances: [{
    id: "retained-instance", modelId: "model-a:gguf:test",
    lifecycle: { _tag: "Stopping", reason: "user_stop", allocation: { _tag: "Planned" } },
  }] })
  // These focused host fixtures expose only operations used by Stop; unrelated capabilities
  // are intentionally absent so this test cannot perform catalog or filesystem work.
  await Effect.runPromise(Effect.gen(function* () {
    const commands = yield* ModelCommands
    yield* commands.stopActiveModel
    yield* commands.stopSlot(PRIMARY_SLOT_ID)
  }).pipe(
    Effect.provide(ModelCommandsLive.pipe(Layer.provide(CatalogModelSyncLive))),
    Effect.provideService(CustomModelCommands, customCommands),
    Effect.provideService(LocalModels, localModels),
    Effect.provideService(IcnInstances, { get: Effect.succeed(snapshot) } as never),
    Effect.provideService(IcnClient, { models: { stopModelInstance: ({ path }: { path: { instance_id: string } }) => Effect.sync(() => { stopped.push(path.instance_id); return {} }) } } as never),
    Effect.provideService(ModelSlotController, { state: Effect.succeed({ slots: { primary: { _tag: "ConfiguredLocal", selection: { providerModelId: "model-a:gguf:test" } } } }) } as never),
    Effect.provideService(IcnCatalog, {} as never),
    Effect.provideService(IcnCatalogInstallations, {} as never),
    Effect.provideService(LocalModelRemovals, {} as never),
  ))
  expect(stopped).toEqual(["retained-instance", "retained-instance"])
})

it("stops a running model before removing it, and removes an idle model without stopping", async () => {
  const { Effect, Schema } = await import("effect")
  const { IcnClient, IcnInstances, IcnCatalog, IcnCatalogInstallations } = await import("@magnitudedev/icn")
  const { ModelInstancesSnapshot } = await import("@magnitudedev/icn-protocol/schemas")
  const { ModelCommands, ModelCommandsLive } = await import("./model-commands")
  const { ModelSlotController } = await import("./model-slot-controller")
  const { LocalModelRemovals } = await import("./local-model-removals")
  const calls: string[] = []
  const snapshot = Schema.decodeUnknownSync(ModelInstancesSnapshot)({ revision: 1, instances: [{
    id: "loaded-instance", modelId: "model-a:gguf:test",
    lifecycle: { _tag: "Loading", fraction: 0.5, stage: "loading_weights" },
  }] })
  const { ModelIdSchema } = await import("@magnitudedev/acn-protocol")
  const remove = (model: string) => Effect.gen(function* () {
    const commands = yield* ModelCommands
    yield* commands.remove(Schema.decodeUnknownSync(ModelIdSchema)(model))
  }).pipe(
    Effect.provide(ModelCommandsLive.pipe(Layer.provide(CatalogModelSyncLive))),
    Effect.provideService(CustomModelCommands, customCommands),
    Effect.provideService(LocalModels, localModels),
    Effect.provideService(IcnInstances, { get: Effect.succeed(snapshot) } as never),
    Effect.provideService(IcnClient, { models: { stopModelInstance: ({ path }: { path: { instance_id: string } }) => Effect.sync(() => { calls.push(`stop ${path.instance_id}`); return {} }) } } as never),
    Effect.provideService(LocalModelRemovals, { remove: (modelId: string) => Effect.sync(() => { calls.push(`remove ${modelId}`); return {} }) } as never),
    Effect.provideService(ModelSlotController, {} as never),
    Effect.provideService(IcnCatalog, {} as never),
    Effect.provideService(IcnCatalogInstallations, {} as never),
  )
  await Effect.runPromise(remove("model-a:gguf:test"))
  await Effect.runPromise(remove("model-b:gguf:test"))
  await Effect.runPromise(remove("hf:owner/repo/model-Q4_K_M.gguf"))
  const external = await Effect.runPromise(remove("model-c:gguf:test").pipe(Effect.flip))
  expect(calls).toEqual(["stop loaded-instance", "remove model-a:gguf:test", "remove model-b:gguf:test"])
  expect(external.message).toBe("model-c:gguf:test uses files in your Hugging Face cache "
    + "(/home/ada/.cache/huggingface/hub/models--o--r/snapshots/c/m.gguf), which Magnitude never deletes. "
    + "Hide it with `magnitude models hide model-c:gguf:test`, or delete it with the Hugging Face CLI (`hf cache delete`).")
  expect(customCalls).toContain("custom remove hf:owner/repo/model-Q4_K_M.gguf")
})

it("reports whether a catalog sync installs, updates, or finds the model current", async () => {
  const { IcnClient, IcnInstances, IcnCatalog, IcnCatalogInstallations } = await import("@magnitudedev/icn")
  const { ModelIdSchema } = await import("@magnitudedev/acn-protocol")
  const { ModelCommands, ModelCommandsLive } = await import("./model-commands")
  const { ModelSlotController } = await import("./model-slot-controller")
  const { LocalModelRemovals } = await import("./local-model-removals")
  const sync = (localState: string, admission: string) => Effect.gen(function* () {
    const commands = yield* ModelCommands
    return (yield* commands.sync(Schema.decodeUnknownSync(ModelIdSchema)("model-a:gguf:test"))).outcome
  }).pipe(
    Effect.provide(ModelCommandsLive.pipe(Layer.provide(CatalogModelSyncLive))),
    Effect.provideService(CustomModelCommands, customCommands),
    Effect.provideService(LocalModels, localModels),
    Effect.provideService(IcnInstances, { get: Effect.succeed({ revision: 1, instances: [] }) } as never),
    Effect.provideService(IcnClient, {
      catalog: { installCatalogModel: () => Effect.succeed({ _tag: admission, operationId: "op" }) },
    } as never),
    Effect.provideService(IcnCatalog, {
      get: Effect.succeed({ revision: 1, state: { models: [{ id: "model-a:gguf:test", localState: { _tag: localState } }] } }),
      refresh: Effect.void,
    } as never),
    Effect.provideService(IcnCatalogInstallations, { refresh: Effect.void } as never),
    Effect.provideService(ModelSlotController, {} as never),
    Effect.provideService(LocalModelRemovals, { forget: () => Effect.void } as never),
  )
  expect(await Effect.runPromise(sync("NotInstalled", "Admitted"))).toBe("Installing")
  expect(await Effect.runPromise(sync("Installed", "Admitted"))).toBe("Updating")
  expect(await Effect.runPromise(sync("Installed", "Current"))).toBe("AlreadyCurrent")
})

describe("load acknowledgement", () => {
  const loadFixture = async (initial: unknown, ensure: (modelId: string) => unknown) => {
    const { IcnClient, IcnInstances, IcnCatalog, IcnCatalogInstallations } = await import("@magnitudedev/icn")
    const { ModelInstancesSnapshot } = await import("@magnitudedev/icn-protocol/schemas")
    const { ModelCommands, ModelCommandsLive } = await import("./model-commands")
    const { ModelSlotController } = await import("./model-slot-controller")
    const { LocalModelRemovals } = await import("./local-model-removals")
    const decode = Schema.decodeUnknownSync(ModelInstancesSnapshot)
    const instances = await Effect.runPromise(SubscriptionRef.make(decode(initial)))
    const slotsRefreshed: string[] = []
    const run = <A, E>(use: (commands: ModelCommandsApi) => Effect.Effect<A, E>) =>
      Effect.runPromise(ModelCommands.pipe(
        Effect.flatMap(use),
        Effect.either,
        Effect.provide(ModelCommandsLive.pipe(Layer.provide(CatalogModelSyncLive))),
        Effect.provideService(CustomModelCommands, customCommands),
    Effect.provideService(LocalModels, localModels),
    Effect.provideService(CustomModelCommands, customCommands),
    Effect.provideService(LocalModels, localModels),
        Effect.provideService(IcnInstances, { get: SubscriptionRef.get(instances), changes: instances.changes } as never),
        Effect.provideService(IcnClient, { models: { ensureModelInstance: ({ payload }: { payload: { modelId: string } }) => ensure(payload.modelId) } } as never),
        Effect.provideService(ModelSlotController, {
          state: Effect.succeed({ slots: { primary: { _tag: "ConfiguredLocal", selection: { providerModelId: "model-a:gguf:test" } } } }),
          refresh: Effect.sync(() => { slotsRefreshed.push("primary") }),
        } as never),
        Effect.provideService(IcnCatalog, {} as never),
        Effect.provideService(IcnCatalogInstallations, {} as never),
        Effect.provideService(LocalModelRemovals, {} as never),
      ))
    const publish = (snapshot: unknown) => SubscriptionRef.set(instances, decode(snapshot))
    return { run, publish, slotsRefreshed }
  }
  const loading = (revision: number) => ({ revision, instances: [{
    id: "instance-1", modelId: "model-a:gguf:test",
    lifecycle: { _tag: "Loading", stage: "optimizing", fraction: 0.1 },
  }] })

  it("acknowledges a load once ICN admits it while the warm demand keeps waiting for readiness", async () => {
    const { ModelIdSchema } = await import("@magnitudedev/acn-protocol")
    const demands: string[] = []
    let demandInterrupted = false
    const fixture = await loadFixture({ revision: 1, instances: [] }, (modelId) => Effect.sync(() => { demands.push(modelId) }).pipe(
      Effect.zipRight(Effect.never),
      Effect.onInterrupt(() => Effect.sync(() => { demandInterrupted = true })),
    ))
    const outcome = await fixture.run((commands) => Effect.gen(function* () {
      const load = yield* commands.loadModel(Schema.decodeUnknownSync(ModelIdSchema)("model-a:gguf:test")).pipe(Effect.fork)
      yield* Effect.yieldNow()
      expect(Option.isNone(yield* Fiber.poll(load))).toBe(true)
      yield* fixture.publish(loading(2))
      const acknowledged = yield* Fiber.join(load)
      yield* Effect.yieldNow()
      expect(demandInterrupted).toBe(false)
      return acknowledged
    }))

    expect(outcome).toMatchObject({ _tag: "Right", right: {} })
    expect(demands).toEqual(["model-a:gguf:test"])
  })

  it("delivers the warm demand for an already admitted model and acknowledges without waiting", async () => {
    const { ModelIdSchema } = await import("@magnitudedev/acn-protocol")
    const demands: string[] = []
    const fixture = await loadFixture(loading(1), (modelId) => Effect.sync(() => { demands.push(modelId) }).pipe(Effect.zipRight(Effect.never)))
    const outcome = await fixture.run((commands) => commands.loadModel(Schema.decodeUnknownSync(ModelIdSchema)("model-a:gguf:test")).pipe(
      Effect.zipLeft(Effect.yieldNow()),
    ))

    expect(outcome).toMatchObject({ _tag: "Right", right: {} })
    expect(demands).toEqual(["model-a:gguf:test"])
  })

  it("loads the catalog model when a request names the ID its files had as a found model before an upgrade", async () => {
    const { ModelIdSchema } = await import("@magnitudedev/acn-protocol")
    const demands: string[] = []
    const fixture = await loadFixture(loading(1), (modelId) => Effect.sync(() => { demands.push(modelId) }).pipe(Effect.zipRight(Effect.never)))
    const outcome = await fixture.run((commands) => commands.loadModel(Schema.decodeUnknownSync(ModelIdSchema)(UPGRADED_ALIAS)).pipe(
      Effect.zipLeft(Effect.yieldNow()),
    ))
    expect(outcome).toMatchObject({ _tag: "Right", right: {} })
    expect(demands).toEqual(["model-a:gguf:test"])
  })

  it("reports a demand rejected before admission", async () => {
    const { ModelIdSchema } = await import("@magnitudedev/acn-protocol")
    const fixture = await loadFixture({ revision: 1, instances: [] }, () => Effect.fail({
      _tag: "GeneratedClientRemoteError", operationId: "ensureModelInstance", status: 409, headers: {},
      body: { error: { code: "memory_shortage", message: "Not enough memory", type: "model_error", param: Option.none() } },
    }))
    const outcome = await fixture.run((commands) => commands.loadModel(Schema.decodeUnknownSync(ModelIdSchema)("model-a:gguf:test")))

    expect(outcome).toMatchObject({ _tag: "Left", left: { _tag: "LocalModelMutationFailed", code: "memory_shortage" } })
  })

  it("acknowledges a slot load at admission and refreshes the slot", async () => {
    const { PRIMARY_SLOT_ID } = await import("@magnitudedev/acn-protocol")
    const fixture = await loadFixture({ revision: 1, instances: [] }, () => Effect.never)
    const outcome = await fixture.run((commands) => Effect.gen(function* () {
      const load = yield* commands.loadSlot(PRIMARY_SLOT_ID).pipe(Effect.fork)
      yield* Effect.yieldNow()
      yield* fixture.publish(loading(2))
      return yield* Fiber.join(load)
    }))

    expect(outcome).toMatchObject({ _tag: "Right", right: {} })
    expect(fixture.slotsRefreshed).toEqual(["primary"])
  })
})
