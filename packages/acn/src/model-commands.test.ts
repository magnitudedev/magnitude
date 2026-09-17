import { Option } from "effect"
import { describe, expect, it } from "vitest"
import { modelCommandFailure } from "./model-commands"

describe("modelCommandFailure", () => {
  it("preserves the server's structured remote failure details", () => {
    const failure = modelCommandFailure("load_model", {
      _tag: "GeneratedClientRemoteError",
      operationId: "ensureModelInstance",
      status: 409,
      headers: {},
      body: {
        error: {
          code: "low_memory",
          message: "Not enough memory to load the selected model",
          type: "model_error",
          param: Option.none(),
        },
      },
    } as never)

    expect(failure).toMatchObject({
      _tag: "LocalModelMutationFailed",
      code: "low_memory",
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
    Effect.provide(ModelCommandsLive),
    Effect.provideService(IcnInstances, { get: Effect.succeed(snapshot) } as never),
    Effect.provideService(IcnClient, { models: { stopModelInstance: ({ path }: { path: { instance_id: string } }) => Effect.sync(() => { stopped.push(path.instance_id); return {} }) } } as never),
    Effect.provideService(ModelSlotController, { state: Effect.succeed({ slots: { primary: { _tag: "ConfiguredLocal", selection: { providerModelId: "model-a:gguf:test" } } } }) } as never),
    Effect.provideService(IcnCatalog, {} as never),
    Effect.provideService(IcnCatalogInstallations, {} as never),
    Effect.provideService(LocalModelRemovals, {} as never),
  ))
  expect(stopped).toEqual(["retained-instance", "retained-instance"])
})
