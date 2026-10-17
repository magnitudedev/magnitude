import { Effect, Either, Layer, Option, Schema, Stream } from "effect"
import { expect, it } from "vitest"
import { IcnCatalog, IcnCatalogInstallations, IcnClient, IcnInstances } from "@magnitudedev/icn"
import { ModelIdSchema, ModelVersionNotRunnable } from "@magnitudedev/acn-protocol"
import { CatalogModelSyncLive } from "./catalog-model-sync"
import { CustomModelCommands } from "./custom-model-commands"
import { LocalModelRemovals } from "./local-model-removals"
import { LocalModels } from "./local-models"
import { ModelCommands, ModelCommandsLive } from "./model-commands"
import { ModelSlotController } from "./model-slot-controller"

const FOUND = "hf:bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
const ADDED = "hf:owner/repo/model-Q4_K_M.gguf"
const RUNNABLE = "hf:owner/repo/other-Q4_K_M.gguf"
const reasons = [{ _tag: "UnrecognizedTensor", name: "rope_freqs.weight" }] as const

const localModels = {
  state: Effect.succeed({
    models: [
      { _tag: "Custom", modelId: ADDED, servingState: { _tag: "Assessed", assessment: { _tag: "Unsupported", reasons } } },
      { _tag: "Custom", modelId: RUNNABLE, servingState: { _tag: "Assessed", assessment: { _tag: "Fits" } } },
    ],
    unsupported: [{ modelId: Option.some(FOUND), reasons }],
    hidden: [],
  }),
} as never

/** `models load` of a model the engine can't run fails with the engine's reasons and never asks ICN to load it. */
it("refuses to load a model the engine can't run, with its reasons", async () => {
  const demanded: string[] = []
  const load = (model: string) => Effect.gen(function* () {
    const commands = yield* ModelCommands
    return yield* commands.loadModel(Schema.decodeUnknownSync(ModelIdSchema)(model))
  }).pipe(
    Effect.either,
    Effect.provide(ModelCommandsLive.pipe(Layer.provide(CatalogModelSyncLive))),
    Effect.provideService(CustomModelCommands, {} as never),
    Effect.provideService(LocalModels, localModels),
    Effect.provideService(IcnInstances, { get: Effect.succeed({ revision: 0, instances: [] }), changes: Stream.never } as never),
    Effect.provideService(IcnClient, {
      models: {
        ensureModelInstance: ({ payload }: { payload: { modelId: string } }) => Effect.sync(() => {
          demanded.push(payload.modelId)
          return {}
        }),
      },
    } as never),
    Effect.provideService(ModelSlotController, {} as never),
    Effect.provideService(IcnCatalog, {} as never),
    Effect.provideService(IcnCatalogInstallations, {} as never),
    Effect.provideService(LocalModelRemovals, {} as never),
    Effect.scoped,
  )

  for (const model of [FOUND, ADDED]) {
    const result = await Effect.runPromise(load(model))
    expect(Either.isLeft(result)).toBe(true)
    const error = Either.getLeft(result).pipe(Option.getOrThrow)
    expect(error).toBeInstanceOf(ModelVersionNotRunnable)
    expect(error).toMatchObject({ reference: model, verdict: { _tag: "Unsupported", reasons } })
  }
  expect(demanded).toEqual([])

  const runnable = await Effect.runPromise(load(RUNNABLE))
  expect(Either.isRight(runnable)).toBe(true)
  expect(demanded).toEqual([RUNNABLE])
})
