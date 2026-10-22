import { Data, Effect, Layer, Option, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { IcnLocalModels } from "@magnitudedev/icn"
import type { LocalModelsConfiguration } from "@magnitudedev/icn-protocol/schemas"
import { CustomFormModelIdSchema, type LocalModelMutationFailed } from "@magnitudedev/acn-protocol"
import type { CustomModelsConfig } from "@magnitudedev/storage"
import { AcnChanges } from "./changes"
import { CustomModels, type CustomModelsApi } from "./custom-models"
import { CustomModelReconciler, CustomModelReconcilerLive, disappearedCustomModels } from "./custom-model-reconciler"
import { ModelLocations } from "./model-locations"
import { ModelSlotController } from "./model-slot-controller"
import {
  addedId,
  customModelsConfig,
  drafterReference,
  foundId,
  localId,
  makeTestCustomModels,
} from "./custom-models.test-support"

describe("custom model disappearance", () => {
  it("names removed added models and newly hidden models only", () => {
    const previous = customModelsConfig()
    const next = { ...previous, added: {}, hidden: [foundId] }
    expect(disappearedCustomModels(previous, next)).toEqual([addedId, foundId])
    expect(disappearedCustomModels(next, next)).toEqual([])
  })

  it("doesn't take a hidden local model renamed to its current label for a newly hidden one", () => {
    const olderLabel = CustomFormModelIdSchema.make("local:old-name-q4_k_m:0123456789ab")
    const previous = customModelsConfig({ hidden: [olderLabel] })
    expect(disappearedCustomModels(previous, { ...previous, hidden: [localId] })).toEqual([])
  })
})

interface Recorded {
  readonly configured: LocalModelsConfiguration[]
  readonly cleared: string[]
  readonly published: string[]
}

/** Runs `use` against a reconciler over these choices, with ICN and the slots recorded. */
const reconciling = (
  config: CustomModelsConfig,
  use: (reconciler: CustomModelReconciler["Type"], custom: CustomModelsApi) => Effect.Effect<void, LocalModelMutationFailed>,
) => {
  const recorded: Recorded = { configured: [], cleared: [], published: [] }
  return Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const custom = yield* makeTestCustomModels(config)
    const layer = CustomModelReconcilerLive.pipe(Layer.provide(Layer.mergeAll(
      Layer.succeed(CustomModels, custom),
      Layer.succeed(ModelLocations, {
        environment: { platform: "linux", env: {}, home: "/h" },
        resolve: Effect.succeed([{
          kind: "HuggingFaceCache" as const,
          paths: [{ path: "/h/.cache/huggingface/hub", state: "Found" as const }] as const,
          state: "Found" as const,
        }]),
        list: () => Effect.succeed([]),
      }),
      Layer.succeed(IcnLocalModels, {
        configure: (configuration: LocalModelsConfiguration) => Effect.sync(() => { recorded.configured.push(configuration) }),
      } as never),
      Layer.succeed(ModelSlotController, {
        state: Effect.succeed({ slots: {
          primary: { _tag: "ConfiguredLocal", slotId: "primary", selection: { providerId: "local", providerModelId: foundId } },
          secondary: { _tag: "Unassigned", slotId: "secondary" },
        } }),
        updateModelSlot: (slotId: string) => Effect.sync(() => { recorded.cleared.push(slotId) }),
      } as never),
      Layer.succeed(AcnChanges, {
        publish: ({ operation }: { readonly operation: string }) => Effect.sync(() => { recorded.published.push(operation) }),
        stream: Stream.empty,
      } as never),
    )))
    yield* Effect.flatMap(CustomModelReconciler, (reconciler) => use(reconciler, custom)).pipe(Effect.provide(layer))
  }))).then(() => recorded)
}

describe("custom model reconciliation", () => {
  it("pushes config.json's choices converted for ICN, and clears a slot whose model disappears", async () => {
    const recorded = await reconciling(customModelsConfig({ locations: { lmstudio: false }, paths: ["/models"] }), (reconciler, custom) =>
      Effect.gen(function* () {
        yield* reconciler.push
        yield* Effect.sleep("50 millis")
        yield* custom.modify((config) => Effect.succeed([undefined, { ...config, hidden: [foundId] }] as const))
        yield* Effect.sleep("50 millis")
      }))
    const first = recorded.configured[0]!
    expect(first.roots).toEqual([
      { kind: "HuggingFaceCache", path: "/h/.cache/huggingface/hub", enabled: true },
      { kind: "Folder", path: "/models", enabled: true },
    ])
    expect(first.added).toEqual([
      { repository: "owner/model-GGUF", path: "model-Q4_K_M.gguf", revision: "c0ffee", projector: Option.none() },
    ])
    // The engine works out a drafter's method from its file; none is sent.
    expect(first.drafters).toEqual([{ modelId: foundId, drafter: { _tag: "Reference", reference: drafterReference } }])
    expect(first.hidden).toEqual([])
    expect(recorded.configured.at(-1)!.hidden).toEqual([foundId])
    expect(recorded.cleared).toEqual(["primary"])
    expect(recorded.published).toContain("ListModelLocations")
  })

  it("sends no drafter for automatic choice and None for none", async () => {
    const recorded = await reconciling(customModelsConfig({ drafters: { [localId]: "none" } }), (reconciler) => reconciler.push)
    expect(recorded.configured[0]!.drafters).toEqual([{ modelId: localId, drafter: { _tag: "None" } }])
  })

  it("delivers a hidden external copy of a catalog model to ICN at start, as after a restart", async () => {
    const alias = "hf:openbmb/MiniCPM5-1B-GGUF/MiniCPM5-1B-Q4_K_M.gguf"
    const recorded = await reconciling(customModelsConfig({ drafters: {}, hidden: [alias] }), () => Effect.sleep("50 millis"))
    expect(recorded.configured[0]!.hidden).toEqual([alias])
  })
})
