import { Effect, Layer, Option, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { IcnCandidates, IcnLocalModels } from "@magnitudedev/icn"
import type { LocalModelsConfiguration } from "@magnitudedev/icn-protocol/schemas"
import { AcnChanges } from "./changes"
import { CustomModels } from "./custom-models"
import { CustomModelReconciler, CustomModelReconcilerLive, disappearedCustomModels } from "./custom-model-reconciler"
import { DrafterMethodsLive } from "./drafter-references"
import { ModelLocations } from "./model-locations"
import { ModelSlotController } from "./model-slot-controller"
import {
  addedId,
  customModelsConfig,
  drafterReference,
  foundId,
  importedId,
  makeTestCustomModels,
} from "./custom-models.test-support"

describe("custom model disappearance", () => {
  it("names forgotten declarations and newly hidden models only", () => {
    const previous = customModelsConfig()
    const next = { ...previous, declarations: { [importedId]: previous.declarations[importedId]! }, hidden: [foundId] }
    expect(disappearedCustomModels(previous, next)).toEqual([addedId, foundId])
    expect(disappearedCustomModels(next, next)).toEqual([])
  })
})

describe("custom model reconciliation", () => {
  it("relearns drafter methods, pushes the whole snapshot, and clears a slot whose model disappears", async () => {
    const configured: LocalModelsConfiguration[] = []
    const cleared: string[] = []
    const published: string[] = []
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const custom = yield* makeTestCustomModels(customModelsConfig())
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
        DrafterMethodsLive,
        Layer.succeed(IcnLocalModels, {
          configure: (configuration: LocalModelsConfiguration) => Effect.sync(() => { configured.push(configuration) }),
        } as never),
        Layer.succeed(IcnCandidates, {
          drafters: () => Effect.succeed([{
            category: "CatalogDrafter",
            reference: drafterReference,
            name: "Magnitude DFlash for Qwen3.6 35B-A3B",
            label: "Magnitude DFlash",
            lineage: "Lineage",
            method: Option.some({ _tag: "DSpark" }),
            onThisComputer: false,
            sizeBytes: 1,
            verdict: { _tag: "Bound", method: { _tag: "DSpark" }, memoryBytes: 1 },
          }]),
        } as never),
        Layer.succeed(ModelSlotController, {
          state: Effect.succeed({ slots: {
            primary: { _tag: "ConfiguredLocal", slotId: "primary", selection: { providerId: "local", providerModelId: foundId } },
            secondary: { _tag: "Unassigned", slotId: "secondary" },
          } }),
          updateModelSlot: (slotId: string) => Effect.sync(() => { cleared.push(slotId) }),
        } as never),
        Layer.succeed(AcnChanges, {
          publish: ({ operation }: { readonly operation: string }) => Effect.sync(() => { published.push(operation) }),
          stream: Stream.empty,
        } as never),
      )))
      yield* Effect.gen(function* () {
        const reconciler = yield* CustomModelReconciler
        yield* reconciler.push
        yield* custom.modify((config) => Effect.succeed([undefined, { ...config, hidden: [foundId] }] as const))
        yield* Effect.sleep("50 millis")
      }).pipe(Effect.provide(layer))
    })))
    const first = configured[0]!
    expect(first.roots).toEqual([{ kind: "HuggingFaceCache", path: "/h/.cache/huggingface/hub", enabled: true }])
    expect(first.drafters).toEqual([{
      modelId: foundId,
      drafter: { _tag: "Reference", method: { _tag: "DSpark" }, reference: drafterReference },
    }])
    expect(first.hidden).toEqual([])
    expect(cleared).toEqual(["primary"])
    expect(published).toContain("ListModelLocations")
    expect(configured.length).toBeGreaterThanOrEqual(2)
  })

  it("relearns the drafter method of an added model still downloading from its declared repository file", async () => {
    const configured: LocalModelsConfiguration[] = []
    const requests: string[] = []
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const custom = yield* makeTestCustomModels(customModelsConfig({
        drafters: { [addedId]: { _tag: "Reference", reference: drafterReference } },
      }))
      const layer = CustomModelReconcilerLive.pipe(Layer.provide(Layer.mergeAll(
        Layer.succeed(CustomModels, custom),
        Layer.succeed(ModelLocations, {
          environment: { platform: "linux", env: {}, home: "/h" },
          resolve: Effect.succeed([]),
          list: () => Effect.succeed([]),
        }),
        DrafterMethodsLive,
        Layer.succeed(IcnLocalModels, {
          configure: (configuration: LocalModelsConfiguration) => Effect.sync(() => { configured.push(configuration) }),
        } as never),
        Layer.succeed(IcnCandidates, {
          drafters: (request: { readonly _tag: string }) => Effect.sync(() => {
            requests.push(request._tag)
            return request._tag === "Model" ? [] : [{
              category: "CatalogDrafter",
              reference: drafterReference,
              name: "Magnitude DFlash for Qwen3.6 35B-A3B",
              label: "Magnitude DFlash",
              lineage: "Lineage",
              method: Option.some({ _tag: "DFlash" }),
              onThisComputer: false,
              sizeBytes: 1,
              verdict: { _tag: "Bound", method: { _tag: "DFlash" }, memoryBytes: 1 },
            }]
          }),
        } as never),
        Layer.succeed(ModelSlotController, {
          state: Effect.succeed({ slots: {
            primary: { _tag: "Unassigned", slotId: "primary" },
            secondary: { _tag: "Unassigned", slotId: "secondary" },
          } }),
          updateModelSlot: () => Effect.void,
        } as never),
        Layer.succeed(AcnChanges, { publish: () => Effect.void, stream: Stream.empty } as never),
      )))
      yield* Effect.flatMap(CustomModelReconciler, (reconciler) => reconciler.push).pipe(Effect.provide(layer))
    })))
    expect(requests.slice(0, 2)).toEqual(["Model", "Candidate"])
    expect(configured[0]!.drafters).toEqual([{
      modelId: addedId,
      drafter: { _tag: "Reference", method: { _tag: "DFlash" }, reference: drafterReference },
    }])
  })

  it("delivers a hidden external copy of a catalog model to ICN at start, as after a restart", async () => {
    const alias = "hf:openbmb/MiniCPM5-1B-GGUF/MiniCPM5-1B-Q4_K_M.gguf"
    const configured: LocalModelsConfiguration[] = []
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const custom = yield* makeTestCustomModels(customModelsConfig({ drafters: {}, hidden: [alias] as never }))
      const layer = CustomModelReconcilerLive.pipe(Layer.provide(Layer.mergeAll(
        Layer.succeed(CustomModels, custom),
        Layer.succeed(ModelLocations, {
          environment: { platform: "linux", env: {}, home: "/h" },
          resolve: Effect.succeed([]),
          list: () => Effect.succeed([]),
        }),
        DrafterMethodsLive,
        Layer.succeed(IcnLocalModels, {
          configure: (configuration: LocalModelsConfiguration) => Effect.sync(() => { configured.push(configuration) }),
        } as never),
        Layer.succeed(IcnCandidates, { drafters: () => Effect.succeed([]) } as never),
        Layer.succeed(ModelSlotController, {
          state: Effect.succeed({ slots: {
            primary: { _tag: "Unassigned", slotId: "primary" },
            secondary: { _tag: "Unassigned", slotId: "secondary" },
          } }),
          updateModelSlot: () => Effect.void,
        } as never),
        Layer.succeed(AcnChanges, { publish: () => Effect.void, stream: Stream.empty } as never),
      )))
      yield* Effect.flatMap(CustomModelReconciler, () => Effect.sleep("50 millis")).pipe(Effect.provide(layer))
    })))
    expect(configured[0]!.hidden).toEqual([alias])
  })
})
