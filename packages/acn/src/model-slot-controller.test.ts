import { Effect, Layer, Option, Schema, Stream } from "effect"
import { describe, expect, it } from "vitest"
import {
  LocalModelsStateSchema,
  type ProviderModelCatalogState,
  type SlotSelection,
} from "@magnitudedev/acn-protocol"
import { IcnInstances } from "@magnitudedev/icn"
import { PROVIDER_ID as LOCAL_PROVIDER_ID } from "@magnitudedev/icn/provider"
import { ReasoningEffortSchema } from "@magnitudedev/sdk"
import { MagnitudeStorage, type MagnitudeStorageShape } from "@magnitudedev/storage"
import { AcnChanges } from "./changes"
import { LocalModels } from "./local-models"
import { LocalProviderOfferings } from "./local-provider-offerings"
import { ModelSelection, type ModelSelectionState } from "./model-selection"
import { ModelSlotController, ModelSlotControllerLive } from "./model-slot-controller"
import { ProviderModelCatalog } from "./provider-model-catalog"

describe("model slot reconciliation", () => {
  it("preserves a local selection while its discovered model is unavailable", async () => {
    const localModels = Schema.validateSync(LocalModelsStateSchema)({
      preparation: {
        discovery: { complete: true, modelsFound: 1 },
        assessment: { complete: true, settledModels: 1, totalModels: 1 },
      },
      models: [{
        _tag: "Discovered",
        modelId: "hf:owner/repository/model.gguf",
        presentation: {
          displayName: "model",
          variantLabel: "GGUF",
          description: "Discovered in the Hugging Face cache",
          license: Option.none(),
          sourceUrls: ["https://huggingface.co/owner/repository"],
        },
        state: {
          _tag: "Unavailable",
          installation: {
            _tag: "Resolved",
            installedBytes: 1,
            primaryPath: "/model.gguf",
            ownership: "ExternalHuggingFace",
          },
          failure: {
            code: "invalid_gguf",
            message: "The selected artifact is invalid",
            retryable: false,
          },
        },
      }],
    })
    const selection: SlotSelection = {
      providerId: LOCAL_PROVIDER_ID,
      providerModelId: localModels.models[0]!.modelId,
      reasoningEffort: ReasoningEffortSchema.make("none"),
    }
    const selected: ModelSelectionState = {
      slots: { primary: Option.some(selection), secondary: Option.none() },
      recentModels: { primary: [], secondary: [] },
      favorites: [],
    }
    const cleared: string[] = []
    const catalogState: ProviderModelCatalogState = {
      _tag: "Ready",
      providers: [{
        providerId: LOCAL_PROVIDER_ID,
        displayName: "Local",
        kind: "Local",
        authentication: "NotRequired",
        availability: { _tag: "Available" },
      }],
      models: [],
    }
    const dependencies = Layer.mergeAll(
      Layer.succeed(ModelSelection, ModelSelection.of({
        get: Effect.succeed(selected),
        changes: Stream.never,
        updateSlot: (slotId) => Effect.sync(() => { cleared.push(slotId) }),
        recordUse: () => Effect.void,
        setFavorite: () => Effect.void,
      })),
      Layer.succeed(LocalModels, LocalModels.of({
        state: Effect.succeed(localModels),
        changes: Stream.never,
        refresh: Effect.void,
      })),
      Layer.succeed(LocalProviderOfferings, LocalProviderOfferings.of({
        ready: Effect.succeed(true),
        list: Effect.succeed([]),
        changes: Stream.never,
        catalog: Effect.succeed([]),
        catalogChanges: Stream.never,
        resolve: () => Effect.die("unused"),
      })),
      Layer.succeed(ProviderModelCatalog, ProviderModelCatalog.of({
        state: Effect.succeed(catalogState),
        changes: Stream.never,
        refresh: () => Effect.void,
      })),
      Layer.succeed(IcnInstances, IcnInstances.of({
        get: Effect.succeed({ revision: 0, instances: [] }),
        changes: Stream.never,
        initialized: Effect.succeed(true),
        refresh: Effect.void,
      })),
      Layer.succeed(AcnChanges, AcnChanges.of({
        publish: () => Effect.void,
        stream: Stream.never,
      })),
      Layer.succeed(MagnitudeStorage, {
        config: {
          getContextLimitPolicy: () => Effect.succeed({ softCapRatio: 0.9, softCapMaxTokens: 200_000 }),
        },
      } as unknown as MagnitudeStorageShape),
    )

    await Effect.runPromise(Effect.scoped(ModelSlotController.pipe(
      Effect.provide(ModelSlotControllerLive.pipe(Layer.provide(dependencies))),
      Effect.flatMap((controller) => controller.state),
    )))

    expect(cleared).toEqual([])
    expect(selected.slots.primary).toEqual(Option.some(selection))
  })
})
