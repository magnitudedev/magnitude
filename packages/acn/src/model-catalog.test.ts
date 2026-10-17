import { Effect, Layer, Option, PubSub, Ref, Stream } from "effect"
import { describe, expect, it } from "vitest"
import {
  PRIMARY_SLOT_ID,
  type LocalModelsState,
  type ProviderModelCatalogEntry,
  type ProviderModelCatalogState,
} from "@magnitudedev/acn-protocol"
import { ProviderIdSchema, ProviderModelIdSchema } from "@magnitudedev/providers/client"
import { PROVIDER_ID as LOCAL_PROVIDER_ID } from "@magnitudedev/icn/provider"
import { Models } from "@magnitudedev/acn-protocol"
import { AcnChanges, type AcnChangesApi } from "./changes"
import { LocalModels } from "./local-models"
import { ModelCatalog, ModelCatalogLive, projectModelCatalog } from "./model-catalog"
import { ProviderModelCatalog } from "./provider-model-catalog"

const remoteProviderId = ProviderIdSchema.make("remote-test")

const offering = (
  providerId: typeof remoteProviderId,
  providerModelId: string,
): ProviderModelCatalogEntry => ({
  providerId,
  providerModelId: ProviderModelIdSchema.make(providerModelId),
  modelFamilyId: Option.none(),
  displayName: providerModelId,
  variantLabel: Option.none(),
  supportedSlots: [PRIMARY_SLOT_ID],
  contextWindow: 8_192,
  maxOutputTokens: 1_024,
  memory: Option.none(),
  capabilities: {
    vision: false,
    tools: true,
    structuredOutput: true,
    reasoning: { supported: false, efforts: [], defaultEffort: Option.none() },
  },
  availability: { _tag: "Available" },
  pricing: Option.none(),
})

const local = (overrides: Partial<LocalModelsState> = {}): LocalModelsState => ({
  preparation: {
    discovery: { complete: true, modelsFound: 0 },
    assessment: { complete: true, settledModels: 0, totalModels: 0 },
  },
  models: [],
  unsupported: [],
      hidden: [],
      ...overrides,
})

describe("ACN model catalog projection", () => {
  it("keeps local model state visible while remote provider discovery initializes", () => {
    expect(projectModelCatalog({ _tag: "Loading" }, local())).toMatchObject({
      _tag: "Refreshing",
      providers: [],
      models: [],
      localModelPreparation: {
        discovery: { complete: true, modelsFound: 0 },
        assessment: { complete: true, settledModels: 0, totalModels: 0 },
      },
    })
  })

  it("publishes remote rows once and does not duplicate native local offerings", () => {
    const providers: ProviderModelCatalogState = {
      _tag: "Ready",
      providers: [
        {
          providerId: remoteProviderId,
          displayName: "Remote",
          kind: "Hosted",
          authentication: "Authenticated",
          availability: { _tag: "Available" },
        },
        {
          providerId: LOCAL_PROVIDER_ID,
          displayName: "Local",
          kind: "Local",
          authentication: "NotRequired",
          availability: { _tag: "Available" },
        },
      ],
      models: [
        offering(remoteProviderId, "remote-model"),
        offering(LOCAL_PROVIDER_ID, "local-model"),
      ],
    }

    const result = projectModelCatalog(providers, local())

    expect(result._tag).toBe("Ready")
    if (result._tag !== "Ready") return
    expect(result.models).toEqual([{
      _tag: "Remote",
      offering: providers.models[0],
    }])
  })

  it("retains valid rows while local reconciliation is incomplete", () => {
    const providers: ProviderModelCatalogState = {
      _tag: "Ready",
      providers: [{
        providerId: remoteProviderId,
        displayName: "Remote",
        kind: "Hosted",
        authentication: "Authenticated",
        availability: { _tag: "Available" },
      }],
      models: [offering(remoteProviderId, "remote-model")],
    }
    const result = projectModelCatalog(providers, local({
      preparation: {
        discovery: { complete: false, modelsFound: 0 },
        assessment: { complete: false, settledModels: 0, totalModels: 0 },
      },
    }))

    expect(result._tag).toBe("Refreshing")
    if (result._tag !== "Refreshing") return
    expect(result.models).toHaveLength(1)
    expect(result.failures).toEqual([])
  })
})

describe("model catalog changes", () => {
  it("asks clients to reread model locations after every local model change", async () => {
    const operations = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const localChanges = yield* PubSub.unbounded<void>()
      const published = yield* Ref.make<readonly Parameters<AcnChangesApi["publish"]>[0][]>([])
      const layer = ModelCatalogLive.pipe(Layer.provide(Layer.mergeAll(
        Layer.succeed(ProviderModelCatalog, ProviderModelCatalog.of({
          state: Effect.succeed({ providers: [], models: [] } as unknown as ProviderModelCatalogState),
          changes: Stream.never,
          refresh: () => Effect.void,
        })),
        Layer.succeed(LocalModels, LocalModels.of({
          state: Effect.succeed(local()),
          changes: Stream.fromPubSub(localChanges),
          refresh: Effect.void,
        })),
        Layer.succeed(AcnChanges, AcnChanges.of({
          publish: (change) => Ref.update(published, (all) => [...all, change]),
          stream: Stream.never,
        })),
      )))
      yield* Layer.build(layer)
      yield* Effect.sleep("20 millis")
      yield* PubSub.publish(localChanges, undefined)
      yield* Effect.sleep("20 millis")
      return (yield* Ref.get(published)).map((change) => change.operation)
    })))
    expect(operations).toContain(Models.listModelLocations._tag)
    expect(operations).toContain(Models.getCatalog._tag)
  })
})
