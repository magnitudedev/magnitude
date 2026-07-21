import { describe, expect, it } from "vitest"
import { Effect, Option, Stream } from "effect"
import { type IcnApiClient as IcnApiClientService, Generated } from "@magnitudedev/icn"
import type { LocalModelConfigurationApi } from "./model-configuration"
import { reconcileSelectedServingConfiguration } from "./serving-configuration"

const model = (contextLength: number): Generated.Model => ({
  id: "model-1",
  source: { type: "hugging_face", repository: "test/repo", revision: "commit" },
  location: { type: "file", component: { path: "model.gguf", role: "weights" } },
  serving_configuration: Option.some({
    profile: { context_length: contextLength, parallel_sequences: 1 },
  }),
} as unknown as Generated.Model)

const configuration: LocalModelConfigurationApi = {
  get: Effect.succeed({
    selectedProfile: {
      configurationId: "catalog:p1:ctx200000",
      catalogModelId: "catalog",
      contextTokens: 200_000,
      providerModelId: "model-1",
    },
  }),
  getModels: Effect.succeed({}),
  selectProfile: () => Effect.void,
  updateSlots: () => Effect.void,
  recordUse: () => Effect.void,
  revision: Effect.succeed(0),
  changes: Stream.empty,
}

describe("selected local serving configuration", () => {
  it("reconciles the durable selected context before exposing inventory", async () => {
    let configureCalls = 0
    const client = {
      models: {
        configureModelServing: ({ payload }: { payload: { context_length: number } }) => Effect.sync(() => {
          configureCalls++
          return model(payload.context_length)
        }),
      },
    } as unknown as IcnApiClientService

    const result = await Effect.runPromise(
      reconcileSelectedServingConfiguration(client, configuration, [model(4_096)]),
    )

    expect(configureCalls).toBe(1)
    expect(Option.getOrThrow(result[0]!.serving_configuration)!.profile.context_length).toBe(200_000)
  })

  it("does not write when ICN already has the selected profile", async () => {
    let configureCalls = 0
    const client = {
      models: {
        configureModelServing: () => Effect.sync(() => {
          configureCalls++
          return model(200_000)
        }),
      },
    } as unknown as IcnApiClientService

    await Effect.runPromise(
      reconcileSelectedServingConfiguration(client, configuration, [model(200_000)]),
    )

    expect(configureCalls).toBe(0)
  })
})
