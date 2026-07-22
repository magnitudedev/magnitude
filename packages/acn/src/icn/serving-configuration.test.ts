import { describe, expect, it } from "vitest"
import { Effect, Option, Ref, Schema } from "effect"
import { Generated } from "@magnitudedev/icn"
import type { MagnitudeConfig, SelectedLocalModelProfile } from "@magnitudedev/storage"
import { makeLocalModelConfiguration } from "../model-configuration"
import {
  reconcileSelectedServingConfiguration,
  type ServingConfigurationInventory,
} from "./serving-configuration"

const model = (contextLength: number): Generated.Model => Schema.decodeUnknownSync(Generated.Model)({
  id: "model-1",
  object: "model",
  owned_by: "local",
  source: {
    type: "hugging_face",
    repository: "test/repo",
    requested_revision: "main",
    commit: "commit",
  },
  location: {
    type: "file",
    path: "model.gguf",
    integrity: { type: "unverified", reason: "test fixture" },
    component: {
      path: "model.gguf",
      role: "weights",
      size_bytes: 1,
      content: { type: "unknown" },
    },
  },
  availability: { type: "available", ready_at: 1 },
  residency: { type: "not_resident" },
  hardware: { type: "not_assessed", reason: "test fixture" },
  properties: { type: "pending" },
  serving_configuration: {
    profile: { context_length: contextLength, parallel_sequences: 1 },
  },
})

const makeConfiguration = (selectedProfile: SelectedLocalModelProfile) => Effect.gen(function* () {
  const stored = yield* Ref.make<MagnitudeConfig>({ localInference: { selectedProfile } })
  return yield* makeLocalModelConfiguration({
    getLocalInferenceConfig: () => Ref.get(stored).pipe(Effect.map((config) => config.localInference ?? null)),
    getModelConfig: () => Ref.get(stored).pipe(Effect.map((config) => config.models ?? null)),
    update: (update) => Ref.updateAndGet(stored, update),
  })
})

const selectedProfile: SelectedLocalModelProfile = {
  configurationId: "catalog:p1:ctx200000",
  catalogModelId: "catalog",
  contextTokens: 200_000,
  providerModelId: "model-1",
}

const makeInventory = (
  current: Generated.Model,
  configure: ServingConfigurationInventory["configureModelServing"],
): ServingConfigurationInventory => ({
  get: Effect.succeed({
    revision: 0,
    state: { object: "list", data: [current] },
  }),
  configureModelServing: configure,
})

describe("selected local serving configuration", () => {
  it("reconciles the durable selected context before exposing inventory", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const configureCalls = yield* Ref.make(0)
      const configuration = yield* makeConfiguration(selectedProfile)
      const inventory = makeInventory(
        model(4_096),
        ({ payload }) => Ref.update(configureCalls, (count) => count + 1).pipe(
          Effect.as(model(payload.context_length)),
        ),
      )
      const models = yield* reconcileSelectedServingConfiguration(inventory, configuration)
      return { models, configureCalls: yield* Ref.get(configureCalls) }
    }))

    expect(result.configureCalls).toBe(1)
    const first = result.models.at(0)
    expect(first).toBeDefined()
    if (!first) throw new Error("configured inventory omitted its model")
    const serving = Option.getOrThrow(Option.filter(
      first.serving_configuration,
      (value): value is NonNullable<typeof value> => value !== null,
    ))
    expect(serving.profile.context_length).toBe(200_000)
  })

  it("does not write when ICN already has the selected profile", async () => {
    const configureCalls = await Effect.runPromise(Effect.gen(function* () {
      const calls = yield* Ref.make(0)
      const configuration = yield* makeConfiguration(selectedProfile)
      const inventory = makeInventory(
        model(200_000),
        () => Ref.update(calls, (count) => count + 1).pipe(Effect.as(model(200_000))),
      )
      yield* reconcileSelectedServingConfiguration(inventory, configuration)
      return yield* Ref.get(calls)
    }))

    expect(configureCalls).toBe(0)
  })

  it("does not infer an ICN model from an incomplete selection", async () => {
    const configureCalls = await Effect.runPromise(Effect.gen(function* () {
      const calls = yield* Ref.make(0)
      const configuration = yield* makeConfiguration({
        configurationId: "catalog:p1:ctx200000",
        catalogModelId: "catalog",
        contextTokens: 200_000,
      })
      const inventory = makeInventory(
        model(4_096),
        () => Ref.update(calls, (count) => count + 1).pipe(Effect.as(model(200_000))),
      )
      yield* reconcileSelectedServingConfiguration(inventory, configuration)
      return yield* Ref.get(calls)
    }))

    expect(configureCalls).toBe(0)
  })
})
