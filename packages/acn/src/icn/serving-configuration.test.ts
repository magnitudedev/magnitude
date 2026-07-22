import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import { describe, expect, it } from "vitest"
import { Effect, Option, Ref, Schema, Stream } from "effect"
import {
  Generated,
  IcnClient,
  IcnInventory,
} from "@magnitudedev/icn"
import { makeIcnApiClient } from "../../../icn/src/generated/client"
import type { MagnitudeConfig, SelectedLocalModelProfile } from "@magnitudedev/storage"
import {
  LocalModelConfiguration,
  makeLocalModelConfiguration,
  type LocalModelConfigurationApi,
} from "../model-configuration"
import { reconcileSelectedServingConfiguration } from "./serving-configuration"

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

const provideServices = <A, E, R>(
  effect: Effect.Effect<A, E, R>,
  current: Generated.Model,
  configuration: LocalModelConfigurationApi,
  configureCalls: Ref.Ref<number>,
) => Effect.gen(function* () {
    const snapshot = {
      revision: 0,
      state: { object: "list", data: [current] },
    } as const
    const encoded = yield* Schema.encode(Generated.Model)(model(200_000))
    const http = HttpClient.make((request) => Ref.update(configureCalls, (count) => count + 1).pipe(
      Effect.as(HttpClientResponse.fromWeb(request, new Response(JSON.stringify(encoded), {
        status: 200,
        headers: { "content-type": "application/json" },
      }))),
    ))
    const client = yield* makeIcnApiClient({ baseUrl: "http://icn.test" }).pipe(
      Effect.provideService(HttpClient.HttpClient, http),
    )
    const inventory = IcnInventory.of({
      get: Effect.succeed(snapshot),
      changes: Stream.succeed(snapshot),
      refresh: Effect.void,
    })
    return yield* effect.pipe(
      Effect.provideService(IcnClient, client),
      Effect.provideService(IcnInventory, inventory),
      Effect.provideService(LocalModelConfiguration, configuration),
    )
  })

describe("selected local serving configuration", () => {
  it("reconciles the durable selected context before exposing inventory", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const configureCalls = yield* Ref.make(0)
      const configuration = yield* makeConfiguration(selectedProfile)
      const models = yield* provideServices(
        reconcileSelectedServingConfiguration(),
        model(4_096),
        configuration,
        configureCalls,
      )
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
      yield* provideServices(
        reconcileSelectedServingConfiguration(),
        model(200_000),
        configuration,
        calls,
      )
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
      yield* provideServices(
        reconcileSelectedServingConfiguration(),
        model(4_096),
        configuration,
        calls,
      )
      return yield* Ref.get(calls)
    }))

    expect(configureCalls).toBe(0)
  })
})
