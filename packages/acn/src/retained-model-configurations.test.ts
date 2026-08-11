import { Effect, Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelServingConfigurationIdSchema,
  type ModelServingConfiguration,
} from "@magnitudedev/acn-protocol"
import { ModelStateSchema } from "@magnitudedev/storage"
import { makeRetainedModelConfigurations } from "./retained-model-configurations"
import { makeTestModelState } from "./model-state.test-support"

const configuration = (
  id: string,
  contextLength: number,
): ModelServingConfiguration => ({
  id: ModelServingConfigurationIdSchema.make(id),
  bundle: {
    _tag: "Standalone",
    package: {
      id: "package-a",
      source: { _tag: "Local", path: "/models" },
      files: [],
      relationships: [],
      properties: {
        format: "gguf",
        quantization: "Q4",
        quantizationName: "4-bit",
        architecture: "test",
        maximumContextLength: 65_536,
      },
    },
  },
  profile: { contextLength },
} as unknown as ModelServingConfiguration)

const emptyState = Schema.decodeUnknownSync(ModelStateSchema)({
  configurations: [],
  configurationRecoveryCompleted: false,
})

const makeService = Effect.gen(function* () {
  const state = yield* makeTestModelState(emptyState)
  return { state, service: makeRetainedModelConfigurations(state) }
})

describe("RetainedModelConfigurations", () => {
  it("replaces the retained configuration for a bundle", async () => {
    const retained = await Effect.runPromise(Effect.gen(function* () {
      const { service } = yield* makeService
      yield* service.materialize(configuration("configuration-a", 32_768))
      yield* service.materialize(configuration("configuration-b", 65_536))
      return yield* service.get
    }))
    expect(retained.map(({ id }) => id)).toEqual(["configuration-b"])
  })

  it("does not recover a default beside an authoritative custom profile", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const { state, service } = yield* makeService
      yield* service.materialize(configuration("configuration-custom", 32_768))
      const additions = yield* service.completeRecovery([
        configuration("configuration-default", 65_536),
      ])
      return { additions, state: yield* state.get }
    }))
    expect(result.additions).toEqual([])
    expect(result.state.configurationRecoveryCompleted).toBe(true)
    expect(result.state.configurations.map(({ id }) => id)).toEqual(["configuration-custom"])
  })

  it("removes the exact configuration and its local selection references atomically", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const { state, service } = yield* makeService
      const saved = yield* service.materialize(configuration("configuration-a", 32_768))
      yield* state.update((current) => ({
        ...current,
        slots: {
          ...current.slots,
          primary: Option.some({
            providerId: "local",
            providerModelId: saved.id,
            reasoningEffort: "none",
          } as never),
        },
        favorites: [{ providerId: "local", providerModelId: saved.id } as never],
      }))
      yield* service.remove(saved.id)
      return yield* state.get
    }))
    expect(result.configurations).toEqual([])
    expect(Option.isNone(result.slots.primary)).toBe(true)
    expect(result.favorites).toEqual([])
  })
})
