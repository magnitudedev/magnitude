import { describe, expect, it } from "vitest"
import { Effect, Ref } from "effect"
import type { MagnitudeConfig } from "@magnitudedev/storage"
import { makeLocalModelConfiguration } from "./model-configuration"

describe("LocalModelConfiguration", () => {
  it("atomically commits and clears the binding with only the local slot", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const state = yield* Ref.make<MagnitudeConfig>({
        localInference: {
          usage: { localModelRole: "main", sessionConcurrency: "one" },
        },
        models: {
          slots: {
            secondary: { providerId: "magnitude", providerModelId: "cloud-model" },
          },
        },
      })
      const updateCount = yield* Ref.make(0)
      const storage = {
        getLocalInferenceConfig: () => Ref.get(state).pipe(
          Effect.map((config) => config.localInference ?? null),
        ),
        updateModelConfig: () => Effect.void,
        update: (f: (config: MagnitudeConfig) => MagnitudeConfig) => Ref.modify(
          state,
          (current) => {
            const next = f(current)
            return [next, next]
          },
        ).pipe(Effect.tap(() => Ref.update(updateCount, (count) => count + 1))),
      }
      const configuration = yield* makeLocalModelConfiguration(storage)
      yield* configuration.activateLocal({
        _tag: "Managed",
        selectionId: "selection",
        artifactId: "artifact",
        providerModelId: "local:model",
        contextTokens: 100_000,
        parallelSlots: 1,
      })
      const activated = yield* Ref.get(state)
      const updatesAfterActivation = yield* Ref.get(updateCount)
      yield* configuration.disableLocal
      return {
        activated,
        disabled: yield* Ref.get(state),
        updatesAfterActivation,
        totalUpdates: yield* Ref.get(updateCount),
      }
    }))

    expect(result.updatesAfterActivation).toBe(1)
    expect(result.activated.localInference?.binding).toMatchObject({
      _tag: "Managed",
      artifactId: "artifact",
      providerModelId: "local:model",
    })
    expect(result.activated.models?.slots).toEqual({
      primary: { providerId: "llamacpp", providerModelId: "local:model" },
      secondary: { providerId: "magnitude", providerModelId: "cloud-model" },
    })

    expect(result.totalUpdates).toBe(2)
    expect(result.disabled.localInference).toEqual({
      usage: { localModelRole: "main", sessionConcurrency: "one" },
    })
    expect(result.disabled.models?.slots).toEqual({
      secondary: { providerId: "magnitude", providerModelId: "cloud-model" },
    })
  })
})
