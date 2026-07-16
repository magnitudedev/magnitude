import { describe, expect, it } from "vitest"
import { Effect, Ref } from "effect"
import type { MagnitudeConfig } from "@magnitudedev/storage"
import { makeLocalModelConfiguration, reconcileLocalModelSlots, type LocalSlotCandidate } from "./model-configuration"

const candidate = (
  providerModelId: string,
  ownership: "managed" | "external",
  residency: LocalSlotCandidate["residency"],
  availability: LocalSlotCandidate["availability"] = "available",
): LocalSlotCandidate => ({
  providerModelId,
  ownership,
  residency,
  availability,
  productRank: 0,
  externalPriority: 0,
})

describe("LocalModelConfiguration", () => {
  it("keeps cloud slots fixed while an external loaded model takes over local slots", () => {
    const result = reconcileLocalModelSlots({
      models: {
        slots: {
          primary: { providerId: "magnitude", providerModelId: "cloud" },
          secondary: { providerId: "llamacpp", providerModelId: "managed-a" },
        },
      },
    }, [candidate("managed-a", "managed", "loaded"), candidate("external-a", "external", "loaded")])

    expect(result.config.models?.slots).toEqual({
      primary: { providerId: "magnitude", providerModelId: "cloud" },
      secondary: { providerId: "llamacpp", providerModelId: "external-a" },
    })
    expect(result.config.models?.localSlotIntent).toEqual({ primary: "cloud", secondary: "local" })
  })

  it("retains an available unloaded local selection until a loaded candidate appears", () => {
    const current: MagnitudeConfig = {
      models: {
        slots: { primary: { providerId: "llamacpp", providerModelId: "managed-a" } },
        localSlotIntent: { primary: "local" },
      },
    }
    expect(reconcileLocalModelSlots(current, [candidate("managed-a", "managed", "unloaded")]).changed).toBe(false)
  })

  it("falls through disabled selection to per-slot recency without crossing to cloud", () => {
    const result = reconcileLocalModelSlots({
      models: {
        slots: {
          primary: { providerId: "llamacpp", providerModelId: "disabled" },
          secondary: { providerId: "magnitude", providerModelId: "cloud" },
        },
        localSlotIntent: { primary: "local", secondary: "cloud" },
        localModelRecency: { primary: ["disabled", "recent", "older"] },
      },
    }, [
      candidate("disabled", "managed", "unloaded", "disabled"),
      candidate("older", "managed", "unloaded"),
      candidate("recent", "managed", "unloaded"),
    ])

    expect(result.config.models?.slots?.primary?.providerModelId).toBe("recent")
    expect(result.config.models?.slots?.secondary?.providerModelId).toBe("cloud")
  })

  it("clears an unavailable local slot but preserves local intent", () => {
    const result = reconcileLocalModelSlots({
      models: {
        slots: { primary: { providerId: "llamacpp", providerModelId: "gone" } },
        localSlotIntent: { primary: "local" },
      },
    }, [])
    expect(result.config.models?.slots?.primary).toBeUndefined()
    expect(result.config.models?.localSlotIntent?.primary).toBe("local")
  })

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
