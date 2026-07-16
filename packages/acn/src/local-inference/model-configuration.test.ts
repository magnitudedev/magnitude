import { describe, expect, it } from "vitest"
import { Effect, Ref } from "effect"
import type { MagnitudeConfig } from "@magnitudedev/storage"
import { makeLocalModelConfiguration, reconcileLocalModelSlots, type LocalSlotCandidate } from "./model-configuration"

const candidate = (
  providerModelId: string,
  ownership: "managed" | "external",
  residency: "loaded" | "sleeping" | "unloaded" | "loading" | "failed" | "unknown",
  availability: LocalSlotCandidate["availability"] = "available",
): LocalSlotCandidate => ({
  providerModelId,
  availability,
  externalLoaded: ownership === "external" && residency === "loaded",
  managedLoaded: ownership === "managed" && residency === "loaded",
  sleeping: residency === "sleeping",
  managedRestorable: ownership === "managed" && residency !== "failed",
  demandLoading: ownership === "managed" && residency === "loading",
  productRank: 0,
  externalPriority: 0,
})

const reconcile = (
  current: MagnitudeConfig,
  candidates: readonly LocalSlotCandidate[],
  authoritativeModelIds: ReadonlySet<string> = new Set(candidates.map((item) => item.providerModelId)),
) => reconcileLocalModelSlots(current, { authoritativeModelIds, candidates })

describe("LocalModelConfiguration", () => {
  it("keeps cloud slots fixed and does not replace a running local selection", () => {
    const result = reconcile({
      models: {
        slots: {
          primary: { providerId: "magnitude", providerModelId: "cloud" },
          secondary: { providerId: "llamacpp", providerModelId: "managed-a" },
        },
      },
    }, [candidate("managed-a", "managed", "loaded"), candidate("external-a", "external", "loaded")])

    expect(result.config.models?.slots).toEqual({
      primary: { providerId: "magnitude", providerModelId: "cloud" },
      secondary: { providerId: "llamacpp", providerModelId: "managed-a" },
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
    expect(reconcile(current, [candidate("managed-a", "managed", "unloaded")]).changed).toBe(false)
  })

  it("replaces an unloaded undemanded managed selection when an external model starts", () => {
    const current: MagnitudeConfig = {
      models: {
        slots: { primary: { providerId: "llamacpp", providerModelId: "managed-a" } },
        localSlotIntent: { primary: "local" },
      },
    }
    const result = reconcile(current, [
      candidate("managed-a", "managed", "unloaded"),
      candidate("external-b", "external", "loaded"),
    ])
    expect(result.config.models?.slots?.primary?.providerModelId).toBe("external-b")
  })

  it("protects a selected model while its demand load is active", () => {
    const current: MagnitudeConfig = {
      models: {
        slots: { primary: { providerId: "llamacpp", providerModelId: "managed-a" } },
        localSlotIntent: { primary: "local" },
      },
    }
    const loading = { ...candidate("managed-a", "managed", "loading"), managedLoaded: false, demandLoading: true }
    const result = reconcile(current, [loading, candidate("external-b", "external", "loaded")])
    expect(result.changed).toBe(false)
    expect(result.config.models?.slots?.primary?.providerModelId).toBe("managed-a")
  })

  it("selects the next candidate after an external-only selection stops", () => {
    const current: MagnitudeConfig = {
      models: {
        slots: { primary: { providerId: "llamacpp", providerModelId: "external-a" } },
        localSlotIntent: { primary: "local" },
      },
    }
    const result = reconcile(current, [
      candidate("external-a", "external", "failed", "disabled"),
      candidate("managed-b", "managed", "unloaded"),
    ])
    expect(result.config.models?.slots?.primary?.providerModelId).toBe("managed-b")
  })

  it("falls through disabled selection to per-slot recency without crossing to cloud", () => {
    const result = reconcile({
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
    const result = reconcile({
      models: {
        slots: { primary: { providerId: "llamacpp", providerModelId: "gone" } },
        localSlotIntent: { primary: "local" },
      },
    }, [])
    expect(result.config.models?.slots?.primary).toBeUndefined()
    expect(result.config.models?.localSlotIntent?.primary).toBe("local")
  })

  it("uses authoritative existence rather than provider-model ID syntax", () => {
    const existing = reconcile({
      models: {
        slots: { primary: { providerId: "llamacpp", providerModelId: "arbitrary-id" } },
        localSlotIntent: { primary: "local" },
      },
    }, [candidate("arbitrary-id", "managed", "unloaded")], new Set(["arbitrary-id"]))
    expect(existing.changed).toBe(false)

    const nonexistentHash = `lmp_${"a".repeat(64)}`
    const missing = reconcile({
      models: {
        slots: { primary: { providerId: "llamacpp", providerModelId: nonexistentHash } },
        localSlotIntent: { primary: "local" },
      },
    }, [], new Set())
    expect(missing.config.models?.slots?.primary).toBeUndefined()
  })

  it("drops every unresolved local model reference in the same reconciliation", () => {
    const result = reconcile({
      localInference: {
        usage: { localModelRole: "main", sessionConcurrency: "one" },
        binding: {
          _tag: "Managed",
          selectionId: "missing",
          artifactId: "missing",
          providerModelId: "missing",
          contextTokens: 100_000,
          parallelSlots: 1,
        },
      },
      models: {
        slots: { primary: { providerId: "llamacpp", providerModelId: "missing" } },
        localSlotIntent: { primary: "local" },
        localModelRecency: { primary: ["missing", "available"], secondary: ["missing"] },
      },
    }, [
      candidate("missing", "external", "failed", "disabled"),
      candidate("available", "managed", "unloaded"),
    ], new Set(["available"]))

    expect(result.changed).toBe(true)
    expect(result.config.localInference).toEqual({
      usage: { localModelRole: "main", sessionConcurrency: "one" },
    })
    expect(result.config.models).toEqual({
      slots: { primary: { providerId: "llamacpp", providerModelId: "available" } },
      localSlotIntent: { primary: "local" },
      localModelRecency: { primary: ["available"] },
    })
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
        getModelConfig: () => Ref.get(state).pipe(Effect.map((config) => config.models ?? null)),
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
