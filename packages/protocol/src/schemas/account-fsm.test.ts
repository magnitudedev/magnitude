import { describe, expect, it } from "vitest"
import {
  ModelCatalogLifecycle,
  ModelCatalogLoading,
  ModelCatalogUnavailable,
  ModelSlotsLifecycle,
  ModelSlotsLoading,
  SlotUnassigned,
  ProviderCatalogUnavailable,
} from "./account"
import { ProviderIdSchema } from "@magnitudedev/ai"

describe("model catalog lifecycle", () => {
  it("moves through loading, ready, refreshing, and degraded", () => {
    const loading = new ModelCatalogLoading({})
    const ready = ModelCatalogLifecycle.transition(loading, "ready", { models: [], providers: [] })
    const refreshing = ModelCatalogLifecycle.transition(ready, "refreshing", { failures: [] })
    const failure = new ProviderCatalogUnavailable({
      providerId: ProviderIdSchema.make("magnitude"),
      message: "offline",
    })
    const degraded = ModelCatalogLifecycle.transition(refreshing, "degraded", {
      models: [],
      providers: [],
      failures: [failure],
    })

    expect([loading._tag, ready._tag, refreshing._tag, degraded._tag]).toEqual([
      "loading",
      "ready",
      "refreshing",
      "degraded",
    ])
  })

  it("rejects transitions that bypass refreshing", () => {
    const ready = ModelCatalogLifecycle.transition(new ModelCatalogLoading({}), "ready", { models: [], providers: [] })
    expect(() => {
      // @ts-expect-error exercising runtime validation for an illegal transition
      ModelCatalogLifecycle.transition(ready, "degraded", { models: [], providers: [], failures: [] })
    }).toThrow("Invalid FSM transition")
  })

  it("represents retrying an unavailable catalog as refreshing", () => {
    const unavailable = new ModelCatalogUnavailable({ providers: [], failures: [] })
    const refreshing = ModelCatalogLifecycle.transition(unavailable, "refreshing", { models: [] })

    expect(refreshing._tag).toBe("refreshing")
  })
})

describe("model slots lifecycle", () => {
  it("requires the same explicit refresh boundary", () => {
    const ready = ModelSlotsLifecycle.transition(new ModelSlotsLoading({}), "ready", {
      slots: {
        primary: new SlotUnassigned({ slotId: "primary", reason: "no_candidate" }),
        secondary: new SlotUnassigned({ slotId: "secondary", reason: "no_candidate" }),
      },
      config: { slots: {}, localSlotIntent: {} },
    })
    const refreshing = ModelSlotsLifecycle.transition(ready, "refreshing", { failures: [] })

    expect(refreshing._tag).toBe("refreshing")
  })
})
