import { describe, expect, it } from "vitest"
import { Option } from "effect"
import type { ModelSlotsState } from "@magnitudedev/sdk"
import { CLI_EXIT_OBSERVATION_FALLBACK, deriveCliExitNotice } from "./cli-exit-notice"

const state = (lifecycle: "Ready" | "Loading" | "Stopped", duplicate = false): ModelSlotsState => {
  const local = {
    _tag: "ConfiguredLocal",
    slotId: "primary",
    selection: {
      providerId: "local",
      providerModelId: "local:test",
      reasoningEffort: Option.none(),
    },
    descriptor: {
      providerId: "local",
      providerModelId: "local:test",
      displayName: "Qwen Test",
      variantLabel: Option.none(),
    },
    availability: { _tag: "Available" },
    instance: Option.some({
      id: "instance-1",
      configurationId: "configuration-1",
      lifecycle:
        lifecycle === "Ready"
          ? { _tag: "Ready", allocation: {} }
          : lifecycle === "Loading"
          ? {
              _tag: "Loading",
              stage: "loading",
              progress: Option.none(),
              plannedAllocation: Option.none(),
            }
          : { _tag: "Stopped", reason: "user_stop" },
    }),
    actions: lifecycle === "Stopped" ? ["Load"] : ["Stop"],
  }
  return {
    slots: {
      primary: local,
      secondary: duplicate ? { ...local, slotId: "secondary" } : { _tag: "Unassigned", slotId: "secondary" },
    },
    recentModels: { primary: [], secondary: [] },
    favoriteModels: [],
  } as unknown as ModelSlotsState
}

const withDistinctSecondary = (modelSlots: ModelSlotsState): ModelSlotsState => {
  const primary = modelSlots.slots.primary
  if (primary._tag !== "ConfiguredLocal") throw new Error("test requires a local primary slot")
  return {
    ...modelSlots,
    slots: {
      ...modelSlots.slots,
      secondary: {
        ...primary,
        slotId: "secondary",
        descriptor: {
          ...primary.descriptor,
          providerModelId: "local:second",
          displayName: "Llama Test",
        },
        instance: Option.some({
          id: "instance-2",
          configurationId: "configuration-2",
          lifecycle: {
            _tag: "Loading",
            stage: "loading",
            progress: Option.none(),
            plannedAllocation: Option.none(),
          },
        }),
      },
    },
  } as ModelSlotsState
}

const notice = (modelSlots: ModelSlotsState, connectedClientCount: number) =>
  Option.getOrUndefined(deriveCliExitNotice(Option.some({ modelSlots, connectedClientCount })))

describe("deriveCliExitNotice", () => {
  it("reports a fresh ten-minute boundary for the final client", () => {
    expect(notice(state("Ready"), 0)).toBe(
      "Qwen Test is still running and will stop automatically after 10 minutes.\n" +
        "Run `magnitude stop` to stop the current daemon and release the model now."
    )
  })

  it("uses singular and plural remaining-client copy", () => {
    expect(notice(state("Ready"), 1)).toContain("1 other client is connected")
    expect(notice(state("Loading"), 2)).toContain("is still loading. 2 other clients are connected")
  })

  it("deduplicates the same projected model instance", () => {
    expect(notice(state("Loading", true), 0)).toBe(
      "Qwen Test is still loading. When idle, it will stop automatically after 10 minutes.\n" +
        "Run `magnitude stop` to stop the current daemon and release the model now."
    )
  })

  it("names every distinct resident model instead of falling back", () => {
    expect(notice(withDistinctSecondary(state("Ready")), 0)).toContain(
      "Qwen Test (running) and Llama Test (loading) are still active"
    )
  })

  it("omits the notice for inactive model state", () => {
    expect(
      Option.isNone(
        deriveCliExitNotice(
          Option.some({
            modelSlots: state("Stopped"),
            connectedClientCount: 0,
          })
        )
      )
    ).toBe(true)
  })

  it("uses bounded fallback copy when observation failed", () => {
    expect(Option.getOrUndefined(deriveCliExitNotice(Option.none()))).toBe(CLI_EXIT_OBSERVATION_FALLBACK)
  })
})
