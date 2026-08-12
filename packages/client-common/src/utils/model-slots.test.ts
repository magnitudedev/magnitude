import { Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelInstanceIdSchema,
  ModelServingConfigurationIdSchema,
  ModelSlotConfiguredLocal,
  ModelSlotUnassigned,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  SECONDARY_SLOT_ID,
} from "@magnitudedev/sdk"
import {
  deriveLocalModelLoadActivity,
  isModelSlotConfigured,
  modelSlotInstanceId,
  modelSlotResidentAllocation,
} from "./model-slots"

const selection = {
  providerId: ProviderIdSchema.make("local"),
  providerModelId: ProviderModelIdSchema.make("configuration"),
  reasoningEffort: ReasoningEffortSchema.make("none"),
}
const descriptor = {
  providerId: selection.providerId,
  providerModelId: selection.providerModelId,
  displayName: "Local model",
  variantLabel: Option.none(),
}
const instanceId = ModelInstanceIdSchema.make("instance")
const configurationId = ModelServingConfigurationIdSchema.make("configuration")
const allocation = {
  contextWindowTokens: 4096,
  parallelSequences: 2,
  physicalContextTokens: 8192,
  memoryDomains: [],
}
const configured = (lifecycle: {
  readonly _tag: "Loading"
  readonly stage: "loading"
  readonly progress: Option.Option<number>
  readonly plannedAllocation: Option.Option<never>
} | {
  readonly _tag: "Ready"
  readonly allocation: typeof allocation
}) => new ModelSlotConfiguredLocal({
  slotId: PRIMARY_SLOT_ID,
  selection,
  descriptor,
  availability: { _tag: "Available" },
  instance: Option.some({ id: instanceId, configurationId, lifecycle }),
  actions: lifecycle._tag === "Ready" || lifecycle._tag === "Loading" ? ["Stop"] : [],
})

describe("canonical model-slot helpers", () => {
  it("preserves configuration independently of physical lifecycle", () => {
    expect(isModelSlotConfigured(new ModelSlotUnassigned({ slotId: PRIMARY_SLOT_ID }))).toBe(false)
    expect(isModelSlotConfigured(configured({
      _tag: "Loading",
      stage: "loading",
      progress: Option.some(0.4),
      plannedAllocation: Option.none(),
    }))).toBe(true)
  })

  it("derives activity and identity from the same embedded instance", () => {
    const primary = configured({
      _tag: "Loading",
      stage: "loading",
      progress: Option.some(0.4),
      plannedAllocation: Option.none(),
    })
    const state = {
      slots: {
        primary,
        secondary: new ModelSlotUnassigned({ slotId: SECONDARY_SLOT_ID }),
      },
      recentModels: { primary: [], secondary: [] },
      favoriteModels: [],
    }
    expect(deriveLocalModelLoadActivity(state, PRIMARY_SLOT_ID)).toBe(primary)
    expect(Option.getOrThrow(modelSlotInstanceId(primary))).toBe(instanceId)
  })

  it("reports resident memory only from a ready or resident-stopping instance", () => {
    const ready = configured({ _tag: "Ready", allocation })
    expect(Option.getOrThrow(modelSlotResidentAllocation(ready))).toStrictEqual(allocation)
    expect(Option.isNone(modelSlotResidentAllocation(configured({
      _tag: "Loading",
      stage: "loading",
      progress: Option.none(),
      plannedAllocation: Option.none(),
    })))).toBe(true)
  })
})
