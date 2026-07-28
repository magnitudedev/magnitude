import { Option } from "effect"
import { expect, test } from "vitest"
import {
  ModelSlotConfiguredLocal,
  ModelInstanceIdSchema,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  SECONDARY_SLOT_ID,
} from "@magnitudedev/sdk"
import {
  GIB,
  LOCAL_PROVIDER_ID,
  makeView,
  TEST_CONFIGURATION_ID,
  TEST_MEMORY_DOMAIN_ID,
  TEST_MODEL_ID,
  TEST_REASONING_EFFORT,
} from "./test-fixtures"

const { deriveLocalInferenceFooterView } = await import("./footer-status")
const instanceId = ModelInstanceIdSchema.make("test-instance")
const selection = {
  providerId: LOCAL_PROVIDER_ID,
  providerModelId: TEST_MODEL_ID,
  reasoningEffort: TEST_REASONING_EFFORT,
}
const localSlot = (
  slotId: typeof PRIMARY_SLOT_ID | typeof SECONDARY_SLOT_ID,
  instance: ModelSlotConfiguredLocal["instance"],
) => new ModelSlotConfiguredLocal({
  slotId,
  selection,
  descriptor: {
    providerId: LOCAL_PROVIDER_ID,
    providerModelId: TEST_MODEL_ID,
    displayName: "Qwen Test",
  },
  availability: { _tag: "Available" },
  instance,
  actions: Option.isSome(instance) && instance.value.lifecycle._tag === "Loading"
    ? ["Stop"]
    : [],
})
const loadingInstance = (progress: number) => Option.some({
  id: instanceId,
  configurationId: TEST_CONFIGURATION_ID,
  lifecycle: {
    _tag: "Loading" as const,
    stage: "loading" as const,
    progress: Option.some(progress),
    plannedAllocation: Option.none(),
  },
})

test("ready status exposes the model, residency, and complete resident allocation", () => {
  const state = makeView({
    allocation: {
      contextWindowTokens: 32_768,
      parallelSequences: 1,
      physicalContextTokens: 32_768,
      memoryDomains: [{
        memoryDomainId: TEST_MEMORY_DOMAIN_ID,
        modelBytes: 13 * GIB,
        contextBytes: 2 * GIB,
        computeBytes: GIB,
        auxiliaryBytes: 0,
      }],
    },
  })
  expect(deriveLocalInferenceFooterView(
    state.models,
    state.slots,
    "Qwen Test",
    LOCAL_PROVIDER_ID,
    PRIMARY_SLOT_ID,
  )).toEqual({
    modelName: "Qwen Test",
    residency: "loaded",
    memoryLabel: "16 GB mem",
  })
})

test("slot residency remains visible when local-model inventory is unavailable", () => {
  const state = makeView({
    allocation: {
      contextWindowTokens: 32_768,
      parallelSequences: 1,
      physicalContextTokens: 32_768,
      memoryDomains: [{
        memoryDomainId: TEST_MEMORY_DOMAIN_ID,
        modelBytes: 13 * GIB,
        contextBytes: 2 * GIB,
        computeBytes: GIB,
        auxiliaryBytes: 0,
      }],
    },
  })
  expect(deriveLocalInferenceFooterView(
    null,
    state.slots,
    "Qwen Test",
    LOCAL_PROVIDER_ID,
    PRIMARY_SLOT_ID,
  )).toEqual({
    modelName: "Qwen Test",
    residency: "loaded",
    memoryLabel: "16 GB mem",
  })
})

test("loading status remains in the activity rail", () => {
  const ready = makeView()
  const state = {
    ...ready,
    slots: {
      ...ready.slots,
      slots: {
        ...ready.slots.slots,
        primary: localSlot(PRIMARY_SLOT_ID, loadingInstance(0.42)),
      },
    },
  }
  const footer = deriveLocalInferenceFooterView(
    state.models,
    state.slots,
    "Qwen Test",
    LOCAL_PROVIDER_ID,
    PRIMARY_SLOT_ID,
  )
  expect(footer).toEqual({ modelName: "Qwen Test", residency: "loading", memoryLabel: null })
})

test("memory state comes from the selected slot", () => {
  const ready = makeView()
  const state = {
    ...ready,
    slots: {
      ...ready.slots,
      slots: {
        ...ready.slots.slots,
        secondary: localSlot(SECONDARY_SLOT_ID, loadingInstance(0.27)),
      },
    },
  }
  expect(deriveLocalInferenceFooterView(
    state.models,
    state.slots,
    "Qwen Test",
    LOCAL_PROVIDER_ID,
    SECONDARY_SLOT_ID,
  )).toMatchObject({
    residency: "loading",
    memoryLabel: null,
  })
})

test("idle status keeps reasoning available and hides memory", () => {
  const ready = makeView()
  const state = {
    ...ready,
    slots: {
      ...ready.slots,
      slots: {
        ...ready.slots.slots,
        primary: localSlot(PRIMARY_SLOT_ID, Option.none()),
      },
    },
  }
  expect(deriveLocalInferenceFooterView(
    state.models,
    state.slots,
    "Qwen Test",
    LOCAL_PROVIDER_ID,
    PRIMARY_SLOT_ID,
  )).toEqual({
    modelName: "Qwen Test",
    residency: "not_loaded",
    memoryLabel: null,
  })
})

test("cloud selection exposes the model with no local runtime status", () => {
  expect(deriveLocalInferenceFooterView(
    null,
    null,
    "Claude Max",
    ProviderIdSchema.make("magnitude"),
    PRIMARY_SLOT_ID,
  )).toEqual({
    modelName: "Claude Max",
    residency: null,
    memoryLabel: null,
  })
})
