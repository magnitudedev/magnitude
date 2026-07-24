import { Option } from "effect"
import { expect, test } from "vitest"
import {
  ModelSlotLoadingLocalModel,
  ModelSlotUnloadedLocalModel,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  SECONDARY_SLOT_ID,
} from "@magnitudedev/sdk"
import { GIB, LOCAL_PROVIDER_ID, makeHardware, makeView, TEST_MEMORY_DOMAIN_ID, TEST_MODEL_ID, TEST_REASONING_EFFORT } from "./test-fixtures"

const { deriveLocalInferenceFooterView } = await import("./footer-status")

test("ready status exposes the model and minimal resident memory", () => {
  const state = makeView({
    hardware: makeHardware({
      residentMemory: Option.some({
        domains: [{
          memoryDomainId: TEST_MEMORY_DOMAIN_ID,
          modelBytes: 13 * GIB,
          contextBytes: 2 * GIB,
          computeBytes: GIB,
          auxiliaryBytes: 0,
        }],
      }),
    }),
  })
  expect(deriveLocalInferenceFooterView(state, "Qwen Test", LOCAL_PROVIDER_ID, PRIMARY_SLOT_ID)).toEqual({
    modelName: "Qwen Test",
    memoryLabel: "18 / 24 GiB",
  })
})

test("loading hides memory while status remains in the activity rail", () => {
  const ready = makeView()
  const state = { ...ready, slots: { ...ready.slots, slots: { ...ready.slots.slots, primary: new ModelSlotLoadingLocalModel({
    slotId: PRIMARY_SLOT_ID,
    selection: {
      providerId: LOCAL_PROVIDER_ID,
      providerModelId: TEST_MODEL_ID,
      reasoningEffort: TEST_REASONING_EFFORT,
    },
    percentage: 42,
  }) } } }
  const footer = deriveLocalInferenceFooterView(state, "Qwen Test", LOCAL_PROVIDER_ID, PRIMARY_SLOT_ID)
  expect(footer).toEqual({ modelName: "Qwen Test", memoryLabel: null })
})

test("memory state comes from the selected slot", () => {
  const ready = makeView()
  const state = {
    ...ready,
    slots: {
      ...ready.slots,
      slots: {
        ...ready.slots.slots,
        secondary: new ModelSlotLoadingLocalModel({
          slotId: SECONDARY_SLOT_ID,
          selection: {
            providerId: LOCAL_PROVIDER_ID,
            providerModelId: TEST_MODEL_ID,
            reasoningEffort: TEST_REASONING_EFFORT,
          },
          percentage: 27,
        }),
      },
    },
  }
  expect(deriveLocalInferenceFooterView(
    state,
    "Qwen Test",
    LOCAL_PROVIDER_ID,
    SECONDARY_SLOT_ID,
  )).toMatchObject({
    memoryLabel: null,
  })
})

test("idle status keeps reasoning available and hides memory", () => {
  const ready = makeView()
  const selection = {
    providerId: LOCAL_PROVIDER_ID,
    providerModelId: TEST_MODEL_ID,
    reasoningEffort: TEST_REASONING_EFFORT,
  }
  const state = {
    ...ready,
    slots: {
      ...ready.slots,
      slots: {
        ...ready.slots.slots,
        primary: new ModelSlotUnloadedLocalModel({ slotId: PRIMARY_SLOT_ID, selection }),
      },
    },
  }
  expect(deriveLocalInferenceFooterView(state, "Qwen Test", LOCAL_PROVIDER_ID, PRIMARY_SLOT_ID)).toEqual({
    modelName: "Qwen Test",
    memoryLabel: null,
  })
})

test("cloud selection exposes the model with no local runtime status", () => {
  expect(deriveLocalInferenceFooterView(
    null,
    "Claude Max",
    ProviderIdSchema.make("magnitude"),
    PRIMARY_SLOT_ID,
  )).toEqual({
    modelName: "Claude Max",
    memoryLabel: null,
  })
})
