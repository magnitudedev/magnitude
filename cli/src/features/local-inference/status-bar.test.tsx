import { act } from "react"
import { testRender } from "@opentui/react/test-utils"
import { Option } from "effect"
import { expect, test, vi } from "vitest"
import { ModelSlotLoadingLocalModel, PRIMARY_SLOT_ID, ProviderIdSchema } from "@magnitudedev/sdk"
import { GIB, LOCAL_PROVIDER_ID, makeHardware, makeView, TEST_MEMORY_DOMAIN_ID, TEST_MODEL_ID, TEST_REASONING_EFFORT } from "./test-fixtures"

vi.mock("../../hooks/use-theme", () => ({
  useTheme: () => ({
    primary: "blue", secondary: "gray", info: "cyan", link: "blue",
    foreground: "white", muted: "gray", border: "gray", warning: "magenta",
  }),
}))

const { LocalInferenceStatusBar } = await import("./status-bar")

test("ready status renders model and resident memory", async () => {
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
  const view = await testRender(
    <LocalInferenceStatusBar state={state} width={100} selectedModelName="Qwen Test" selectedProviderId={LOCAL_PROVIDER_ID} onOpenModels={() => {}} onOpenHardware={() => {}} />,
    { width: 110, height: 5 },
  )
  try {
    await act(view.renderOnce)
    expect(view.captureCharFrame()).toContain("Qwen Test")
    expect(view.captureCharFrame()).toContain("Ready")
    expect(view.captureCharFrame()).toContain("Memory")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("loading status shows native progress", async () => {
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
  const view = await testRender(
    <LocalInferenceStatusBar state={state} width={100} selectedModelName="Qwen Test" selectedProviderId={LOCAL_PROVIDER_ID} onOpenModels={() => {}} onOpenHardware={() => {}} />,
    { width: 110, height: 5 },
  )
  try {
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame).toContain("Loading 42%")
    expect(frame).not.toContain("Loading Loading")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("cloud selection keeps the model bar visible without local state", async () => {
  const view = await testRender(
    <LocalInferenceStatusBar state={null} width={100} selectedModelName="Claude Max" selectedProviderId={ProviderIdSchema.make("magnitude")} onOpenModels={() => {}} onOpenHardware={() => {}} />,
    { width: 110, height: 5 },
  )
  try {
    await act(view.renderOnce)
    expect(view.captureCharFrame()).toContain("Claude Max")
    expect(view.captureCharFrame()).toContain("Cloud")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})
