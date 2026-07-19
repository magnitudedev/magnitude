import { expect, test, vi } from "vitest"
import { act } from "react"
import { testRender } from "@opentui/react/test-utils"

const saveApiKey = vi.fn()

vi.mock("@magnitudedev/client-common", async (importOriginal) => ({
  ...await importOriginal<typeof import("@magnitudedev/client-common")>(),
  useSettingsState: () => ({
    apiKey: { status: "none" },
    keyAlreadySet: false,
    loading: false,
    loadError: null,
    saving: false,
    saveError: null,
    saveApiKey,
    disconnectApiKey: () => {},
  }),
}))

vi.mock("../../hooks/use-theme", () => ({
  useTheme: () => ({
    primary: "blue",
    foreground: "white",
    muted: "gray",
    success: "green",
    error: "red",
    border: "gray",
  }),
}))

const { CloudModelsScreen } = await import("./screen")

test("standalone cloud setup saves the shared Magnitude API key", async () => {
  saveApiKey.mockClear()
  const view = await testRender(
    <CloudModelsScreen onExit={() => {}} />,
    { width: 120, height: 30 },
  )

  try {
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame).toContain("CLOUD MODELS")
    expect(frame).not.toContain("CLOUD MODELS (OPTIONAL)")
    expect(frame).not.toContain("Back to local models")
    expect(frame).not.toContain("Skip for now")

    await act(async () => view.mockInput.typeText("k"))
    await act(async () => view.mockInput.pressEnter())
    expect(saveApiKey).toHaveBeenCalledWith("k")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})
