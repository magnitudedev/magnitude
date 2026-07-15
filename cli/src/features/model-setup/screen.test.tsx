import { expect, test, vi } from "vitest"
import { Result } from "@effect-atom/atom-react"
import { testRender } from "@opentui/react/test-utils"
import { act } from "react"

const localInferenceState = {
  schemaVersion: 3,
  usage: null,
  activeBinding: null,
  distribution: { _tag: "Missing" },
  host: { _tag: "Unavailable", message: "not needed" },
  choices: [],
  recommendations: [],
  warnings: [],
} as const

vi.mock("@magnitudedev/client-common", async (importOriginal) => ({
  ...await importOriginal<typeof import("@magnitudedev/client-common")>(),
  useSettingsState: () => ({
    keyAlreadySet: false,
    saveApiKey: () => {},
    saving: false,
    saveError: null,
  }),
  useLocalInferenceState: () => ({
    state: Result.success(localInferenceState),
    mutationResults: [Result.initial()],
    configureUsage: () => {},
    installDistribution: () => {},
    downloadModel: () => {},
    activateModel: () => {},
    deleteModel: () => {},
    restart: () => {},
    disable: () => {},
  }),
}))
vi.mock("../../hooks/use-theme", () => ({
  useTheme: () => ({
    primary: "blue",
    foreground: "white",
    muted: "gray",
    success: "green",
  }),
}))

const { ModelSetupScreen, resolveModelSetupSurface } = await import("./screen")

test("local management renders local inference even when it is entered independently", async () => {
  const view = await testRender(
    <ModelSetupScreen
      initialStep="local"
      mode="management"
      onExit={() => {}}
    />,
    { width: 80, height: 24 },
  )

  try {
    await act(view.renderOnce)
    expect(view.captureCharFrame()).toContain("LOCAL MODEL SETUP")
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("management preserves an explicitly selected cloud surface", () => {
  expect(resolveModelSetupSurface("management", "cloud")).toBe("cloud-provider")
})
