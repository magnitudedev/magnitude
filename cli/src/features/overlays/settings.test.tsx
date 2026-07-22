import { expect, test, vi } from "vitest"
import { Result } from "@effect-atom/atom-react"
import { Option } from "effect"
import { testRender } from "@opentui/react/test-utils"
import { act } from "react"
import { LocalInferenceAcceleratorIdSchema, LocalInferenceMemoryDomainIdSchema } from "@magnitudedev/sdk"
import { GIB, makeHardware, makeView } from "../local-inference/test-fixtures"

const textPosition = (frame: string, needle: string) => {
  const lines = frame.split("\n")
  const y = lines.findIndex((line) => line.includes(needle))
  if (y < 0) throw new Error(`Could not find ${needle}`)
  return { x: lines[y]!.indexOf(needle), y }
}

const memoryDomainId = LocalInferenceMemoryDomainIdSchema.make("system")
const localInferenceState = makeView({
  entries: [],
  ready: false,
  hardware: makeHardware({
    platform: "MacOS",
    architecture: "Arm64",
    processor: Option.some("Apple M4 Max"),
    totalSystemMemoryBytes: 64 * GIB,
    availableSystemMemoryBytes: Option.some(12 * GIB),
    accelerators: [{
      acceleratorId: LocalInferenceAcceleratorIdSchema.make("metal"),
      name: "Apple M4 Max",
      backend: "Metal",
      memoryDomainId,
    }],
    memoryDomains: [{
      memoryDomainId,
      kind: "UnifiedMemory",
      totalBytes: 64 * GIB,
      stableCapacityBytes: 51.2 * GIB,
      availableBytes: Option.some(12 * GIB),
      sharesSystemMemory: true,
    }],
    residentMemory: Option.some({
      domains: [{
        memoryDomainId,
        modelBytes: 27 * GIB,
        contextBytes: 6 * GIB,
        computeBytes: 1.5 * GIB,
        auxiliaryBytes: 0.5 * GIB,
      }],
    }),
  }),
})

vi.mock("@magnitudedev/client-common", async (importOriginal) => ({
  ...await importOriginal<typeof import("@magnitudedev/client-common")>(),
  useLocalInferenceQuery: () => Result.success(localInferenceState),
}))

vi.mock("../../hooks/use-theme", () => ({
  useTheme: () => ({
    primary: "blue",
    foreground: "white",
    muted: "gray",
    success: "green",
    error: "red",
    warning: "yellow",
    border: "gray",
    secondary: "gray",
    info: "cyan",
    link: "blue",
    terminalDetectedBg: "black",
  }),
}))

const { SettingsOverlay } = await import("./settings")

test("settings starts with detected hardware followed by explicit Magnitude Cloud status", async () => {
  const view = await testRender(
    <SettingsOverlay
      isVisible
      onClose={() => {}}
      auth={{
        source: "none",
        key: null,
        maskedKey: null,
        envVarName: null,
        save: () => {},
        clear: () => {},
        saving: false,
        error: null,
      }}
      slots={[]}
      onManageLocalModels={() => {}}
    />,
    { width: 120, height: 45 },
  )

  try {
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame).toContain("DETECTED HARDWARE")
    expect(frame).toContain("Apple M4 Max")
    expect(frame).toContain("Metal acceleration")
    expect(frame).toContain("Apple M4 Max · Unified memory")
    expect(frame).toContain("52.0 GiB / 64.0 GiB used")
    expect(frame).toContain("Weights       29.0 GiB")
    expect(frame).toContain("KV cache      6.0 GiB")
    expect(frame).toContain("System & apps 17.0 GiB")
    expect(frame).toContain("Free          12.0 GiB")
    expect(frame).toContain("Magnitude Cloud")
    expect(frame).toContain("○ Not connected")
    expect(frame).not.toContain("No Magnitude Cloud API key · No cloud model access")
    expect(frame).toContain("Add API Key")
    expect(frame).toContain("https://app.magnitude.dev")
    expect(frame).toContain("[Copy link]")
    expect(frame).not.toContain("Install runtime")
    expect(frame.indexOf("DETECTED HARDWARE")).toBeLessThan(frame.indexOf("Magnitude Cloud"))

    const addApiKey = textPosition(frame, "Add API Key")
    await act(async () => view.mockMouse.click(addApiKey.x, addApiKey.y))
    await act(view.renderOnce)
    const editFrame = view.captureCharFrame()
    expect(editFrame).not.toContain("Add API Key")
    expect(editFrame).toContain("○ Not connected")
    expect(editFrame).toContain("Paste Magnitude Cloud API key")
    expect(editFrame.indexOf("Paste Magnitude Cloud API key")).toBeLessThan(editFrame.indexOf("Save (Enter)"))
    expect(editFrame.indexOf("Save (Enter)")).toBeLessThan(editFrame.indexOf("Get an API key"))
    expect(editFrame).not.toContain("Configure Cloud fallback")
    expect(editFrame).not.toContain("Enter to save, Esc to cancel")
    expect(textPosition(editFrame, "Save (Enter)").y - textPosition(editFrame, "Paste Magnitude Cloud API key").y).toBe(2)
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("settings shows the connection source and key preview on one status line", async () => {
  const configured = await testRender(
    <SettingsOverlay
      isVisible
      onClose={() => {}}
      auth={{
        source: "config",
        key: null,
        maskedKey: "mag_………1234",
        envVarName: null,
        save: () => {},
        clear: () => {},
        saving: false,
        error: null,
      }}
      slots={[]}
      onManageLocalModels={() => {}}
    />,
    { width: 120, height: 45 },
  )

  try {
    await act(configured.renderOnce)
    const statusLine = configured.captureCharFrame().split("\n").find((line) => line.includes("Connected via API key"))
    expect(statusLine).toContain("● Connected via API key (mag_………1234)")
  } finally {
    await act(async () => configured.renderer.destroy())
  }

  const fromEnvironment = await testRender(
    <SettingsOverlay
      isVisible
      onClose={() => {}}
      auth={{
        source: "env",
        key: "mag_testabcdefghijkl1234",
        maskedKey: null,
        envVarName: "MAGNITUDE_API_KEY",
        save: () => {},
        clear: () => {},
        saving: false,
        error: null,
      }}
      slots={[]}
      onManageLocalModels={() => {}}
    />,
    { width: 120, height: 45 },
  )

  try {
    await act(fromEnvironment.renderOnce)
    const statusLine = fromEnvironment.captureCharFrame().split("\n").find((line) => line.includes("Connected via MAGNITUDE_API_KEY"))
    expect(statusLine).toContain("● Connected via MAGNITUDE_API_KEY (mag_test………1234)")
  } finally {
    await act(async () => fromEnvironment.renderer.destroy())
  }
})
