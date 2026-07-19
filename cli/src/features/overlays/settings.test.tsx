import { expect, test, vi } from "vitest"
import { Result } from "@effect-atom/atom-react"
import { Option } from "effect"
import { testRender } from "@opentui/react/test-utils"
import { act } from "react"
import type { LocalInferenceState } from "@magnitudedev/sdk"

const gib = 1024 ** 3
const localInferenceState = {
  usage: { sessionConcurrency: "one" },
  activeBinding: null,
  llamaCpp: {
    minimumBuild: 8868,
    recommendedBuild: 10011,
    installations: [],
    selectedInstallationId: Option.none(),
    activeManagedInstallationId: Option.none(),
    managedInstall: { availability: { _tag: "Available", build: 10011 }, operation: { _tag: "Idle" } },
    diagnostics: [],
  },
  host: {
    _tag: "Available",
    profile: {
      platform: "darwin",
      architecture: "arm64",
      systemMemoryBytes: 64 * gib,
      cpuModel: "Apple M4 Max",
      logicalCores: 16,
      memoryDomains: [{
        id: "system",
        kind: "system",
        totalCapacityBytes: 64 * gib,
        stableCapacityBytes: 51.2 * gib,
        currentFreeBytes: null,
        sharesSystemMemory: false,
        backendNames: [],
        deviceNames: [],
        splitGroupId: null,
      }, {
        id: "metal",
        kind: "unified_working_set",
        totalCapacityBytes: 48 * gib,
        stableCapacityBytes: 43.2 * gib,
        currentFreeBytes: null,
        sharesSystemMemory: true,
        backendNames: ["Metal"],
        deviceNames: ["Apple M4 Max"],
        splitGroupId: null,
      }],
    },
  },
  choices: [],
  operations: [],
  recommendations: [],
  warnings: [],
} as const satisfies LocalInferenceState

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
      onConfigureCloud={() => {}}
    />,
    { width: 120, height: 45 },
  )

  try {
    await act(view.renderOnce)
    const frame = view.captureCharFrame()
    expect(frame).toContain("DETECTED HARDWARE")
    expect(frame).toContain("Apple M4 Max")
    expect(frame).toContain("64.0 GiB unified memory · Metal GPU acceleration")
    expect(frame).toContain("Magnitude Cloud")
    expect(frame).toContain("No Magnitude Cloud API key · No cloud model access")
    expect(frame).toContain("https://app.magnitude.dev")
    expect(frame).toContain("[Copy link]")
    expect(frame).not.toContain("llama.cpp")
    expect(frame.indexOf("DETECTED HARDWARE")).toBeLessThan(frame.indexOf("Magnitude Cloud"))
  } finally {
    await act(async () => view.renderer.destroy())
  }
})
