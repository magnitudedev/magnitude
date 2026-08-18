import { act } from "react"
import {
  Registry,
  RegistryContext,
  scheduleTask,
} from "@effect-atom/atom-react"
import { testRender } from "@opentui/react/test-utils"
import { describe, expect, it, vi } from "vitest"

vi.mock("../../hooks/use-theme", () => ({
  useTheme: () => ({
    accent: "blue",
    text: {
      body: "white",
      supporting: "gray",
    },
  }),
}))

const { UpdatePrompt } = await import("./prompt")
const { CliStartupRoot, makeCliRootStateAtom } = await import("../../runtime/root")

describe("UpdatePrompt", () => {
  it("shows the Codex-style choices and exact package-manager command", async () => {
    const onSelect = vi.fn()
    const view = await testRender(
      <UpdatePrompt
        currentVersion="0.0.1-alpha.34"
        latestVersion="0.0.1-alpha.35"
        action={{
          method: "npm",
          command: "npm",
          args: ["install", "-g", "@magnitudedev/cli"],
        }}
        onSelect={onSelect}
      />,
      { width: 100, height: 24 },
    )

    try {
      await act(view.renderOnce)
      const frame = view.captureCharFrame()
      expect(frame).toContain("Update available!")
      expect(frame).toContain("0.0.1-alpha.34 -> 0.0.1-alpha.35")
      expect(frame).toContain("1. Update now (runs `npm install -g @magnitudedev/cli`)")
      expect(frame).toContain("2. Skip")
      expect(frame).toContain("3. Skip until next version")
      expect(frame).toContain("Press Enter to continue")

      await act(async () => view.mockInput.pressArrow("down"))
      await act(async () => view.mockInput.pressEnter())
      expect(onSelect).toHaveBeenCalledWith({ _tag: "Skip" })
    } finally {
      await act(async () => view.renderer.destroy())
    }
  })

  it("transitions through the stable CLI root without leaving prompt keyboard handlers", async () => {
    const onSelect = vi.fn()
    const reactTestEnvironment = globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean
    }
    const actEnvironment = reactTestEnvironment.IS_REACT_ACT_ENVIRONMENT
    reactTestEnvironment.IS_REACT_ACT_ENVIRONMENT = true
    const registry = Registry.make({ scheduleTask, defaultIdleTTL: 5_000 })
    const stateAtom = makeCliRootStateAtom()
    registry.set(stateAtom, {
      _tag: "UpdateAvailable",
      currentVersion: "0.0.1-alpha.34",
      latestVersion: "0.0.1-alpha.35",
      action: {
        method: "npm",
        command: "npm",
        args: ["install", "-g", "@magnitudedev/cli"],
      },
    })
    const view = await testRender(
      <RegistryContext.Provider value={registry}>
        <CliStartupRoot stateAtom={stateAtom} onUpdateSelect={onSelect} />
      </RegistryContext.Provider>,
      { width: 100, height: 24 },
    )

    try {
      await act(view.renderOnce)
      await act(async () => view.mockInput.pressArrow("down"))
      await act(async () => view.mockInput.pressEnter())
      expect(onSelect).toHaveBeenCalledWith({ _tag: "Skip" })

      await act(async () => registry.set(
        stateAtom,
        { _tag: "Starting", stage: "Platform" },
      ))
      expect(view.captureCharFrame()).toContain("Update available!")
      await act(view.renderOnce)
      expect(view.captureCharFrame()).toContain("Starting local services…")
      await act(async () => view.mockInput.pressCtrlC())

      expect(onSelect).toHaveBeenCalledTimes(1)
    } finally {
      await act(async () => view.renderer.destroy())
      reactTestEnvironment.IS_REACT_ACT_ENVIRONMENT = actEnvironment
    }
  })
})
