import { act, useCallback } from "react"
import { createTestRenderer } from "@opentui/core/testing"
import { createRoot, useKeyboard } from "@opentui/react"
import { testRender } from "@opentui/react/test-utils"
import { describe, expect, it, vi } from "vitest"

vi.mock("../../hooks/use-theme", () => ({
  useTheme: () => ({
    primary: "blue",
    foreground: "white",
    muted: "gray",
  }),
}))

const { UpdatePrompt } = await import("./prompt")

function MainKeyboard({ onCtrlC }: { readonly onCtrlC: () => void }) {
  useKeyboard(useCallback((key) => {
    if (key.ctrl && key.name === "c" && !key.defaultPrevented) onCtrlC()
  }, [onCtrlC]))
  return <text>Main TUI</text>
}

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

  it("releases Ctrl+C after skipping the update prompt", async () => {
    const onSelect = vi.fn()
    const onMainCtrlC = vi.fn()
    const reactTestEnvironment = globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean
    }
    const actEnvironment = reactTestEnvironment.IS_REACT_ACT_ENVIRONMENT
    reactTestEnvironment.IS_REACT_ACT_ENVIRONMENT = true
    const view = await createTestRenderer({ width: 100, height: 24 })
    const promptRoot = createRoot(view.renderer)

    try {
      await act(async () => promptRoot.render(
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
      ))
      await act(async () => view.mockInput.pressArrow("down"))
      await act(async () => view.mockInput.pressEnter())
      expect(onSelect).toHaveBeenCalledWith({ _tag: "Skip" })

      await act(async () => promptRoot.unmount())
      const mainRoot = createRoot(view.renderer)
      await act(async () => mainRoot.render(
        <MainKeyboard onCtrlC={onMainCtrlC} />,
      ))
      await act(async () => view.mockInput.pressCtrlC())

      expect(onMainCtrlC).toHaveBeenCalledOnce()
      expect(onSelect).toHaveBeenCalledTimes(1)
    } finally {
      await act(async () => view.renderer.destroy())
      reactTestEnvironment.IS_REACT_ACT_ENVIRONMENT = actEnvironment
    }
  })
})
