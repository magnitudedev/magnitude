import { act } from "react"
import { KeyEvent } from "@opentui/core"
import { testRender } from "@opentui/react/test-utils"
import { describe, expect, it, vi } from "vitest"
import { HarnessIdSchema } from "@magnitudedev/client-common"
import { makeModel } from "../local-inference/test-fixtures"
import { HarnessChooser } from "./harness"

const keyboard = vi.hoisted(
  (): { handler: ((key: KeyEvent) => void) | undefined } => ({ handler: undefined }),
)

vi.mock("@opentui/react", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@opentui/react")>()),
  useKeyboard: (handler: (key: KeyEvent) => void) => { keyboard.handler = handler },
}))

const keyEvent = (name: string) => new KeyEvent({
  name,
  ctrl: false,
  meta: false,
  shift: false,
  option: false,
  sequence: "",
  number: false,
  raw: "",
  eventType: "press",
  source: "raw",
})

const destinations = [
  { id: HarnessIdSchema.make("magnitude"), name: "Magnitude", availability: "Installed" as const, selectable: true, note: "Optimized for local models" },
  { id: HarnessIdSchema.make("codex"), name: "Codex", availability: "Installed" as const, selectable: true },
  { id: HarnessIdSchema.make("cline"), name: "Cline", availability: "Not installed" as const, selectable: false },
]

describe("harness chooser layout", () => {
  it("renders one unboxed menu in the shared setup frame", async () => {
    const view = await testRender(
      <HarnessChooser
        width={120}
        model={makeModel()}
        destinations={destinations}
        applying={null}
        onBack={() => undefined}
        onContinue={vi.fn()}
      />,
      { width: 120, height: 40 },
    )

    try {
      await act(view.renderOnce)
      const frame = view.captureCharFrame()
      expect(frame).not.toContain("┌")
      expect(frame).toContain("› Magnitude  Optimized for local models")
      expect(frame).not.toContain("Recommended")
      expect(frame).not.toContain("Magnitude  Optimized for local models  Installed")
      expect(frame).toContain("  [x] Launch Magnitude on startup")
      expect(frame).not.toContain("[x] Magnitude  Optimized")
      expect(frame).not.toContain("Continue with Magnitude")
      expect(frame).toContain("↑/↓ choose · Space toggle · Enter continue · Esc back")
      const lines = frame.split("\n")
      const guidanceRow = lines.findIndex((line) => line.includes("You can change harness connections later"))
      expect(lines[guidanceRow - 1]?.trim()).toBe("")
    } finally {
      await act(async () => view.renderer.destroy())
    }
  })

  it("navigates harnesses and toggles as one menu and continues from any row", async () => {
    const onContinue = vi.fn()
    const view = await testRender(
      <HarnessChooser
        width={120}
        model={makeModel()}
        destinations={destinations}
        applying={null}
        onBack={() => undefined}
        onContinue={onContinue}
      />,
      { width: 120, height: 40 },
    )

    try {
      await act(view.renderOnce)
      act(() => keyboard.handler?.(keyEvent("enter")))
      expect(onContinue).toHaveBeenCalledWith(expect.objectContaining({ harness: "magnitude" }))
      for (const expected of [
        "› Codex",
        "› [x] Launch Magnitude on startup",
        "› [x] Install the Magnitude skill globally",
      ]) {
        act(() => keyboard.handler?.(keyEvent("down")))
        await act(view.renderOnce)
        expect(view.captureCharFrame()).toContain(expected)
      }
      act(() => keyboard.handler?.(keyEvent("enter")))
      await act(view.renderOnce)
      expect(onContinue).toHaveBeenCalledTimes(2)
      expect(onContinue).toHaveBeenCalledWith(expect.objectContaining({ harness: "codex" }))
    } finally {
      await act(async () => view.renderer.destroy())
    }
  })
})
