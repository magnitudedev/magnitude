import { describe, expect, it } from "vitest"
import { raiseWindow, type RaisableWindow } from "./window-raise"

const recording = (minimized: boolean) => {
  const calls: string[] = []
  const window: RaisableWindow = {
    isMinimized: () => minimized,
    restore: () => { calls.push("restore") },
    show: () => { calls.push("show") },
    focus: () => { calls.push("focus") },
    setAlwaysOnTop: (flag: boolean) => { calls.push(`alwaysOnTop ${flag}`) },
  }
  const application = { focus: (options?: { steal: boolean }) => { calls.push(`app focus${options?.steal ? " steal" : ""}`) } }
  return { calls, window, application }
}

describe("raising the window", () => {
  it("takes focus on Windows by making the window topmost only while it focuses", () => {
    const { calls, window, application } = recording(true)
    raiseWindow(window, application, "win32")
    expect(calls).toEqual(["restore", "alwaysOnTop true", "show", "focus", "alwaysOnTop false"])
  })

  it("activates the app on macOS even while another app is active", () => {
    const { calls, window, application } = recording(false)
    raiseWindow(window, application, "darwin")
    expect(calls).toEqual(["show", "app focus steal", "focus"])
  })

  it("restores, shows and focuses on Linux", () => {
    const { calls, window, application } = recording(true)
    raiseWindow(window, application, "linux")
    expect(calls).toEqual(["restore", "show", "focus"])
  })
})
