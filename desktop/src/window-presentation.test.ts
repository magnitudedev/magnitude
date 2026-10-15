import { Effect } from "effect"
import { expect, it } from "vitest"
import { presentWindow } from "./window-presentation"

/** A window whose process was started hidden: like Windows, the first show takes the launcher's SW_HIDE. */
const startedWindow = (startedHidden: boolean) => {
  const calls: string[] = []
  let visible = false
  let pendingStartupShow = startedHidden
  return {
    calls,
    window: {
      isMinimized: () => false,
      restore: () => { calls.push("restore") },
      show: () => {
        calls.push("show")
        if (pendingStartupShow) pendingStartupShow = false
        else visible = true
      },
      isVisible: () => visible,
      focus: () => { calls.push("focus") },
    },
  }
}

it("shows a Windows window whose process was started hidden", async () => {
  const started = startedWindow(true)
  await Effect.runPromise(presentWindow("win32", started.window))
  expect(started.window.isVisible()).toBe(true)
  expect(started.calls).toEqual(["show", "show", "focus"])
})
it("shows a normally started window once", async () => {
  for (const platform of ["win32", "darwin", "linux"] as const) {
    const started = startedWindow(false)
    await Effect.runPromise(presentWindow(platform, started.window))
    expect(started.window.isVisible()).toBe(true)
    expect(started.calls).toEqual(["show", "focus"])
  }
})
it("restores a minimized window before showing it", async () => {
  const started = startedWindow(false)
  await Effect.runPromise(presentWindow("win32", { ...started.window, isMinimized: () => true }))
  expect(started.calls).toEqual(["restore", "show", "focus"])
})
