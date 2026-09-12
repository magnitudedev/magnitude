import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { Effect } from "effect"
import { makeLoginStartup, WINDOWS_APPLICATION_ID } from "./login-startup"

const app = vi.hoisted(() => ({ isPackaged: true, getLoginItemSettings: vi.fn(), setLoginItemSettings: vi.fn() }))
vi.mock("electron", () => ({ app }))
const executable = "C:\\Program Files\\Magnitude\\Magnitude.exe"
const launchItem = { name: WINDOWS_APPLICATION_ID, path: executable, args: [], scope: "user", enabled: true }

beforeEach(() => {
  vi.stubGlobal("process", { ...process, platform: "win32", execPath: executable })
  app.getLoginItemSettings.mockReset()
  app.setLoginItemSettings.mockReset()
})
afterEach(() => vi.unstubAllGlobals())
const read = () => Effect.runPromise(Effect.flatMap(makeLoginStartup(false), service => service.read))

describe("Windows login registration evidence", () => {
  it("uses complete-command matching when Electron omits switches from launchItems.args", async () => {
    app.getLoginItemSettings.mockReturnValue({ openAtLogin: true, launchItems: [launchItem] })
    expect(await read()).toEqual({ _tag: "Enabled" })
    expect(app.getLoginItemSettings).toHaveBeenCalledWith({ path: `"${executable}"`, args: ["--background"] })
  })
  it.each([
    { openAtLogin: false, launchItems: [launchItem] },
    { openAtLogin: true, launchItems: [{ ...launchItem, enabled: false }] },
    { openAtLogin: true, launchItems: [{ ...launchItem, name: "unrelated-entry" }] },
    { openAtLogin: true, launchItems: [{ ...launchItem, scope: "machine" }] },
    { openAtLogin: true, launchItems: [{ ...launchItem, enabled: false }, { ...launchItem, name: "unrelated-entry" }] },
  ])("does not combine another entry's approval with the app registration: %j", async settings => {
    app.getLoginItemSettings.mockReturnValue(settings)
    expect(await read()).toEqual({ _tag: "Disabled" })
  })
  it("does not report an unapplied OS change as successful", async () => {
    app.getLoginItemSettings.mockReturnValue({ openAtLogin: false, launchItems: [] })
    const result = await Effect.runPromise(Effect.flatMap(makeLoginStartup(false), service => service.set(true)).pipe(Effect.either))
    expect(result._tag).toBe("Left")
    expect(app.setLoginItemSettings).toHaveBeenCalledWith({ path: `"${executable}"`, args: ["--background"], openAtLogin: true })
  })
})
