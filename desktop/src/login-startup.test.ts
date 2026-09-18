import { mkdtemp, readdir, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { NodeContext } from "@effect/platform-node"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { Effect } from "effect"
import { initializeLoginStartup, makeLoginStartup, WINDOWS_APPLICATION_ID } from "./login-startup"

const app = vi.hoisted(() => ({ isPackaged: true, isInApplicationsFolder: vi.fn(), getLoginItemSettings: vi.fn(), setLoginItemSettings: vi.fn() }))
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

describe("macOS first-run login startup", () => {
  let directory: string
  beforeEach(async () => {
    vi.stubGlobal("process", { ...process, platform: "darwin" })
    directory = await mkdtemp(join(tmpdir(), "magnitude-login-test-"))
    app.isInApplicationsFolder.mockReturnValue(true)
    app.isPackaged = true
  })
  afterEach(async () => { await rm(directory, { recursive: true, force: true }) })
  const initialize = (isolated = false) => Effect.runPromise(Effect.gen(function* () {
    const service = yield* makeLoginStartup(isolated)
    yield* initializeLoginStartup(service, directory, isolated)
    return yield* service.read
  }).pipe(Effect.provide(NodeContext.layer)))
  it("automatically registers a fresh installation and preserves later OS opt-out", async () => {
    app.getLoginItemSettings.mockReturnValue({ status: "not-found" })
    app.setLoginItemSettings.mockImplementation(() => app.getLoginItemSettings.mockReturnValue({ status: "enabled" }))
    expect(await initialize()).toEqual({ _tag: "Enabled" })
    expect(app.setLoginItemSettings).toHaveBeenCalledTimes(1)
    app.getLoginItemSettings.mockReturnValue({ status: "not-found" })
    expect(await initialize()).toEqual({ _tag: "Disabled" })
    expect(app.setLoginItemSettings).toHaveBeenCalledTimes(1)
  })
  it.each(["not-registered", "enabled", "requires-approval"])("preserves existing %s state", async status => {
    app.getLoginItemSettings.mockReturnValue({ status })
    await initialize()
    expect(app.setLoginItemSettings).not.toHaveBeenCalled()
  })
  it("accepts registration requiring OS approval", async () => {
    app.getLoginItemSettings.mockReturnValue({ status: "not-found" })
    app.setLoginItemSettings.mockImplementation(() => app.getLoginItemSettings.mockReturnValue({ status: "requires-approval" }))
    expect(await initialize()).toEqual({ _tag: "RequiresApproval" })
  })
  it("reports failed registration without repeated automatic attempts; explicit retry works", async () => {
    app.getLoginItemSettings.mockReturnValue({ status: "not-found" })
    await expect(initialize()).rejects.toThrow("did not apply")
    expect(await initialize()).toEqual({ _tag: "Disabled" })
    expect(app.setLoginItemSettings).toHaveBeenCalledTimes(1)
    app.setLoginItemSettings.mockImplementation(() => app.getLoginItemSettings.mockReturnValue({ status: "enabled" }))
    expect(await Effect.runPromise(Effect.flatMap(makeLoginStartup(false), service => service.set(true)))).toEqual({ _tag: "Enabled" })
  })
  it("never registers during a Settings read", async () => {
    app.getLoginItemSettings.mockReturnValue({ status: "not-found" })
    expect(await read()).toEqual({ _tag: "Disabled" })
    expect(app.setLoginItemSettings).not.toHaveBeenCalled()
  })
  it.each(["isolated", "unpackaged", "outside-applications"])("excludes %s", async mode => {
    app.isPackaged = mode !== "unpackaged"
    app.isInApplicationsFolder.mockReturnValue(mode !== "outside-applications")
    app.getLoginItemSettings.mockReturnValue({ status: "not-found" })
    await initialize(mode === "isolated")
    expect(app.setLoginItemSettings).not.toHaveBeenCalled()
    expect(await readdir(directory)).toEqual([])
  })
  it("does not register if the attempt cannot be recorded", async () => {
    await rm(directory, { recursive: true })
    app.getLoginItemSettings.mockReturnValue({ status: "not-found" })
    await expect(initialize()).rejects.toThrow("Could not initialize")
    expect(app.setLoginItemSettings).not.toHaveBeenCalled()
  })
})
