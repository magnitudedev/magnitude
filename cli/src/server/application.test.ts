import { Effect } from "effect"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ApplicationUpdateControlFailed } from "@magnitudedev/sdk/desktop-host"
const calls = vi.hoisted(() => ({ observe: vi.fn(), update: vi.fn(), launch: vi.fn() }))
vi.mock("@magnitudedev/daemon-management/bun", () => ({ bundledWindowsNative: {} }))
vi.mock("@magnitudedev/daemon-management/desktop-native", () => ({
  applicationStateDirectory: () => Effect.succeed("/test/state"),
  makeDesktopApplicationHost: () => ({
    desktopApplication: { observe: Effect.suspend(() => calls.observe()), ensure: Effect.suspend(() => calls.launch()) },
    updateDesktopApplication: (action: string) => calls.update(action),
    desktopDataDirectory: "/test/data", desktopIsolatedProfile: true,
  }),
}))
import { updateApplication } from "./application"
const state = { transfer: { _tag: "Idle" }, check: { _tag: "Idle" }, preference: { _tag: "Known", autoDownload: true } }
beforeEach(() => { vi.clearAllMocks(); calls.update.mockReturnValue(Effect.succeed(state)) })
describe("application update routing", () => {
  it.each(["Desktop", "Headless"])("routes to the observed %s owner without launching or local mutation", async owner => {
    calls.observe.mockReturnValue(Effect.succeed({ owner: { _tag: owner } }))
    expect(await Effect.runPromise(updateApplication("download"))).toEqual({ owner, state })
    expect(calls.update).toHaveBeenCalledExactlyOnceWith("download")
    expect(calls.launch).not.toHaveBeenCalled()
  })
  it("reaches the installed-only maintenance guard after confirmed absence without launching", async () => {
    calls.observe.mockReturnValue(Effect.fail({ _tag: "ApplicationControlUnavailable", message: "Absent" }))
    const error = await Effect.runPromise(updateApplication("status").pipe(Effect.flip))
    expect(error.message).toBe("Application updates require an installed Magnitude application.")
    expect(calls.update).not.toHaveBeenCalled()
    expect(calls.launch).not.toHaveBeenCalled()
  })
  it("does not replay an owner mutation locally after a lost reply", async () => {
    calls.observe.mockReturnValue(Effect.succeed({ owner: { _tag: "Desktop" } }))
    calls.update.mockReturnValue(Effect.fail(new ApplicationUpdateControlFailed({ message: "Connection lost" })))
    expect((await Effect.runPromise(updateApplication("discard").pipe(Effect.flip))).message).toBe("Connection lost")
    expect(calls.update).toHaveBeenCalledTimes(1)
    expect(calls.launch).not.toHaveBeenCalled()
  })
})
