import { Effect } from "effect"
import { describe, expect, it, vi } from "vitest"
import { nativeQuitFailureDecision, resolveQuitFailure } from "./quit-failure"

describe("failed application cleanup", () => {
  it.each([0, 1, 2])("requires the explicit force choice to exit (native choice %s)", async response => {
    const forceQuit = vi.fn()
    const showDialog = vi.fn(async () => ({ response, checkboxChecked: false }))
    const retry = await Effect.runPromise(resolveQuitFailure("The service did not stop", { decide: nativeQuitFailureDecision(showDialog), forceQuit }))
    expect(JSON.stringify(showDialog.mock.calls)).not.toContain("The service did not stop")
    expect(retry).toBe(response === 1)
    expect(forceQuit).toHaveBeenCalledTimes(response === 2 ? 1 : 0)
    expect(showDialog).toHaveBeenCalledWith(expect.objectContaining({
      cancelId: 0,
      defaultId: 1,
      detail: expect.stringContaining("some background processes may still be running"),
    }))
  })

  it.each([["KeepOpen", false, 0], ["RetryQuit", true, 0], ["ForceQuit", false, 1]] as const)("applies the in-app decision %s", async (decision, retry, forced) => {
    const forceQuit = vi.fn()
    expect(await Effect.runPromise(resolveQuitFailure("Cleanup failed", { decide: Effect.succeed(decision), forceQuit }))).toBe(retry)
    expect(forceQuit).toHaveBeenCalledTimes(forced)
  })

  it("retains ownership when the native dialog fails", async () => {
    const forceQuit = vi.fn()
    const retry = await Effect.runPromise(resolveQuitFailure("Cleanup failed", {
      decide: nativeQuitFailureDecision(() => Promise.reject(new Error("Dialog unavailable"))), forceQuit,
    }))
    expect(retry).toBe(false)
    expect(forceQuit).not.toHaveBeenCalled()
  })
})
