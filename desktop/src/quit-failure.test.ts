import { Effect } from "effect"
import { describe, expect, it, vi } from "vitest"
import { resolveQuitFailure } from "./quit-failure"

describe("failed application cleanup", () => {
  it.each([0, 1, 2])("requires the explicit force choice to exit (choice %s)", async response => {
    const forceQuit = vi.fn()
    const showDialog = vi.fn(async () => ({ response, checkboxChecked: false }))
    const retry = await Effect.runPromise(resolveQuitFailure("The service did not stop", { showDialog, forceQuit }))
    expect(retry).toBe(response === 1)
    expect(forceQuit).toHaveBeenCalledTimes(response === 2 ? 1 : 0)
    expect(showDialog).toHaveBeenCalledWith(expect.objectContaining({
      cancelId: 0,
      defaultId: 1,
      detail: expect.stringContaining("without confirming that all background processes stopped"),
    }))
  })

  it("retains ownership when the native dialog fails", async () => {
    const forceQuit = vi.fn()
    const retry = await Effect.runPromise(resolveQuitFailure("Cleanup failed", {
      showDialog: () => Promise.reject(new Error("Dialog unavailable")), forceQuit,
    }))
    expect(retry).toBe(false)
    expect(forceQuit).not.toHaveBeenCalled()
  })
})
