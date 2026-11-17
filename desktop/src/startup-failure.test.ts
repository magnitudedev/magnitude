import { Cause, Effect } from "effect"
import { describe, expect, it } from "vitest"
import { ApplicationControlFailed, ApplicationOwnershipFailed, serveApplicationControl } from "@magnitudedev/daemon-management/desktop-native"
import { STARTUP_FAILED, startupFailureDetail } from "./startup-failure"

describe("the startup error dialog", () => {
  it("gives a control or ownership failure that isn't an unusable data folder the general advice, not its raw error", () => {
    expect(startupFailureDetail(Cause.fail(new ApplicationControlFailed({ message: "Error: listen EACCES: permission denied /data/application.sock" })))).toBe(STARTUP_FAILED)
    expect(startupFailureDetail(Cause.fail(new ApplicationOwnershipFailed({ message: "The existing application directory could not be resolved." })))).toBe(STARTUP_FAILED)
  })

  it.skipIf(process.platform === "win32")("names a data folder too deep for the control socket", async () => {
    const path = `/${"deep/".repeat(30)}application.sock`
    const failure = await Effect.runPromise(serveApplicationControl(path, {
      snapshot: Effect.die("unused"), dispatch: () => Effect.void, login: () => Effect.die("unused"), update: () => Effect.die("unused"),
    }).pipe(Effect.flip, Effect.scoped))
    const detail = startupFailureDetail(Cause.fail(failure))
    expect(detail).toContain(path)
    expect(detail).not.toBe(STARTUP_FAILED)
  })
})
