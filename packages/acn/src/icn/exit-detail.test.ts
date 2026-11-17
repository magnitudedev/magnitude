import { describe, expect, it } from "vitest"
import { Error as PlatformError } from "@effect/platform"
import { Cause, Effect, Option } from "effect"
import { IcnUnexpectedExit } from "@magnitudedev/icn"
import { makeAcnServiceLifecycle } from "../service-lifecycle"
import { acnStartupFailureDetail } from "../server"
import { icnExitDetail } from "./layer"

const commandLine = "/Users/someone/Applications/Magnitude.app/Contents/Resources/magnitude-inference serve --bind 127.0.0.1:0 --instance-id abc --installation /Users/someone/.magnitude/engine"
const exitCodeUnreadable = new PlatformError.SystemError({ reason: "Unknown", module: "Command", method: "exitCode", pathOrDescriptor: commandLine })

const stoppedWith = (detail: string | undefined) => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const lifecycle = yield* makeAcnServiceLifecycle()
  yield* lifecycle.beginStopping({ reason: "icn-exited", detail })
  return yield* lifecycle.awaitStopping
})))

describe("an engine that stops unexpectedly", () => {
  it("is described by its exit status, with no command line or path", async () => {
    const stopping = await stoppedWith(icnExitDetail(new IcnUnexpectedExit({ pid: 4242, code: 134, output: commandLine })))
    expect(stopping.safeDetail).toEqual(Option.some("The inference engine stopped unexpectedly (exit status 134)."))
  })

  it("killed by a signal, leaves no detail, so the owner's own words show instead of the command line", async () => {
    expect(exitCodeUnreadable.message).toContain("magnitude-inference")
    const stopping = await stoppedWith(icnExitDetail(exitCodeUnreadable))
    expect(stopping.safeDetail).toEqual(Option.none())
  })

  it("before it was ready, keeps the command line out of the startup failure", () => {
    const detail = acnStartupFailureDetail(Cause.fail(exitCodeUnreadable))
    expect(detail).not.toContain("magnitude-inference")
    expect(detail).not.toContain("/Users/")
  })
})
