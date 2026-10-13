import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { describe, expect, it } from "vitest"
import { serveStopSignals, statusIgnoresHangup } from "./serve-signals"

describe("serve stop signals", () => {
  it("reads an inherited ignored hang-up from Linux process status", () => {
    expect(statusIgnoresHangup("Name:\tbun\nSigIgn:\t0000000001001001\nSigCgt:\t0\n")).toBe(true)
    expect(statusIgnoresHangup("Name:\tbun\nSigIgn:\t0000000001001000\nSigCgt:\t0\n")).toBe(false)
    expect(statusIgnoresHangup("Name:\tbun\n")).toBe(false)
  })

  it("keeps platform stop signals outside Linux", async () => {
    const run = (platform: NodeJS.Platform) => Effect.runPromise(serveStopSignals(platform).pipe(Effect.provide(BunContext.layer)))
    expect(await run("win32")).toEqual(["SIGINT", "SIGTERM", "SIGBREAK"])
    expect(await run("darwin")).toEqual(["SIGINT", "SIGTERM", "SIGHUP"])
  })
})
