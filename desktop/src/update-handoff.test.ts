import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { Effect } from "effect"
import { MacApplicationInstallation, MacInstallationObservationFailed } from "@magnitudedev/daemon-management/desktop-native"
import { describe, expect, it } from "vitest"
import { makeUpdateHandoff } from "./update-handoff"

const fixture = async (test: (directory: string) => Promise<void>) => {
  const directory = await mkdtemp(join(tmpdir(), "magnitude-handoff-"))
  try { await test(directory) } finally { await rm(directory, { recursive: true, force: true }) }
}
const make = (directory: string, isInstalling: MacApplicationInstallation["isInstalling"]) => Effect.runPromise(
  makeUpdateHandoff(directory, "/Applications/Magnitude.app").pipe(Effect.provideService(MacApplicationInstallation, { isInstalling })),
)
describe("application update startup handoff", () => {
  it("normal startup does not query or create an update job", () => fixture(async directory => {
    const handoff = await make(directory, () => Effect.die("must not query"))
    expect(await Effect.runPromise(handoff.inspect("1.0.0"))).toEqual({ _tag: "Continue" })
  }))
  it("defers an old app only while the recorded bundle's native installer is active", () => fixture(async directory => {
    const handoff = await make(directory, bundle => { expect(bundle).toBe("/Applications/Magnitude.app"); return Effect.succeed(true) })
    await Effect.runPromise(handoff.record("2.0.0"))
    expect(await Effect.runPromise(handoff.inspect("1.0.0"))).toEqual({ _tag: "Defer" })
    expect(JSON.parse(await readFile(join(directory, "update-handoff.json"), "utf8")).version).toBe("2.0.0")
  }))
  it.each(["2.0.0", "3.0.0"])("admits completed or superseding version %s without waiting for its relaunch helper", version => fixture(async directory => {
    const handoff = await make(directory, () => Effect.die("must not wait on our own relaunch"))
    await Effect.runPromise(handoff.record("2.0.0"))
    expect(await Effect.runPromise(handoff.inspect(version))).toEqual({ _tag: "Continue" })
    await expect(readFile(join(directory, "update-handoff.json"))).rejects.toMatchObject({ code: "ENOENT" })
  }))
  it("permits recovery with a visible failure when native installation has stopped", () => fixture(async directory => {
    const handoff = await make(directory, () => Effect.succeed(false))
    await Effect.runPromise(handoff.record("2.0.0"))
    expect(await Effect.runPromise(handoff.inspect("1.0.0"))).toMatchObject({ _tag: "Failed", message: expect.stringContaining("did not finish") })
  }))
  it("cannot grant startup on failed native observation", () => fixture(async directory => {
    const handoff = await make(directory, () => new MacInstallationObservationFailed({ message: "Native job unavailable" }))
    await Effect.runPromise(handoff.record("2.0.0"))
    expect((await Effect.runPromise(handoff.inspect("1.0.0").pipe(Effect.either)))._tag).toBe("Left")
  }))
  it("preserves corrupt evidence instead of treating it as a successful update", () => fixture(async directory => {
    const handoff = await make(directory, () => Effect.die("must not query"))
    const path = join(directory, "update-handoff.json")
    await writeFile(path, '{"version":"invalid"}')
    expect((await Effect.runPromise(handoff.inspect("1.0.0").pipe(Effect.either)))._tag).toBe("Left")
    expect(await readFile(path, "utf8")).toBe('{"version":"invalid"}')
  }))
})
