import { Effect, Schema } from "effect"
import { spawn } from "node:child_process"
import { expect, test } from "vitest"
import { ApplicationIdentity } from "../src/application-identity"
import { DesktopDriver } from "../src/desktop-driver"
import { AssertionFailure } from "../src/domain"
import { verifyUpdateBaseline } from "../src/suites/update"

for (const mode of ["valid", "wrong-initial-version", "wrong-reopened-version", "live-service", "reused-owner", "lost-theme"] as const) {
  test(`update baseline requires an older app with persisted state: ${mode}`, async () => {
    const child = spawn(process.execPath, ["-e", "process.exit(0)"], { stdio: "ignore" })
    await new Promise<void>((resolve, reject) => {
      child.once("error", reject)
      child.once("exit", code => code === 0 ? resolve() : reject(new Error(`Child exited ${code}`)))
    })
    const events: string[] = []
    let launches = 0
    const unused = () => Effect.dieMessage("Unexpected baseline operation")
    const driver = Effect.sync(() => {
      const cycle = ++launches
      const event = (name: string) => Effect.sync(() => { events.push(`${cycle}:${name}`) })
      return {
        updates: { action: unused, wait: unused, automatic: enabled => { expect(enabled).toBe(false); return event("manual") } },
        ready: () => event("ready"),
        host: () => Effect.succeed((mode === "wrong-initial-version" && cycle === 1) || (mode === "wrong-reopened-version" && cycle === 2) ? "9.9.9" : "0.1.3"),
        identity: () => Effect.sync(() => Schema.decodeUnknownSync(ApplicationIdentity)({
          applicationPid: mode === "reused-owner" ? 100 : 100 + cycle,
          servicePid: mode === "live-service" ? process.pid : child.pid!, serviceInstance: `service-${cycle}`,
        })),
        theme: value => { expect(value).toBe("dark"); return event("theme") },
        verifyTheme: value => { expect(value).toBe("dark"); return mode === "lost-theme"
          ? Effect.fail(new AssertionFailure({ message: "Fixture setting was not retained" })) : event("retained") },
        quit: () => event("quit"), navigate: unused, serviceFailure: unused, search: unused, details: unused,
        download: unused, load: unused, connect: unused, connectionFailure: unused, disconnect: unused,
        screenshot: unused, text: unused, restartForUpdate: unused, chrome: unused,
      } satisfies DesktopDriver
    })
    const result = await Effect.runPromise(verifyUpdateBaseline({ driver, stop: Effect.sync(() => { events.push("stop") }) }, "0.1.3").pipe(Effect.either))
    expect(result._tag).toBe(mode === "valid" ? "Right" : "Left")
    if (mode === "valid") expect(events).toEqual(["1:ready", "1:theme", "1:manual", "1:quit", "stop", "2:ready", "2:retained"])
    if (mode === "wrong-initial-version") expect(events).toEqual(["1:ready"])
    if (mode === "live-service") { expect(launches).toBe(1); expect(() => process.kill(process.pid, 0)).not.toThrow() }
  })
}
