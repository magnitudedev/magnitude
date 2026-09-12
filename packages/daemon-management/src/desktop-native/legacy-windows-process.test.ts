import { Duration, Effect, Option, Schema, type Scope } from "effect"
import { describe, expect, it } from "vitest"
import { LegacyWindowsProcesses, legacyWindowsProcessesLayer, type LegacyWindowsProcessBindings } from "./legacy-windows-process"
import { WindowsObservedProcess } from "./windows-process-observer"
const identity = Schema.decodeUnknownSync(WindowsObservedProcess)({ pid: 42, creationTime: "0123456789abcdef", executable: "C:\\Magnitude\\magnitude-service.exe", userSid: "S-1-5-21-1-2-3-1001" })
const fixture = () => {
  const handle = {}; const events: string[] = []; let retired = false
  const native: LegacyWindowsProcessBindings = {
    acquireMigrationProcess: (...args) => { expect(args).toEqual([identity.pid, identity.creationTime, identity.executable, identity.userSid]); events.push("acquire"); return handle },
    startMigrationProcessRetirement: actual => { expect(actual).toBe(handle); events.push("terminate"); retired = true },
    migrationProcessExited: actual => { expect(actual).toBe(handle); events.push("observe"); return retired },
    releaseMigrationProcess: actual => { expect(actual).toBe(handle); events.push("close") },
  }
  return { native, events }
}
const run = <A, E>(native: LegacyWindowsProcessBindings, effect: Effect.Effect<A, E, LegacyWindowsProcesses | Scope.Scope>, timeout = Duration.seconds(10)) =>
  Effect.runPromise(Effect.scoped(effect).pipe(Effect.provide(legacyWindowsProcessesLayer(native, timeout))))
describe("legacy Windows exact retirement (simulated native boundary)", () => {
  it("retains one capability through explicit retirement and releases it exactly once", async () => {
    const { native, events } = fixture()
    await run(native, Effect.gen(function* () {
      const process = yield* (yield* LegacyWindowsProcesses).acquire(identity)
      expect(yield* process.exited).toBe(false)
      expect(events).toEqual(["acquire", "observe"])
      yield* process.retire; yield* process.retire
      expect(yield* process.exited).toBe(true)
      expect(events.filter(event => event === "acquire")).toHaveLength(1)
      expect(events).not.toContain("close")
    }))
    expect(events.filter(event => event === "close")).toHaveLength(1)
  })
  it("closing an unused capability never starts termination", async () => {
    const { native, events } = fixture()
    await run(native, Effect.gen(function* () { yield* (yield* LegacyWindowsProcesses).acquire(identity) }))
    expect(events).toEqual(["acquire", "close"])
  })
  it.each([5, 13, 87])("preserves acquisition failure %s without cleanup or termination", async code => {
    const { native, events } = fixture()
    const result = await run({ ...native, acquireMigrationProcess: () => { throw { win32Code: code } } }, Effect.gen(function* () {
      return yield* Effect.either((yield* LegacyWindowsProcesses).acquire(identity))
    }))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(Option.getOrThrow(result.left.win32Code)).toBe(code)
    expect(events).toEqual([])
  })
  it("retains authority after a termination error so the same process can be retried", async () => {
    const { native, events } = fixture(); let attempts = 0
    await run({ ...native, startMigrationProcessRetirement: handle => { if (attempts++ === 0) throw { win32Code: 5 }; native.startMigrationProcessRetirement(handle) } }, Effect.gen(function* () {
      const process = yield* (yield* LegacyWindowsProcesses).acquire(identity)
      expect((yield* Effect.either(process.retire))._tag).toBe("Left")
      expect(events).not.toContain("close")
      yield* process.retire
    }))
    expect(events).toEqual(["acquire", "terminate", "observe", "close"])
  })
  it("deadline expiry is not retirement and leaves the capability retryable", async () => {
    const { native, events } = fixture(); let complete = false
    await run({ ...native, migrationProcessExited: () => complete }, Effect.gen(function* () {
      const process = yield* (yield* LegacyWindowsProcesses).acquire(identity)
      const failed = yield* Effect.either(process.retire)
      expect(failed._tag).toBe("Left")
      if (failed._tag === "Left") expect(failed.left.message).toContain("unproven")
      expect(events).not.toContain("close")
      complete = true; yield* process.retire
    }), Duration.millis(1))
    expect(events.filter(event => event === "close")).toHaveLength(1)
  })
  it.each([undefined, null, "false", 0])("rejects malformed exit observation %s", async value => {
    const { native, events } = fixture()
    const result = await run({ ...native, migrationProcessExited: () => value }, Effect.gen(function* () {
      return yield* Effect.either((yield* (yield* LegacyWindowsProcesses).acquire(identity)).retire)
    }))
    expect(result._tag).toBe("Left"); expect(events.at(-1)).toBe("close")
  })
})
