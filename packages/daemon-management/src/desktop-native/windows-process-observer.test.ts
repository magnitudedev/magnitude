import { Effect, Fiber, Option } from "effect"
import { describe, expect, it } from "vitest"
import { WindowsProcessObserver, windowsProcessObserverLayer, type WindowsProcessObservation } from "./windows-process-observer"
import { WindowsProcessId } from "@magnitudedev/utils/windows-native"

const pid = WindowsProcessId.make(42)
const details = { pid: 42, creationTime: "1234567890abcdef", executable: "C:\\Magnitude\\magnitude-service.exe", userSid: "S-1-5-21-123-456-789-1001" }
const metadata = { observedProcessDetails: () => details, snapshotProcessParents: () => [{ pid: 42, parentPid: 10 }] }
describe("Windows read-only process observation (simulated native boundary)", () => {
  it("retains one process handle across waits and releases it once with the scope", async () => {
    const handle = {}; let opens = 0, closes = 0, reads = 0
    const layer = windowsProcessObserverLayer({ ...metadata,
      observeProcess: observed => { expect(observed).toBe(pid); opens++; return handle },
      observedProcessExited: observed => { expect(observed).toBe(handle); return ++reads > 1 },
      observedProcessDetails: observed => { expect(observed).toBe(handle); return details },
      releaseObservedProcess: observed => { expect(observed).toBe(handle); closes++ },
    })
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const observer = yield* WindowsProcessObserver
      const observation = Option.getOrThrow(yield* observer.observe(pid))
      expect(yield* observation.details).toEqual(details)
      expect(yield* observer.snapshotParents).toEqual([{ pid, parentPid: 10 }])
      yield* observation.awaitExit
      expect(opens).toBe(1)
      expect(reads).toBe(2)
      expect(closes).toBe(0)
    })).pipe(Effect.provide(layer)))
    expect(closes).toBe(1)
  })

  it("preserves permission failure instead of reporting an absent process", async () => {
    let closes = 0
    const layer = windowsProcessObserverLayer({ ...metadata, observeProcess: () => { throw { win32Code: 5 } }, observedProcessExited: () => false, releaseObservedProcess: () => { closes++ } })
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const observer = yield* WindowsProcessObserver
      return yield* Effect.either(observer.observe(pid))
    })).pipe(Effect.provide(layer)))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(Option.getOrThrow(result.left.win32Code)).toBe(5)
    expect(closes).toBe(0)
  })

  it("represents absence only when native open returns null", async () => {
    const layer = windowsProcessObserverLayer({ ...metadata, observeProcess: () => null, observedProcessExited: () => { throw new Error("No handle") }, releaseObservedProcess: () => { throw new Error("No handle") } })
    const result = await Effect.runPromise(Effect.scoped(Effect.flatMap(WindowsProcessObserver, observer => observer.observe(pid))).pipe(Effect.provide(layer)))
    expect(Option.isNone(result)).toBe(true)
  })

  it("cancels waiting without process mutation and rejects reads through released handles", async () => {
    let released = false; let captured!: WindowsProcessObservation
    const layer = windowsProcessObserverLayer({ ...metadata,
      observeProcess: () => ({}), observedProcessExited: () => { if (released) throw { win32Code: 6 }; return false }, releaseObservedProcess: () => { released = true },
    })
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const observer = yield* WindowsProcessObserver
      captured = Option.getOrThrow(yield* observer.observe(pid))
      const waiting = yield* Effect.forkScoped(captured.awaitExit)
      yield* Effect.yieldNow()
      yield* Fiber.interrupt(waiting)
    })).pipe(Effect.provide(layer)))
    expect(released).toBe(true)
    const result = await Effect.runPromise(Effect.either(captured.exited))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(Option.getOrThrow(result.left.win32Code)).toBe(6)
  })

  it.each([undefined, 0, "missing"])("rejects malformed native handle %s", async value => {
    const layer = windowsProcessObserverLayer({ ...metadata, observeProcess: () => value, observedProcessExited: () => false, releaseObservedProcess: () => { throw new Error("No handle acquired") } })
    const result = await Effect.runPromise(Effect.scoped(Effect.either(Effect.flatMap(WindowsProcessObserver, observer => observer.observe(pid)))).pipe(Effect.provide(layer)))
    expect(result._tag).toBe("Left")
  })
  it.each([
    { ...details, pid: 43 }, { ...details, creationTime: "rounded timestamp" },
    { ...details, userSid: "DOMAIN\\user" }, { ...details, executable: "" },
  ])("rejects invalid or mismatched process details %#", async value => {
    let closed = false
    const layer = windowsProcessObserverLayer({ ...metadata, observeProcess: () => ({}), observedProcessExited: () => false,
      observedProcessDetails: () => value, releaseObservedProcess: () => { closed = true },
    })
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const observer = yield* WindowsProcessObserver
      return yield* Effect.either(Option.getOrThrow(yield* observer.observe(pid)).details)
    })).pipe(Effect.provide(layer)))
    expect(result._tag).toBe("Left")
    expect(closed).toBe(true)
  })
  it("preserves access failure while releasing the retained process handle", async () => {
    let closed = false
    const layer = windowsProcessObserverLayer({ ...metadata, observeProcess: () => ({}), observedProcessExited: () => false,
      observedProcessDetails: () => { throw { win32Code: 5 } }, releaseObservedProcess: () => { closed = true },
    })
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const observer = yield* WindowsProcessObserver
      return yield* Effect.either(Option.getOrThrow(yield* observer.observe(pid)).details)
    })).pipe(Effect.provide(layer)))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(Option.getOrThrow(result.left.win32Code)).toBe(5)
    expect(closed).toBe(true)
  })
  it.each([
    [{ pid: 42, parentPid: 10 }, { pid: 42, parentPid: 20 }],
    [{ pid: 42, parentPid: 42 }], [{ pid: 0, parentPid: 0 }], [{ pid: 42, parentPid: -1 }],
  ].map(rows => [rows] as const))("rejects ambiguous or malformed process-parent snapshots %#", async rows => {
    const layer = windowsProcessObserverLayer({ ...metadata, observeProcess: () => null, observedProcessExited: () => false,
      snapshotProcessParents: () => rows, releaseObservedProcess: () => {},
    })
    const result = await Effect.runPromise(Effect.either(Effect.flatMap(WindowsProcessObserver, observer => observer.snapshotParents)).pipe(Effect.provide(layer)))
    expect(result._tag).toBe("Left")
  })
})
