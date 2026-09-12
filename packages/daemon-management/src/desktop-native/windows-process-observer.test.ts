import { Effect, Fiber, Option } from "effect"
import { describe, expect, it } from "vitest"
import { WindowsProcessObserver, windowsProcessObserverLayer, type WindowsProcessObservation } from "./windows-process-observer"
import { WindowsProcessId } from "@magnitudedev/utils/windows-native"

const pid = WindowsProcessId.make(42)
describe("Windows read-only process observation (simulated native boundary)", () => {
  it("retains one process handle across waits and releases it once with the scope", async () => {
    const handle = {}; let opens = 0, closes = 0, reads = 0
    const layer = windowsProcessObserverLayer({
      observeProcess: observed => { expect(observed).toBe(pid); opens++; return handle },
      observedProcessExited: observed => { expect(observed).toBe(handle); return ++reads > 1 },
      releaseObservedProcess: observed => { expect(observed).toBe(handle); closes++ },
    })
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const observer = yield* WindowsProcessObserver
      const observation = Option.getOrThrow(yield* observer.observe(pid))
      yield* observation.awaitExit
      expect(opens).toBe(1)
      expect(reads).toBe(2)
      expect(closes).toBe(0)
    })).pipe(Effect.provide(layer)))
    expect(closes).toBe(1)
  })

  it("preserves permission failure instead of reporting an absent process", async () => {
    let closes = 0
    const layer = windowsProcessObserverLayer({ observeProcess: () => { throw { win32Code: 5 } }, observedProcessExited: () => false, releaseObservedProcess: () => { closes++ } })
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const observer = yield* WindowsProcessObserver
      return yield* Effect.either(observer.observe(pid))
    })).pipe(Effect.provide(layer)))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(Option.getOrThrow(result.left.win32Code)).toBe(5)
    expect(closes).toBe(0)
  })

  it("represents absence only when native open returns null", async () => {
    const layer = windowsProcessObserverLayer({ observeProcess: () => null, observedProcessExited: () => { throw new Error("No handle") }, releaseObservedProcess: () => { throw new Error("No handle") } })
    const result = await Effect.runPromise(Effect.scoped(Effect.flatMap(WindowsProcessObserver, observer => observer.observe(pid))).pipe(Effect.provide(layer)))
    expect(Option.isNone(result)).toBe(true)
  })

  it("cancels waiting without process mutation and rejects reads through released handles", async () => {
    let released = false; let captured!: WindowsProcessObservation
    const layer = windowsProcessObserverLayer({
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
    const layer = windowsProcessObserverLayer({ observeProcess: () => value, observedProcessExited: () => false, releaseObservedProcess: () => { throw new Error("No handle acquired") } })
    const result = await Effect.runPromise(Effect.scoped(Effect.either(Effect.flatMap(WindowsProcessObserver, observer => observer.observe(pid)))).pipe(Effect.provide(layer)))
    expect(result._tag).toBe("Left")
  })
})
