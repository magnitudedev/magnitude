import { Deferred, Effect, Fiber, Ref } from "effect"
import { describe, expect, it } from "vitest"
import { MacApplicationInstallation, MacInstallationObservationFailed, macUpdateJobIsActive, waitForMacApplicationInstallation } from "./mac-update-installation"

const executable = "/Applications/Magnitude.app/Contents/Frameworks/Squirrel.framework/Resources/ShipIt"
describe("Mac native update launch barrier", () => {
  it.each([
    ["running", true], ["spawn scheduled", true], ["not running", false],
  ])("distinguishes %s from an inactive registration", async (state, expected) => {
    expect(await Effect.runPromise(macUpdateJobIsActive(`\tstate = ${state}\n\tprogram = ${executable}\n`, executable))).toBe(expected)
  })
  it("does not wait for another bundle's installer", async () => {
    expect(await Effect.runPromise(macUpdateJobIsActive("\tstate = running\n\tprogram = /Other.app/ShipIt\n", executable))).toBe(false)
  })
  it("does not reinterpret malformed native evidence as installation absence", async () => {
    const result = await Effect.runPromise(macUpdateJobIsActive("unexpected output", executable).pipe(Effect.either))
    expect(result._tag).toBe("Left")
  })
  it("waits through active installation without mutating or supervising its job", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const observed = yield* Deferred.make<void>()
      const active = yield* Ref.make(true)
      const finished = yield* Ref.make(false)
      const waiter = yield* waitForMacApplicationInstallation("/Applications/Magnitude.app").pipe(
        Effect.provideService(MacApplicationInstallation, { isInstalling: () => Deferred.succeed(observed, undefined).pipe(Effect.zipRight(Ref.get(active))) }),
        Effect.tap(() => Ref.set(finished, true)), Effect.forkScoped,
      )
      yield* Deferred.await(observed)
      expect(yield* Ref.get(finished)).toBe(false)
      yield* Ref.set(active, false)
      yield* Fiber.join(waiter)
      expect(yield* Ref.get(finished)).toBe(true)
    })).pipe(Effect.timeout("2 seconds")))
  })
  it("propagates observation failure instead of granting launch", async () => {
    const result = await Effect.runPromise(waitForMacApplicationInstallation("/Applications/Magnitude.app").pipe(
      Effect.provideService(MacApplicationInstallation, { isInstalling: () => new MacInstallationObservationFailed({ message: "Job query denied" }) }), Effect.either,
    ))
    expect(result._tag).toBe("Left")
  })
})
