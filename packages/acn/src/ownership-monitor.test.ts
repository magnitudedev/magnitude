import { ProcessStartIdentitySchema } from "@magnitudedev/acn-protocol"
import {
  AcnProcessStoreInvalid,
  AcnProcessStoreUnavailable,
  type AcnOwnerRecord,
  type AcnOwnerStore,
} from "@magnitudedev/acn-protocol/coordination"
import { Deferred, Duration, Effect, Option, Ref, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import { installAcnOwnershipMonitor } from "./ownership-monitor"
import { makeAcnServiceLifecycle } from "./service-lifecycle"

const owner = (name: string, port: number): AcnOwnerRecord => ({
  pid: 42,
  processStartIdentity: ProcessStartIdentitySchema.make(`test:${name}`),
  port,
})

const unsupportedReplace: AcnOwnerStore["replaceOwner"] = () =>
  Effect.dieMessage("replaceOwner is not used by the ownership monitor")

const run = <A, E>(effect: Effect.Effect<A, E, never>) =>
  Effect.runPromise(Effect.provide(effect, TestContext.TestContext))

describe("ACN ownership monitor", () => {
  it("runs for the admitted lifetime while the complete owner remains unchanged", async () => {
    await run(Effect.scoped(Effect.gen(function* () {
      const admitted = owner("admitted", 42_001)
      const current = yield* Ref.make(Option.some(admitted))
      const lifecycle = yield* makeAcnServiceLifecycle()
      yield* installAcnOwnershipMonitor({
        current: Ref.get(current),
        replaceOwner: unsupportedReplace,
      }, admitted, lifecycle)

      yield* Effect.yieldNow()
      yield* TestClock.adjust(Duration.seconds(5))
      expect((yield* lifecycle.state)._tag).toBe("Starting")
    })))
  })

  it("stops when the complete owner changes", async () => {
    await run(Effect.scoped(Effect.gen(function* () {
      const admitted = owner("admitted", 42_001)
      const current = yield* Ref.make(Option.some(admitted))
      const observed = yield* Deferred.make<void>()
      const lifecycle = yield* makeAcnServiceLifecycle()
      yield* installAcnOwnershipMonitor({
        current: Ref.get(current).pipe(Effect.tap(() => Deferred.succeed(observed, undefined))),
        replaceOwner: unsupportedReplace,
      }, admitted, lifecycle)

      yield* Deferred.await(observed)
      yield* Ref.set(current, Option.some(owner("successor", 42_002)))
      yield* TestClock.adjust(Duration.seconds(1))
      expect(yield* lifecycle.awaitStopping).toMatchObject({ reason: "ownership-lost" })
    })))
  })

  it("stops when the owner row is removed", async () => {
    await run(Effect.scoped(Effect.gen(function* () {
      const admitted = owner("admitted", 42_001)
      const current = yield* Ref.make(Option.some(admitted))
      const observed = yield* Deferred.make<void>()
      const lifecycle = yield* makeAcnServiceLifecycle()
      yield* installAcnOwnershipMonitor({
        current: Ref.get(current).pipe(Effect.tap(() => Deferred.succeed(observed, undefined))),
        replaceOwner: unsupportedReplace,
      }, admitted, lifecycle)

      yield* Deferred.await(observed)
      yield* Ref.set(current, Option.none())
      yield* TestClock.adjust(Duration.seconds(1))
      expect(yield* lifecycle.awaitStopping).toMatchObject({ reason: "ownership-lost" })
    })))
  })

  it("fails closed when the owner store cannot produce a trustworthy snapshot", async () => {
    await run(Effect.scoped(Effect.gen(function* () {
      const admitted = owner("admitted", 42_001)
      const lifecycle = yield* makeAcnServiceLifecycle()
      yield* installAcnOwnershipMonitor({
        current: Effect.fail(new AcnProcessStoreUnavailable({
          operation: "current-owner",
          path: "test",
          message: "unavailable",
        })),
        replaceOwner: unsupportedReplace,
      }, admitted, lifecycle)

      yield* Effect.yieldNow()
      expect(yield* lifecycle.awaitStopping).toMatchObject({ reason: "fatal" })
    })))
  })

  it("fails closed when ownership monitoring cannot continue", async () => {
    await run(Effect.scoped(Effect.gen(function* () {
      const admitted = owner("admitted", 42_001)
      const lifecycle = yield* makeAcnServiceLifecycle()
      yield* installAcnOwnershipMonitor({
        current: Effect.fail(new AcnProcessStoreInvalid({
          path: "test",
          message: "malformed owner",
        })),
        replaceOwner: unsupportedReplace,
      }, admitted, lifecycle)

      yield* Effect.yieldNow()
      expect(yield* lifecycle.awaitStopping).toMatchObject({ reason: "fatal" })
    })))
  })
})
