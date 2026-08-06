import { ClientIdSchema } from "@magnitudedev/acn-protocol"
import { Cause, Deferred, Duration, Effect, Exit, Fiber, Ref, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import { makeClientLeaseManager } from "./client-lease-manager"
import {
  ModelResidencyPolicy,
  ModelResidencyPolicyUnavailable,
  type ModelResidencyPolicy as ModelResidencyPolicyService,
} from "./model-residency-policy"
import { AcnServiceLifecycle, makeAcnServiceLifecycle } from "./service-lifecycle"

const clientA = ClientIdSchema.make("client-a")
const clientB = ClientIdSchema.make("client-b")

const run = <A, E>(effect: Effect.Effect<A, E, never>) =>
  Effect.runPromise(Effect.provide(effect, TestContext.TestContext))

const makeHarness = (leaseTimeout: Duration.DurationInput = Duration.seconds(35)) =>
  Effect.gen(function* () {
    const transitions = yield* Ref.make<ReadonlyArray<boolean>>([])
    const policy: ModelResidencyPolicyService = {
      setConnected: (connected) => Ref.update(transitions, (current) => [...current, connected]),
    }
    const lifecycle = yield* makeAcnServiceLifecycle(Duration.minutes(30))
    yield* lifecycle.becomeReady(Effect.die("unused RPC"))
    const manager = yield* makeClientLeaseManager(leaseTimeout).pipe(
      Effect.provideService(AcnServiceLifecycle, lifecycle),
      Effect.provideService(ModelResidencyPolicy, policy)
    )
    return { lifecycle, manager, transitions }
  })

describe("ClientLeaseManager", () => {
  it("counts exact clients and publishes only first/final transitions", async () => {
    await run(
      Effect.scoped(
        Effect.gen(function* () {
          const harness = yield* makeHarness()
          expect((yield* harness.manager.renew(clientA)).connectedClientCount).toBe(1)
          expect((yield* harness.manager.renew(clientA)).connectedClientCount).toBe(1)
          expect((yield* harness.manager.renew(clientB)).connectedClientCount).toBe(2)
          expect((yield* harness.manager.release(clientA)).connectedClientCount).toBe(1)
          expect((yield* harness.manager.release(clientB)).connectedClientCount).toBe(0)
          expect((yield* harness.manager.release(clientB)).connectedClientCount).toBe(0)
          expect(yield* Ref.get(harness.transitions)).toEqual([true, false])
        })
      )
    )
  })

  it("expires 35 seconds after the last accepted renewal", async () => {
    await run(
      Effect.scoped(
        Effect.gen(function* () {
          const harness = yield* makeHarness()
          yield* harness.manager.renew(clientA)
          yield* TestClock.adjust(Duration.seconds(15))
          yield* harness.manager.renew(clientA)
          yield* TestClock.adjust(Duration.seconds(34))
          expect((yield* harness.manager.release(clientB)).connectedClientCount).toBe(1)
          yield* TestClock.adjust(Duration.seconds(1))
          yield* Effect.yieldNow()
          expect((yield* harness.manager.release(clientB)).connectedClientCount).toBe(0)
          expect(yield* Ref.get(harness.transitions)).toEqual([true, false])
        })
      )
    )
  })

  it("retains ACN while connected and starts a fresh 30-minute interval on final release", async () => {
    await run(
      Effect.scoped(
        Effect.gen(function* () {
          const harness = yield* makeHarness(Duration.minutes(31))
          yield* harness.manager.renew(clientA)
          yield* TestClock.adjust(Duration.minutes(30))
          expect((yield* harness.lifecycle.state)._tag).toBe("Ready")

          yield* harness.manager.release(clientA)
          yield* TestClock.adjust(Duration.minutes(30).pipe(Duration.subtract(Duration.millis(1))))
          expect((yield* harness.lifecycle.state)._tag).toBe("Ready")
          yield* TestClock.adjust(Duration.millis(1))
          expect((yield* harness.lifecycle.awaitStopping).reason).toBe("idle")
        })
      )
    )
  })

  it("serializes a renewal racing the expiry boundary", async () => {
    await run(
      Effect.scoped(
        Effect.gen(function* () {
          const harness = yield* makeHarness()
          yield* harness.manager.renew(clientA)
          yield* TestClock.adjust(Duration.seconds(35).pipe(Duration.subtract(Duration.nanos(1n))))
          const renewal = yield* harness.manager.renew(clientA).pipe(Effect.fork)
          yield* TestClock.adjust(Duration.nanos(1n))
          yield* Fiber.join(renewal)
          expect((yield* harness.manager.release(clientB)).connectedClientCount).toBe(1)
          yield* TestClock.adjust(Duration.seconds(35))
          yield* Effect.yieldNow()
          expect((yield* harness.manager.release(clientB)).connectedClientCount).toBe(0)
        })
      )
    )
  })

  it("fails closed without committing the first lease when ICN policy is unavailable", async () => {
    await run(
      Effect.scoped(
        Effect.gen(function* () {
          const lifecycle = yield* makeAcnServiceLifecycle(Duration.minutes(30))
          yield* lifecycle.becomeReady(Effect.die("unused RPC"))
          const manager = yield* makeClientLeaseManager().pipe(
            Effect.provideService(AcnServiceLifecycle, lifecycle),
            Effect.provideService(ModelResidencyPolicy, {
              setConnected: () =>
                Effect.fail(
                  new ModelResidencyPolicyUnavailable({
                    operation: "connect",
                    message: "ICN unavailable",
                  })
                ),
            })
          )

          const renewal = yield* Effect.exit(manager.renew(clientA))
          expect(Exit.isFailure(renewal) && Cause.isInterruptedOnly(renewal.cause)).toBe(true)
          expect((yield* manager.release(clientB)).connectedClientCount).toBe(0)
          expect((yield* lifecycle.awaitStopping).reason).toBe("fatal")
        })
      )
    )
  })

  it("preserves the bounded policy timeout inside the atomic lease transition", async () => {
    await run(
      Effect.scoped(
        Effect.gen(function* () {
          const lifecycle = yield* makeAcnServiceLifecycle(Duration.minutes(30))
          yield* lifecycle.becomeReady(Effect.die("unused RPC"))
          const manager = yield* makeClientLeaseManager().pipe(
            Effect.provideService(AcnServiceLifecycle, lifecycle),
            Effect.provideService(ModelResidencyPolicy, {
              setConnected: () =>
                Effect.never.pipe(
                  Effect.timeout(Duration.seconds(2)),
                  Effect.mapError(() =>
                    new ModelResidencyPolicyUnavailable({
                      operation: "connect",
                      message: "ICN timed out",
                    })
                  )
                ),
            })
          )

          const renewal = yield* manager.renew(clientA).pipe(Effect.fork)
          yield* TestClock.adjust(Duration.seconds(2))
          const exit = yield* Fiber.await(renewal)

          expect(Exit.isFailure(exit) && Cause.isInterruptedOnly(exit.cause)).toBe(true)
          expect((yield* manager.release(clientB)).connectedClientCount).toBe(0)
          expect((yield* lifecycle.awaitStopping).reason).toBe("fatal")
        })
      )
    )
  })

  it("commits the matching lease state when cancellation arrives during policy acknowledgement", async () => {
    await run(
      Effect.scoped(
        Effect.gen(function* () {
          const policyEntered = yield* Deferred.make<void>()
          const acknowledgePolicy = yield* Deferred.make<void>()
          const lifecycle = yield* makeAcnServiceLifecycle(Duration.minutes(30))
          yield* lifecycle.becomeReady(Effect.die("unused RPC"))
          const manager = yield* makeClientLeaseManager().pipe(
            Effect.provideService(AcnServiceLifecycle, lifecycle),
            Effect.provideService(ModelResidencyPolicy, {
              setConnected: () =>
                Deferred.succeed(policyEntered, undefined).pipe(
                  Effect.zipRight(Deferred.await(acknowledgePolicy))
                ),
            })
          )

          const renewal = yield* manager.renew(clientA).pipe(Effect.fork)
          yield* Deferred.await(policyEntered)
          const cancellation = yield* Fiber.interrupt(renewal).pipe(Effect.fork)
          yield* Deferred.succeed(acknowledgePolicy, undefined)
          yield* Fiber.join(cancellation)

          expect((yield* manager.release(clientB)).connectedClientCount).toBe(1)
        })
      )
    )
  })
})
