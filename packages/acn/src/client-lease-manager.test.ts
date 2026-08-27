import { ClientIdSchema } from "@magnitudedev/acn-protocol"
import { Duration, Effect, Fiber, Ref, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import { makeClientLeaseManager } from "./client-lease-manager"
import {
  ModelResidencyPolicy,
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
    const lifecycle = yield* makeAcnServiceLifecycle()
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

  it("does not stop ACN when the final RPC client lease is released", async () => {
    await run(
      Effect.scoped(
        Effect.gen(function* () {
          const harness = yield* makeHarness(Duration.minutes(31))
          yield* harness.manager.renew(clientA)
          yield* harness.manager.release(clientA)
          yield* TestClock.adjust(Duration.hours(1))
          expect((yield* harness.lifecycle.state)._tag).toBe("Ready")
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

})
