import { describe, expect, it } from "vitest"
import { Effect, Ref } from "effect"
import { AcnSubscriptions, AcnSubscriptionsLive } from "./acn-subscriptions"
import type { AcnSubscriptionControl } from "@magnitudedev/protocol"

describe("AcnSubscriptions", () => {
  it("targets session suspension and broadcasts ACN termination", async () => {
    const program = Effect.gen(function* () {
      const subscriptions = yield* AcnSubscriptions
      const first = yield* Ref.make<AcnSubscriptionControl[]>([])
      const second = yield* Ref.make<AcnSubscriptionControl[]>([])
      const emit = (target: Ref.Ref<AcnSubscriptionControl[]>) =>
        (control: AcnSubscriptionControl) => Ref.update(target, (all) => [...all, control])

      const firstHandle = yield* subscriptions.register({
        clientId: 1,
        requestId: "1",
        sessionId: "session-a",
        emit: emit(first),
      })
      const secondHandle = yield* subscriptions.register({
        clientId: 2,
        requestId: "2",
        sessionId: "session-b",
        emit: emit(second),
      })

      yield* subscriptions.suspendSession("session-a")
      yield* subscriptions.terminate
      yield* firstHandle.unregister
      yield* secondHandle.unregister
      const firstControls = yield* Ref.get(first)
      const secondControls = yield* Ref.get(second)
      return [firstControls, secondControls] as const
    }).pipe(Effect.provide(AcnSubscriptionsLive))
    const result = await Effect.runPromise(Effect.scoped(program))

    expect(result[0]).toEqual([
      { _tag: "suspended", reason: "session-offloaded" },
      { _tag: "terminated", reason: "acn-shutdown" },
    ])
    expect(result[1]).toEqual([
      { _tag: "terminated", reason: "acn-shutdown" },
    ])
  })

  it("terminates subscriptions admitted after shutdown begins", async () => {
    const program = Effect.gen(function* () {
      const subscriptions = yield* AcnSubscriptions
      const received = yield* Ref.make<AcnSubscriptionControl[]>([])
      yield* subscriptions.terminate
      const handle = yield* subscriptions.register({
        clientId: 1,
        requestId: "late",
        emit: (control) => Ref.update(received, (all) => [...all, control]),
      })
      yield* handle.unregister
      return yield* Ref.get(received)
    }).pipe(Effect.provide(AcnSubscriptionsLive))
    const controls = await Effect.runPromise(Effect.scoped(program))

    expect(controls).toEqual([{ _tag: "terminated", reason: "acn-shutdown" }])
  })
})
