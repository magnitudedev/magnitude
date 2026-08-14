import { describe, expect, it } from "vitest"
import { Effect, Ref } from "effect"
import { AcnSubscriptions, AcnSubscriptionsLive } from "./acn-subscriptions"
import type { AcnSubscriptionControl } from "@magnitudedev/acn-protocol"

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
        close: Effect.void,
      })
      const secondHandle = yield* subscriptions.register({
        clientId: 2,
        requestId: "2",
        sessionId: "session-b",
        emit: emit(second),
        close: Effect.void,
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
      const closed = yield* Ref.make(false)
      yield* subscriptions.terminate
      const handle = yield* subscriptions.register({
        clientId: 1,
        requestId: "late",
        emit: (control) => Ref.update(received, (all) => [...all, control]),
        close: Ref.set(closed, true),
      })
      yield* handle.unregister
      return { controls: yield* Ref.get(received), closed: yield* Ref.get(closed) }
    }).pipe(Effect.provide(AcnSubscriptionsLive))
    const controls = await Effect.runPromise(Effect.scoped(program))

    expect(controls.controls).toEqual([])
    expect(controls.closed).toBe(true)
  })

  it("keeps replacement ownership when an older same-key handle unregisters", async () => {
    const program = Effect.gen(function* () {
      const subscriptions = yield* AcnSubscriptions
      const first = yield* Ref.make<AcnSubscriptionControl[]>([])
      const replacement = yield* Ref.make<AcnSubscriptionControl[]>([])
      const firstHandle = yield* subscriptions.register({
        clientId: 1,
        requestId: "duplicate",
        emit: (control) => Ref.update(first, (all) => [...all, control]),
        close: Effect.void,
      })
      yield* subscriptions.register({
        clientId: 1,
        requestId: "duplicate",
        emit: (control) => Ref.update(replacement, (all) => [...all, control]),
        close: Effect.void,
      })

      yield* firstHandle.unregister
      yield* subscriptions.terminate
      return {
        first: yield* Ref.get(first),
        replacement: yield* Ref.get(replacement),
      }
    }).pipe(Effect.provide(AcnSubscriptionsLive))
    const result = await Effect.runPromise(Effect.scoped(program))

    expect(result.first).toEqual([])
    expect(result.replacement).toEqual([
      { _tag: "terminated", reason: "acn-shutdown" },
    ])
  })
})
